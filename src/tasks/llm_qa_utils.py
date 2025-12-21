# src/tasks/llm_qa_utils.py
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Iterable, Iterator, Optional, Set, TextIO, Tuple

from neo4j import GraphDatabase

from src.utils.json_utils import safe_json_loads
from src.utils.log_utils import log, warn, err, debug


# -------------------------
# Filesystem / JSONL utilities
# -------------------------

def ensure_parent_dir(path: str) -> None:
    """Create the parent directory of `path` if it does not already exist."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """Stream-read a JSONL file and yield one parsed dict per non-empty line.

    Raises if a non-empty line is not valid JSON. This is intentional to surface
    data corruption early. If you want best-effort behavior, add a try/except here.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s)


def write_jsonl(f: TextIO, obj: Dict[str, Any]) -> None:
    """Append one JSON object as a single line to an open file handle."""
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_seen_source_chain_ids(output_jsonl: str) -> Set[str]:
    """Return a set of `source_chain_id` already present in an output QA JSONL.

    Purpose: dedupe across repeated runs so we don't regenerate QA for the same source chain.
    """
    if not output_jsonl or not os.path.exists(output_jsonl):
        return set()
    seen: Set[str] = set()
    for obj in iter_jsonl(output_jsonl):
        cid = obj.get("source_chain_id")
        if isinstance(cid, str) and cid:
            seen.add(cid)
    return seen


# Backwards-compatible alias for older callsites that used a different name.
# Prefer load_seen_source_chain_ids moving forward.
load_existing_chain_ids = load_seen_source_chain_ids


# -------------------------
# Per-k bookkeeping / filtering helpers
# -------------------------

def init_per_k_written(k_list: Iterable[int]) -> Dict[int, int]:
    """Initialize a per-k counter dict for limit enforcement."""
    return {int(k): 0 for k in k_list}


def k_set(k_list: Iterable[int]) -> Set[int]:
    """Normalize a k list to a set of ints."""
    return {int(k) for k in k_list}


def should_take_item(
    item: Dict[str, Any],
    *,
    allowed_k: Set[int],
    per_k_written: Dict[int, int],
    limit_items: Optional[int],
    seen_source_chain_ids: Set[str],
) -> bool:
    """Decide whether an input chain item should be processed by a QA task.

    Filters:
      1) item["k"] must be in allowed_k
      2) if limit_items is not None: per_k_written[item["k"]] must be < limit_items
      3) item["chain_id"] must not already be in seen_source_chain_ids
         (because it will become source_chain_id in the QA output)
    """
    if "k" not in item or "chain_id" not in item:
        return False

    try:
        item_k = int(item["k"])
    except Exception:
        return False

    if item_k not in allowed_k:
        return False

    if limit_items is not None:
        if per_k_written.get(item_k, 0) >= int(limit_items):
            return False

    src_chain_id = item.get("chain_id")
    if isinstance(src_chain_id, str) and src_chain_id in seen_source_chain_ids:
        return False

    return True


def all_k_full(per_k_written: Dict[int, int], k_list: Iterable[int], limit_items: Optional[int]) -> bool:
    """Return True if all k in k_list have reached limit_items (when limit_items is not None)."""
    if limit_items is None:
        return False
    lim = int(limit_items)
    return all(per_k_written.get(int(k), 0) >= lim for k in k_list)


# -------------------------
# Logging helper
# -------------------------

def one_line(s: str, max_len: int = 120) -> str:
    """Compact preview for logs: single line, truncated."""
    s = (s or "").replace("\n", " ").replace("\r", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


# -------------------------
# Enrichment: fetch chunk spans from Neo4j (optional)
# -------------------------

def enrich_from_neo4j(cfg: Dict[str, Any], ctx: Dict[str, Any]) -> None:
    """Enrich ctx with per-chunk char spans from chunk nodes stored in Neo4j.

    Assumes graph extraction inserted chunk nodes with:
      - c.book_id
      - c.chunk_id
      - c.char_start
      - c.char_end

    Requires ctx:
      - ctx["book_id"]
      - ctx["chunks_in_chain_order"]

    Populates:
      - ctx["chunk_char_spans"]: list aligned with ctx["chunks_in_chain_order"]
      - ctx["chain_char_span"]: {"char_start", "char_end", "char_len"}
    """
    neo = cfg["neo4j"]
    driver = GraphDatabase.driver(neo["uri"], auth=(neo["username"], neo["password"]))

    chunk_ids = ctx["chunks_in_chain_order"]

    query = """
    MATCH (c)
    WHERE c.book_id = $book_id AND c.chunk_id IN $chunk_ids
    RETURN c.chunk_id AS chunk_id, c.char_start AS char_start, c.char_end AS char_end
    """

    spans_by_cid: Dict[Any, Dict[str, Any]] = {}
    with driver.session(database=neo["database"]) as session:
        rows = session.run(query, {"book_id": ctx["book_id"], "chunk_ids": chunk_ids})
        for r in rows:
            cid = r["chunk_id"]
            spans_by_cid[cid] = {
                "chunk_id": cid,
                "char_start": r["char_start"],
                "char_end": r["char_end"],
            }

    driver.close()

    # NOTE: this will KeyError if a cid is missing in Neo4j; that is usually desirable
    # because it indicates inconsistency between sampled chains and graph ingestion.
    ordered_spans = [spans_by_cid[cid] for cid in chunk_ids]
    ctx["chunk_char_spans"] = ordered_spans

    char_starts = [s["char_start"] for s in ordered_spans]
    char_ends = [s["char_end"] for s in ordered_spans]
    cs = min(char_starts)
    ce = max(char_ends)
    ctx["chain_char_span"] = {"char_start": cs, "char_end": ce, "char_len": ce - cs}


# -------------------------
# QA validation (shared across QA tasks)
# -------------------------

def validate_qa(
    question: str,
    answer: str,
    final_answer: str,
    start_entity: str,
    k: int,
    min_question_len: int,
) -> Optional[str]:
    """Return None if valid, else a short failure reason string."""
    question = (question or "").strip()
    answer = (answer or "").strip()

    if question == "" or answer == "":
        return "empty_question_or_answer"

    if final_answer and final_answer in question:
        return "final_answer_leaked_in_question"

    if final_answer and final_answer not in answer:
        return "final_answer_missing_in_answer"
    
    # if start_entity != "" and len(start_entity) >= 2 and start_entity not in question:
    #     return "question_not_anchored_on_start_entity"

    # if k >= 2 and len(question) < int(min_question_len):
    #     return "question_too_short_for_multihop"

    return None


# -------------------------
# Core LLM run helper (retry + parse + validate)
# -------------------------

def run_llm_qa_with_retries(
    *,
    task_name: str,
    run_debug: bool,
    llm: Any,
    prompt: str,
    ctx: Dict[str, Any],
    max_retries: int,
    retry_backoff_s: float,
    min_question_len: int,
    item_index: int,
    k_now: int,
    src_chain_id: Optional[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Run LLM -> parse JSON -> validate -> retry. Returns (parsed_json, last_err).

    Expects the model response to be JSON with at least:
      - "question"
      - "answer"

    Validation uses ctx fields:
      - ctx["final_answer"]
      - ctx["start_entity"]
      - ctx["k"]
    """
    parsed: Optional[Dict[str, Any]] = None
    last_err: Optional[str] = None

    for attempt in range(int(max_retries) + 1):
        attempt_str = f"{attempt+1}/{int(max_retries)+1}"
        try:
            debug(
                f"[{task_name}] in={item_index} k={k_now} chain_id={src_chain_id} LLM attempt {attempt_str}",
                run_debug,
            )
            resp = llm.complete(prompt)
            resp_str = str(resp)

            parsed = safe_json_loads(resp_str)
            if not isinstance(parsed, dict):
                raise ValueError(f"safe_json_loads did not return dict; got {type(parsed)}")

            q = str(parsed.get("question", "")).strip()
            a = str(parsed.get("answer", "")).strip()

            reason = validate_qa(
                question=q,
                answer=a,
                final_answer=str(ctx.get("final_answer", "")),
                start_entity=str(ctx.get("start_entity", "")),
                k=int(ctx.get("k", k_now)),
                min_question_len=int(min_question_len),
            )

            if reason is None:
                log(
                    f"[{task_name}] OK in={item_index} k={k_now} chain_id={src_chain_id} "
                    f"q='{one_line(q)}' a='{one_line(a)}'"
                )
                return parsed, None

            last_err = f"validation_failed:{reason}"
            warn(
                f"[{task_name}] FAIL in={item_index} k={k_now} chain_id={src_chain_id} "
                f"attempt={attempt_str} reason={reason} "
                f"q='{one_line(q)}' a='{one_line(a)}'"
            )
            parsed = None

        except Exception as e:
            last_err = f"exception:{repr(e)}"
            warn(
                f"[{task_name}] FAIL in={item_index} k={k_now} chain_id={src_chain_id} "
                f"attempt={attempt_str} error={repr(e)}"
            )
            parsed = None

        if attempt < int(max_retries):
            time.sleep(float(retry_backoff_s))

    err(f"[{task_name}] GIVEUP in={item_index} k={k_now} chain_id={src_chain_id} last_err={last_err}")
    return None, last_err
