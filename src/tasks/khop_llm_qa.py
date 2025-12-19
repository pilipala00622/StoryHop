from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, Optional, Set

from neo4j import GraphDatabase

from src.utils.json_utils import safe_json_loads
from src.utils.log_utils import log, warn, err, debug
from src.tasks.prompt_builders import build_chain_features


# -------------------------
# Enrichment: fetch chunk spans from Neo4j (optional)
# -------------------------

def enrich_from_neo4j(cfg: Dict[str, Any], ctx: Dict[str, Any]) -> None:
    """Enrich ctx with per-chunk char spans from the chunk nodes stored in Neo4j.

    Assumes extract_graph_low_level inserted chunk nodes (LlamaIndex TextNode) with:
      - c.book_id
      - c.chunk_id
      - c.char_start
      - c.char_end

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

    ordered_spans = [spans_by_cid[cid] for cid in chunk_ids]
    ctx["chunk_char_spans"] = ordered_spans

    char_starts = [s["char_start"] for s in ordered_spans]
    char_ends = [s["char_end"] for s in ordered_spans]
    cs = min(char_starts)
    ce = max(char_ends)
    ctx["chain_char_span"] = {"char_start": cs, "char_end": ce, "char_len": ce - cs}


# -------------------------
# Validation (always-on)
# -------------------------

def validate_qa(
    question: str,
    answer: str,
    final_answer: str,
    start_entity: str,
    k: int,
    min_question_len: int,
) -> Optional[str]:
    question = question.strip()
    answer = answer.strip()

    if question == "" or answer == "":
        return "empty_question_or_answer"

    if final_answer in question:
        return "final_answer_leaked_in_question"

    if final_answer not in answer:
        return "final_answer_missing_in_answer"

    if start_entity != "" and len(start_entity) >= 2 and start_entity not in question:
        return "question_not_anchored_on_start_entity"

    if k >= 2 and len(question) < min_question_len:
        return "question_too_short_for_multihop"

    return None


# -------------------------
# Helpers: JSONL dedupe + logging
# -------------------------

def _load_existing_chain_ids(output_jsonl: str) -> Set[str]:
    """Read existing output JSONL and collect already-generated source_chain_id values."""
    if not output_jsonl or not os.path.exists(output_jsonl):
        return set()

    seen: Set[str] = set()
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            cid = obj.get("source_chain_id")
            if isinstance(cid, str) and cid:
                seen.add(cid)
    return seen


def _one_line(s: str, max_len: int = 120) -> str:
    """Compact preview for logs: single line, truncated."""
    s = (s or "").replace("\n", " ").replace("\r", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


# -------------------------
# Core runner: called by DataFactory
# -------------------------

def run_task(
    cfg: Dict[str, Any],
    task_cfg: Dict[str, Any],
    prompt_builder: Callable[[Dict[str, Any]], str],
    llm: Any,
    llm_name: str,
) -> None:
    k_list = task_cfg["k"]  # list[int]
    run = cfg["run"]
    qa = run["qa"]

    input_jsonl = task_cfg["input_jsonl"]
    output_jsonl = task_cfg["output_jsonl"]
    limit_items = task_cfg["limit_items"]  # per-k cap
    if limit_items is not None and limit_items == 0:
        log(f"Skipping task {task_cfg['name']} due to limit_items=0")
        return
    language = qa["language"]
    answer_max_chars = qa["answer_max_chars"]
    min_question_len = qa["min_question_len"]

    max_retries = qa["max_retries"]
    retry_backoff_s = qa["retry_backoff_s"]
    do_enrich_from_neo4j = qa["enrich_from_neo4j"]

    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)) or ".", exist_ok=True)

    # Dedup: skip chains already in output
    seen_chain_ids = _load_existing_chain_ids(output_jsonl)

    log(f"[DataFactory:{task_cfg['name']}] input={input_jsonl} output={output_jsonl} prompt_builder={task_cfg['prompt_builder']}")
    log(f"qa.language={language} qa.answer_max_chars={answer_max_chars} qa.max_retries={max_retries} qa.enrich_from_neo4j={do_enrich_from_neo4j}")
    log(f"task.k={k_list} existing_outputs={len(seen_chain_ids)} limit_items(per-k)={limit_items} llm={llm_name}")

    n_in = 0          # lines read
    n_considered = 0  # after parsing
    n_out = 0         # newly written
    n_skip_k = 0
    n_skip_seen = 0
    n_skip_limit = 0

    # Per-k counters (counts newly written items per k)
    per_k_written: Dict[int, int] = {int(k): 0 for k in k_list}

    # APPEND instead of overwrite
    with open(input_jsonl, "r", encoding="utf-8") as fin, open(output_jsonl, "a", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if line == "":
                continue

            n_in += 1
            item = json.loads(line)
            n_considered += 1

            item_k = item.get("k")
            if item_k not in k_list:
                n_skip_k += 1
                continue

            # Per-k cap: stop generating for this k once it hits limit_items
            if limit_items is not None and per_k_written[int(item_k)] >= int(limit_items):
                n_skip_limit += 1
                continue

            src_chain_id = item.get("chain_id")
            if isinstance(src_chain_id, str) and src_chain_id in seen_chain_ids:
                n_skip_seen += 1
                continue

            chain_features = build_chain_features(item)

            ctx: Dict[str, Any] = {
                **chain_features,
                "language": language,
                "answer_max_chars": answer_max_chars,
                "min_question_len": min_question_len,
            }

            if do_enrich_from_neo4j:
                enrich_from_neo4j(cfg, ctx)

            prompt = prompt_builder(ctx)

            # Clear per-item header
            k_now = int(item_k)
            k_prog = f"{per_k_written.get(k_now, 0)}/{int(limit_items) if limit_items is not None else '∞'}"
            log(f"[{task_cfg['name']}] in={n_in} k={k_now} (written {k_prog}) chain_id={src_chain_id}")

            parsed: Optional[Dict[str, Any]] = None
            last_err: Optional[str] = None

            for attempt in range(int(max_retries) + 1):
                attempt_str = f"{attempt+1}/{int(max_retries)+1}"
                try:
                    debug(f"[{task_cfg['name']}] in={n_in} k={k_now} chain_id={src_chain_id} LLM attempt {attempt_str}", run["debug"])
                    resp = llm.complete(prompt)
                    resp_str = str(resp)

                    # Parse JSON
                    parsed = safe_json_loads(resp_str)

                    # Extract fields early for clearer logging
                    q = str(parsed.get("question", "")).strip()
                    a = str(parsed.get("answer", "")).strip()

                    reason = validate_qa(
                        question=q,
                        answer=a,
                        final_answer=ctx["final_answer"],
                        start_entity=ctx["start_entity"],
                        k=ctx["k"],
                        min_question_len=min_question_len,
                    )

                    if reason is None:
                        # Success preview
                        log(
                            f"[{task_cfg['name']}] OK in={n_in} k={k_now} chain_id={src_chain_id} "
                            f"q='{_one_line(q)}' a='{_one_line(a)}'"
                        )
                        break

                    # Validation failure: show reason + preview
                    last_err = f"validation_failed:{reason}"
                    warn(
                        f"[{task_cfg['name']}] FAIL in={n_in} k={k_now} chain_id={src_chain_id} "
                        f"attempt={attempt_str} reason={reason} "
                        f"q='{_one_line(q)}' a='{_one_line(a)}'"
                    )
                    parsed = None

                except Exception as e:
                    last_err = f"exception:{repr(e)}"
                    warn(
                        f"[{task_cfg['name']}] FAIL in={n_in} k={k_now} chain_id={src_chain_id} "
                        f"attempt={attempt_str} error={repr(e)}"
                    )
                    parsed = None

                if attempt < int(max_retries):
                    time.sleep(float(retry_backoff_s))

            if parsed is None:
                err(f"[{task_cfg['name']}] GIVEUP in={n_in} k={k_now} chain_id={src_chain_id} last_err={last_err}")
                continue

            out_item = {
                "task": task_cfg["name"],
                "book_id": item["book_id"],
                "k": ctx["k"],
                "source_chain_id": item["chain_id"],
                "question": str(parsed.get("question", "")).strip(),
                "answer": str(parsed.get("answer", "")).strip(),
                "final_answer": ctx["final_answer"],
                "prompt_builder": task_cfg["prompt_builder"],
                "llm": llm_name,
                "chain_stats": {
                    "chunks_in_chain_order": ctx["chunks_in_chain_order"],
                    "chunk_span": ctx["chunk_span"],
                    "chunk_order_monotonic_inc": ctx["chunk_order_monotonic_inc"],
                },
            }

            if do_enrich_from_neo4j:
                out_item["chain_stats"].update(
                    {
                        "chunk_char_spans": ctx["chunk_char_spans"],
                        "chain_char_span": ctx["chain_char_span"],
                    }
                )

            fout.write(json.dumps(out_item, ensure_ascii=False) + "\n")
            n_out += 1

            if isinstance(src_chain_id, str) and src_chain_id:
                seen_chain_ids.add(src_chain_id)

            per_k_written[int(item_k)] = per_k_written.get(int(item_k), 0) + 1

            if n_out % 10 == 0:
                log(f"[{task_cfg['name']}] generated {n_out} items (processed {n_in}) per_k_written={per_k_written}")

            # Optional early exit: if all k hit cap, stop scanning input
            if limit_items is not None:
                all_full = all(per_k_written[int(k)] >= int(limit_items) for k in k_list)
                if all_full:
                    break

    log(
        f"[{task_cfg['name']}] done. processed={n_in} considered={n_considered} written={n_out} "
        f"skip_k={n_skip_k} skip_seen={n_skip_seen} skip_limit={n_skip_limit} "
        f"per_k_written={per_k_written} output={output_jsonl}"
    )
