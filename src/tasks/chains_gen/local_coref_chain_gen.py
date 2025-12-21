from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set, Tuple

from src.tasks.llm_qa_utils import ensure_parent_dir, iter_jsonl, write_jsonl
from src.utils.log_utils import log


DEFAULT_PRONOUN_HINTS = [
    "他",
    "她",
    "那人",
    "此人",
    "其",
    "那位",
    "此君",
    "那家伙",
]


def _norm(x: Any) -> str:
    return str(x).replace("\n", " ").replace("\r", " ").strip()


def _load_seen_chain_ids(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    seen: Set[str] = set()
    for obj in iter_jsonl(path):
        cid = obj.get("chain_id")
        if isinstance(cid, str) and cid:
            seen.add(cid)
    return seen


def _group_steps_by_chunk(steps: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    by_cid: Dict[int, List[Dict[str, Any]]] = {}
    for st in steps:
        try:
            cid = int(st["chunk_id"])
        except Exception:
            continue
        by_cid.setdefault(cid, []).append(st)
    return by_cid


def _find_pronoun_step(
    steps_in_chunk: List[Dict[str, Any]],
    pronoun_hints: List[str],
) -> Optional[Tuple[Dict[str, Any], str]]:
    for st in steps_in_chunk:
        ev = _norm(st.get("evidence", ""))
        for p in pronoun_hints:
            if p and p in ev:
                return st, p
    return None


def _pick_entity_candidate_in_same_chunk(
    steps_in_chunk: List[Dict[str, Any]],
    chunk_text: str,
    *,
    exclude_step: Dict[str, Any],
    exclude_pronoun: str,
) -> Optional[str]:
    """Pick an entity name that:
      - comes from source/target.name of a different step
      - appears in the chunk_text (so answer is locally recoverable)
      - is not the pronoun itself
    """
    for st in steps_in_chunk:
        if st is exclude_step:
            continue

        src = _norm((st.get("source") or {}).get("name", ""))
        tgt = _norm((st.get("target") or {}).get("name", ""))

        # Prefer target name; fallback to source name
        for cand in [tgt, src]:
            if not cand:
                continue
            if exclude_pronoun and cand == exclude_pronoun:
                continue
            if cand in chunk_text:
                return cand

    return None


def _witness_for_item(
    item: Dict[str, Any],
    *,
    pronoun_hints: List[str],
) -> Optional[Dict[str, Any]]:
    steps = item["chain"]["steps"]
    by_cid = _group_steps_by_chunk(steps)

    # Require: same chunk_id contains >=2 steps (per task definition)
    for cid, steps_in_chunk in by_cid.items():
        if len(steps_in_chunk) < 2:
            continue

        # Build "single-chunk context" evidence for prompting: include all evidence lines from this chunk
        ev_lines = []
        for st in steps_in_chunk:
            ev = _norm(st.get("evidence", ""))
            if ev:
                ev_lines.append(ev)
        if not ev_lines:
            continue
        chunk_text = "\n".join(dict.fromkeys(ev_lines))  # de-dup while preserving order

        pron = _find_pronoun_step(steps_in_chunk, pronoun_hints)
        if pron is None:
            continue
        pron_step, pron_str = pron

        ans = _pick_entity_candidate_in_same_chunk(
            steps_in_chunk,
            chunk_text,
            exclude_step=pron_step,
            exclude_pronoun=pron_str,
        )
        if not ans:
            continue

        # sanity: ensure answer appears somewhere in chain (contract)
        # (it will, because it comes from source/target.name of a chain step)
        return {
            "pronoun": pron_str,
            "evidence": chunk_text,
            "evidence_chunk_id": int(cid),
            "answer": ans,
            "selected_hop": int(pron_step.get("hop", 1)),
        }

    return None


def run_chain_gen(cfg: Dict[str, Any], task_cfg: Dict[str, Any]) -> None:
    gen_cfg = cfg["chain_gens"][task_cfg["chains_gen_cfg_key"]]

    if not gen_cfg["enabled"]:
        log(f"[{task_cfg['name']}:chain_gen] skipped (enabled=false)")
        return

    input_jsonl = gen_cfg["source_input_jsonl"]
    output_jsonl = task_cfg["qa_input_jsonl"]
    reset = bool(cfg["run"]["reset"])

    pronoun_hints = list(gen_cfg.get("pronoun_hints") or DEFAULT_PRONOUN_HINTS)

    ensure_parent_dir(output_jsonl)
    mode = "w" if reset else "a"
    seen = set() if reset else _load_seen_chain_ids(output_jsonl)

    log(
        f"[{task_cfg['name']}:chain_gen] input={input_jsonl} output={output_jsonl} mode={mode} "
        f"seen={len(seen)} pronoun_hints={len(pronoun_hints)}"
    )

    n_in = 0
    n_out = 0
    n_skip_seen = 0
    n_skip_no_witness = 0

    with open(output_jsonl, mode, encoding="utf-8") as fout:
        for item in iter_jsonl(input_jsonl):
            n_in += 1
            cid = item["chain_id"]

            if not reset and cid in seen:
                n_skip_seen += 1
                continue

            w = _witness_for_item(item, pronoun_hints=pronoun_hints)
            if w is None or not _norm(w.get("answer", "")):
                n_skip_no_witness += 1
                continue

            out = {
                "task": "local_coref",
                "book_id": item["book_id"],
                "chain_id": cid,
                "k": item["k"],  # keep source-k for per-k sampling control
                "chain": item["chain"],
                "full_query": item.get("full_query"),
                "witness": w,
                "final_answer": w["answer"],
            }

            write_jsonl(fout, out)
            n_out += 1
            seen.add(cid)

            if n_out % 200 == 0:
                log(
                    f"[{task_cfg['name']}:chain_gen] wrote={n_out} processed={n_in} "
                    f"skip_seen={n_skip_seen} skip_no_witness={n_skip_no_witness}"
                )

    log(
        f"[{task_cfg['name']}:chain_gen] done processed={n_in} wrote={n_out} "
        f"skip_seen={n_skip_seen} skip_no_witness={n_skip_no_witness} output={output_jsonl}"
    )
