from __future__ import annotations

import os
from typing import Any, Dict, Optional, Set

from src.tasks.llm_qa_utils import ensure_parent_dir, iter_jsonl, write_jsonl
from src.utils.log_utils import log


def _norm(x: Any) -> str:
    return str(x).replace("\n", " ").replace("\r", " ").strip()


def _load_seen_chain_ids(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    seen: Set[str] = set()
    for obj in iter_jsonl(path):
        cid = obj["chain_id"]
        if cid:
            seen.add(cid)
    return seen


def _step_matches(ev: str, *, speech_hints: Set[str], quote_hints: Set[str]) -> bool:
    return any(q in ev for q in quote_hints) or any(h in ev for h in speech_hints)


def _witness_for_item(
    item: Dict[str, Any],
    *,
    speech_hints: Set[str],
    quote_hints: Set[str],
    require_match: bool,
) -> Optional[Dict[str, Any]]:
    steps = item["chain"]["steps"]
    best = None

    for st in steps:
        ev = _norm(st["evidence"])
        if _step_matches(ev, speech_hints=speech_hints, quote_hints=quote_hints):
            best = st
            break

    if best is None:
        if require_match:
            return None
        best = steps[0]

    return {
        "actor": _norm(best["source"]["name"]),
        "rel_type": _norm(best["relation"]["type"]),
        "evidence": _norm(best["evidence"]),
        "evidence_chunk_id": int(best["chunk_id"]),
        "selected_hop": int(best.get("hop", 1)),
    }


def run_chain_gen(cfg: Dict[str, Any], task_cfg: Dict[str, Any]) -> None:
    gen_cfg = cfg["chain_gens"][task_cfg["chains_gen_cfg_key"]]

    if not gen_cfg["enabled"]:
        log(f"[{task_cfg['name']}:chain_gen] skipped (enabled=false)")
        return

    input_jsonl = gen_cfg["source_input_jsonl"]
    output_jsonl = task_cfg["qa_input_jsonl"]
    speech_hints = set(gen_cfg["speech_hints"])
    quote_hints = set(gen_cfg["quote_hints"])
    require_match = bool(gen_cfg["require_match"])
    reset = bool(cfg["run"]["reset"])

    ensure_parent_dir(output_jsonl)
    mode = "w" if reset else "a"
    seen = set() if reset else _load_seen_chain_ids(output_jsonl)

    log(
        f"[{task_cfg['name']}:chain_gen] input={input_jsonl} output={output_jsonl} mode={mode} "
        f"seen={len(seen)} require_match={require_match}"
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

            w = _witness_for_item(
                item,
                speech_hints=speech_hints,
                quote_hints=quote_hints,
                require_match=require_match,
            )
            if w is None or not w["actor"]:
                n_skip_no_witness += 1
                continue

            out = {
                "task": "who_act_what",
                "book_id": item["book_id"],
                "chain_id": cid,
                "k": item["k"],
                "chain": item["chain"],
                "full_query": item.get("full_query"),
                "witness": w,
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
