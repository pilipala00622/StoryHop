from __future__ import annotations

import os
from typing import Any, Dict, Iterator, Optional, Set

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


def _witness_for_item(
    item: Dict[str, Any],
    *,
    allowed_rel_types: Set[str],
    answer_max_chars: int,
) -> Optional[Dict[str, Any]]:
    for i, st in enumerate(item["chain"]["steps"]):
        rel_type = _norm(st["relation"]["type"])
        if rel_type not in allowed_rel_types:
            continue

        entity = _norm(st["source"]["name"])
        value = _norm(st["target"]["name"])
        if not entity or not value or len(value) > answer_max_chars:
            continue

        return {
            "step_idx": i,
            "entity": entity,
            "value": value,
            "rel_type": rel_type,
            "evidence": _norm(st["evidence"]),
            "evidence_chunk_id": int(st["chunk_id"]),
        }
    return None


def generate_attribute_lookup_items(
    *,
    input_chains_jsonl: str,
    allowed_rel_types: Set[str],
    answer_max_chars: int,
) -> Iterator[Dict[str, Any]]:
    for item in iter_jsonl(input_chains_jsonl):
        w = _witness_for_item(item, allowed_rel_types=allowed_rel_types, answer_max_chars=answer_max_chars)
        if w is None:
            continue
        yield {
            "task": "attribute_lookup",
            "book_id": item["book_id"],
            "chain_id": item["chain_id"],
            "k": item["k"],
            "chain": item["chain"],
            "full_query": item["full_query"],
            "witness": w,
        }


def run_chain_gen(cfg: Dict[str, Any], task_cfg: Dict[str, Any]) -> None:
    
    gen_cfg = cfg["chain_gens"][task_cfg["chains_gen_cfg_key"]]
    if not gen_cfg["enabled"]:
        log(f"[{task_cfg['name']}:chain_gen] disabled")
        return
    input_jsonl = gen_cfg["source_input_jsonl"]
    output_jsonl = task_cfg["qa_input_jsonl"]
    allowed_rel_types = set(gen_cfg["allowed_rel_types"])
    answer_max_chars = int(gen_cfg["answer_max_chars"])
    reset = bool(cfg["run"]["reset"])

    ensure_parent_dir(output_jsonl)
    mode = "w" if reset else "a"
    seen = set() if reset else _load_seen_chain_ids(output_jsonl)

    log(
        f"[{task_cfg['name']}:chain_gen] input={input_jsonl} output={output_jsonl} "
        f"mode={mode} seen={len(seen)} allowed_rel_types={sorted(list(allowed_rel_types))} "
        f"answer_max_chars={answer_max_chars}"
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

            w = _witness_for_item(item, allowed_rel_types=allowed_rel_types, answer_max_chars=answer_max_chars)
            if w is None:
                n_skip_no_witness += 1
                continue

            out = {
                "task": "attribute_lookup",
                "book_id": item["book_id"],
                "chain_id": cid,
                "k": item["k"],
                "chain": item["chain"],
                "full_query": item["full_query"],
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
