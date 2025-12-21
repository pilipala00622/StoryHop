from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set, Tuple

from src.tasks.llm_qa_utils import ensure_parent_dir, iter_jsonl, write_jsonl
from src.utils.log_utils import log


DEFAULT_ALIAS_REL_TYPES = {
    "ALIAS_OF",
    "SAME_AS",
    "HAS_ALIAS",
    "MENTION_ALIAS",
    "ALIAS",
}


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


def _format_evidence_from_steps(steps: List[Dict[str, Any]]) -> str:
    # Keep as a single “evidence snippet” for the prompt builder.
    lines: List[str] = []
    for st in steps:
        ev = _norm(st.get("evidence", ""))
        cid = st.get("chunk_id")
        if ev:
            lines.append(f"[chunk_id={cid}] {ev}")
    # de-dup while preserving order
    return "\n".join(dict.fromkeys(lines)).strip()


def _find_explicit_alias_edge(
    steps: List[Dict[str, Any]],
    alias_rel_types: Set[str],
) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """Return (alias, canonical, step) if an explicit alias edge exists."""
    for st in steps:
        rel_type = _norm((st.get("relation") or {}).get("type", ""))
        if rel_type and rel_type in alias_rel_types:
            src = _norm((st.get("source") or {}).get("name", ""))
            tgt = _norm((st.get("target") or {}).get("name", ""))
            if src and tgt and src != tgt:
                # Convention: src is alias surface; tgt is canonical entity name (you can swap if your graph differs)
                return src, tgt, st
    return None


def _witness_for_item(
    item: Dict[str, Any],
    *,
    alias_rel_types: Set[str],
) -> Optional[Dict[str, Any]]:
    steps = item["chain"]["steps"]

    found = _find_explicit_alias_edge(steps, alias_rel_types)
    if found is not None:
        alias, canonical, st = found
        ev = _format_evidence_from_steps([st])

        if not ev or not alias or not canonical:
            return None

        # Contract: answer is canonical entity name; must appear in chain (it does: st.target.name)
        return {
            "alias": alias,
            "canonical": canonical,
            "rel_type": _norm((st.get("relation") or {}).get("type", "")),
            "evidence": ev,
            "evidence_chunk_ids": [int(st["chunk_id"])],
            "selected_hops": [int(st.get("hop", 1))],
        }

    # Fallback heuristic (optional): if no alias edges exist, do not emit witness.
    # Keeping this strict avoids low-precision alias samples.
    return None


def run_chain_gen(cfg: Dict[str, Any], task_cfg: Dict[str, Any]) -> None:
    gen_cfg = cfg["chain_gens"][task_cfg["chains_gen_cfg_key"]]

    if not gen_cfg["enabled"]:
        log(f"[{task_cfg['name']}:chain_gen] skipped (enabled=false)")
        return

    input_jsonl = gen_cfg["source_input_jsonl"]
    output_jsonl = task_cfg["qa_input_jsonl"]
    reset = bool(cfg["run"]["reset"])

    alias_rel_types = set(gen_cfg.get("alias_rel_types") or list(DEFAULT_ALIAS_REL_TYPES))

    ensure_parent_dir(output_jsonl)
    mode = "w" if reset else "a"
    seen = set() if reset else _load_seen_chain_ids(output_jsonl)

    log(
        f"[{task_cfg['name']}:chain_gen] input={input_jsonl} output={output_jsonl} mode={mode} "
        f"seen={len(seen)} alias_rel_types={len(alias_rel_types)}"
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

            w = _witness_for_item(item, alias_rel_types=alias_rel_types)
            if w is None or not _norm(w.get("canonical", "")) or not _norm(w.get("alias", "")):
                n_skip_no_witness += 1
                continue

            out = {
                "task": "alias_coref",
                "book_id": item["book_id"],
                "chain_id": cid,
                "k": item["k"],  # keep source-k for per-k sampling control
                "chain": item["chain"],
                "full_query": item.get("full_query"),
                "witness": w,
                "final_answer": w["canonical"],
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
