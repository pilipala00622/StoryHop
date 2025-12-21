from __future__ import annotations

import json
from typing import Any, Callable, Dict

from src.utils.log_utils import log
from src.tasks.llm_qa_utils import (
    all_k_full,
    ensure_parent_dir,
    enrich_from_neo4j,
    init_per_k_written,
    k_set,
    load_seen_source_chain_ids,
    run_llm_qa_with_retries,
    should_take_item,
    write_jsonl,
)


def run_task(
    cfg: Dict[str, Any],
    task_cfg: Dict[str, Any],
    prompt_builder: Callable[[Dict[str, Any]], str],
    llm: Any,
    llm_name: str,
) -> None:
    run = cfg["run"]
    qa = run["qa"]

    input_jsonl = task_cfg["qa_input_jsonl"]
    output_jsonl = task_cfg["qa_output_jsonl"]
    limit_items = task_cfg["limit_items"]
    k_list = task_cfg["k"]

    ensure_parent_dir(output_jsonl)

    allowed_k = k_set(k_list)
    seen_chain_ids = load_seen_source_chain_ids(output_jsonl)
    per_k_written = init_per_k_written(k_list)

    log(
        f"[{task_cfg['name']}] input={input_jsonl} output={output_jsonl} "
        f"k={k_list} limit_items(per-k)={limit_items} existing_outputs={len(seen_chain_ids)} llm={llm_name}"
    )

    n_in = 0
    n_out = 0
    n_skip_no_witness = 0

    with open(input_jsonl, "r", encoding="utf-8") as fin, open(output_jsonl, "a", encoding="utf-8") as fout:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            n_in += 1
            item = json.loads(s)

            if not should_take_item(
                item,
                allowed_k=allowed_k,
                per_k_written=per_k_written,
                limit_items=limit_items,
                seen_source_chain_ids=seen_chain_ids,
            ):
                continue

            item_k = int(item["k"])
            src_chain_id = item["chain_id"]

            witness = item.get("witness")
            if not isinstance(witness, dict):
                n_skip_no_witness += 1
                continue

            pronoun = str(witness.get("pronoun", "")).strip()
            evidence = str(witness.get("evidence", "")).strip()
            final_answer = str(item.get("final_answer") or witness.get("answer") or "").strip()

            if not pronoun or not evidence or not final_answer:
                n_skip_no_witness += 1
                continue

            ctx: Dict[str, Any] = {
                "book_id": item["book_id"],
                "chain_id": src_chain_id,
                "language": qa["language"],
                "answer_max_chars": qa["answer_max_chars"],
                "min_question_len": qa["min_question_len"],
                "witness": witness,
                "chain": item["chain"],
                "full_query": item.get("full_query"),
                "final_answer": final_answer,
                # IMPORTANT: do not require anchor in question since answer is forbidden in question
                "start_entity": "",
                # skill-level hop count: local/single-chunk
                "k": 1,
                "pronoun": pronoun,
                "evidence": evidence,
                "evidence_chunk_id": int(witness["evidence_chunk_id"]),
            }

            if bool(qa["enrich_from_neo4j"]):
                ctx["chunks_in_chain_order"] = [int(st["chunk_id"]) for st in item["chain"]["steps"]]
                enrich_from_neo4j(cfg, ctx)

            parsed, _last_err = run_llm_qa_with_retries(
                task_name=task_cfg["name"],
                run_debug=bool(run.get("debug", False)),
                llm=llm,
                prompt=prompt_builder(ctx),
                ctx=ctx,
                max_retries=int(qa["max_retries"]),
                retry_backoff_s=float(qa["retry_backoff_s"]),
                min_question_len=int(qa["min_question_len"]),
                item_index=n_in,
                k_now=item_k,
                src_chain_id=src_chain_id,
            )
            if parsed is None:
                continue

            question = str(parsed.get("question", "")).strip()
            answer = str(parsed.get("answer", "")).strip()
            if not question or not answer:
                continue

            out_item = {
                "task": task_cfg["name"],
                "book_id": item["book_id"],
                "k": 1,
                "source_chain_id": src_chain_id,
                "question": question,
                "answer": answer,
                "final_answer": final_answer,
                "prompt_builder": task_cfg["prompt_builder"],
                "llm": llm_name,
                "chain_stats": {
                    "source_k": item_k,
                    "selected_hop": witness.get("selected_hop"),
                    "evidence_chunk_id": witness.get("evidence_chunk_id"),
                    "pronoun": witness.get("pronoun"),
                    "full_query": item.get("full_query"),
                },
            }

            if bool(qa["enrich_from_neo4j"]):
                out_item["chain_stats"].update(
                    {
                        "chunk_char_spans": ctx["chunk_char_spans"],
                        "chain_char_span": ctx["chain_char_span"],
                    }
                )

            write_jsonl(fout, out_item)
            n_out += 1
            seen_chain_ids.add(src_chain_id)
            per_k_written[item_k] = per_k_written.get(item_k, 0) + 1

            if all_k_full(per_k_written, k_list, limit_items):
                break

    log(
        f"[{task_cfg['name']}] done processed={n_in} written={n_out} "
        f"skip_no_witness={n_skip_no_witness} per_k_written={per_k_written} output={output_jsonl}"
    )
