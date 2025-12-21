from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional

from src.utils.log_utils import log
from src.tasks.prompt_builders import build_chain_features
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
    k_list = task_cfg["k"]  # list[int]
    run = cfg["run"]
    qa = run["qa"]

    input_jsonl = task_cfg["qa_input_jsonl"]
    output_jsonl = task_cfg["qa_output_jsonl"]
    limit_items = task_cfg["limit_items"]  # per-k cap
    if limit_items is not None and int(limit_items) == 0:
        log(f"Skipping task {task_cfg['name']} due to limit_items=0")
        return

    language = qa["language"]
    answer_max_chars = qa["answer_max_chars"]
    min_question_len = qa["min_question_len"]

    max_retries = qa["max_retries"]
    retry_backoff_s = qa["retry_backoff_s"]
    do_enrich_from_neo4j = qa["enrich_from_neo4j"]

    ensure_parent_dir(output_jsonl)

    allowed_k = k_set(k_list)
    seen_chain_ids = load_seen_source_chain_ids(output_jsonl)
    per_k_written = init_per_k_written(k_list)

    log(
        f"[DataFactory:{task_cfg['name']}] input={input_jsonl} output={output_jsonl} "
        f"prompt_builder={task_cfg['prompt_builder']}"
    )
    log(
        f"qa.language={language} qa.answer_max_chars={answer_max_chars} "
        f"qa.max_retries={max_retries} qa.enrich_from_neo4j={do_enrich_from_neo4j}"
    )
    log(
        f"task.k={k_list} existing_outputs={len(seen_chain_ids)} "
        f"limit_items(per-k)={limit_items} llm={llm_name}"
    )

    n_in = 0
    n_considered = 0
    n_out = 0
    n_skip_k = 0
    n_skip_seen = 0
    n_skip_limit = 0

    with open(input_jsonl, "r", encoding="utf-8") as fin, open(output_jsonl, "a", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if line == "":
                continue

            n_in += 1
            item = json.loads(line)
            n_considered += 1

            # We still keep detailed skip counters (useful for debugging runs)
            item_k_raw = item.get("k")
            try:
                item_k = int(item_k_raw)
            except Exception:
                n_skip_k += 1
                continue

            src_chain_id = item.get("chain_id")

            # Mirror earlier behavior: count skip reasons explicitly
            if item_k not in allowed_k:
                n_skip_k += 1
                continue

            if limit_items is not None and per_k_written.get(item_k, 0) >= int(limit_items):
                n_skip_limit += 1
                continue

            if isinstance(src_chain_id, str) and src_chain_id in seen_chain_ids:
                n_skip_seen += 1
                continue

            # (Optional) also enforce via shared helper for consistency
            if not should_take_item(
                item,
                allowed_k=allowed_k,
                per_k_written=per_k_written,
                limit_items=limit_items,
                seen_source_chain_ids=seen_chain_ids,
            ):
                # skip counters already handled above; keep behavior stable
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

            k_now = int(item_k)
            k_prog = f"{per_k_written.get(k_now, 0)}/{int(limit_items) if limit_items is not None else '∞'}"
            log(f"[{task_cfg['name']}] in={n_in} k={k_now} (written {k_prog}) chain_id={src_chain_id}")

            parsed, _last_err = run_llm_qa_with_retries(
                task_name=task_cfg["name"],
                run_debug=bool(run["debug"]),
                llm=llm,
                prompt=prompt,
                ctx=ctx,
                max_retries=int(max_retries),
                retry_backoff_s=float(retry_backoff_s),
                min_question_len=int(min_question_len),
                item_index=int(n_in),
                k_now=int(k_now),
                src_chain_id=str(src_chain_id) if isinstance(src_chain_id, str) else None,
            )

            if parsed is None:
                continue

            out_item: Dict[str, Any] = {
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

            write_jsonl(fout, out_item)
            n_out += 1

            if isinstance(src_chain_id, str) and src_chain_id:
                seen_chain_ids.add(src_chain_id)

            per_k_written[k_now] = per_k_written.get(k_now, 0) + 1

            if n_out % 10 == 0:
                log(
                    f"[{task_cfg['name']}] generated {n_out} items (processed {n_in}) "
                    f"per_k_written={per_k_written}"
                )

            if all_k_full(per_k_written, k_list, limit_items):
                break

    log(
        f"[{task_cfg['name']}] done. processed={n_in} considered={n_considered} written={n_out} "
        f"skip_k={n_skip_k} skip_seen={n_skip_seen} skip_limit={n_skip_limit} "
        f"per_k_written={per_k_written} output={output_jsonl}"
    )
