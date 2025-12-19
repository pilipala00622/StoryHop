from __future__ import annotations

from typing import Any, Dict, List


def _chain_text_from_steps(steps: List[Dict[str, Any]]) -> str:
    names = [steps[0]["source"]["name"]]
    names += [st["target"]["name"] for st in steps]
    return " -> ".join(names)


def build_chain_features(item: Dict[str, Any]) -> Dict[str, Any]:
    """Extract task-agnostic features from a sampled k-hop chain.

    This is intentionally generic: it produces reusable fields that prompt
    builders can use to formulate different question types (retrieval, temporal
    ordering, causal/intent inference, etc.).
    """

    chain = item["chain"]
    steps = chain["steps"]

    k = len(steps)

    node_names: List[str] = [steps[0]["source"]["name"]] + [st["target"]["name"] for st in steps]
    rel_types: List[str] = [st["relation"]["type"] for st in steps]
    evidences: List[str] = [st["evidence"] for st in steps]
    chunk_ids_in_order: List[Any] = [st["chunk_id"] for st in steps]

    # Normalize chunk ids to ints when possible (useful for ordering/span signals)
    chunk_ids_int: List[int] = []
    for cid in chunk_ids_in_order:
        # chunk ids are expected to be numeric in this pipeline
        chunk_ids_int.append(int(cid))

    uniq_chunk_ids: List[int] = []
    for cid in chunk_ids_int:
        if cid not in uniq_chunk_ids:
            uniq_chunk_ids.append(cid)

    min_chunk_id = min(chunk_ids_int) if chunk_ids_int else None
    max_chunk_id = max(chunk_ids_int) if chunk_ids_int else None

    # Monotonicity in the order that evidence was traversed along the path
    nondecreasing = all(chunk_ids_int[i] <= chunk_ids_int[i + 1] for i in range(len(chunk_ids_int) - 1))
    strictly_increasing = all(chunk_ids_int[i] < chunk_ids_int[i + 1] for i in range(len(chunk_ids_int) - 1))

    chain_with_evidence_lines = []
    for i, st in enumerate(steps):
        hop = st["hop"]
        src = st["source"]["name"]
        rel = st["relation"]["type"]
        tgt = st["target"]["name"]
        ev = st["evidence"].replace("\n", " ").strip()
        cid = st["chunk_id"]
        chain_with_evidence_lines.append(
            f"- hop{hop}: {src} --[{rel}]--> {tgt} | evidence: {ev} | chunk_id: {cid}"
        )

    return {
        "book_id": item["book_id"],
        "chain_id": item["chain_id"],
        "k": item["k"],
        "final_answer": item["final_answer"],
        "start_entity": chain["start"]["name"],
        "end_entity": chain["end"]["name"],
        "node_names_in_order": node_names,
        "rel_types_in_order": rel_types,
        "evidences_in_order": evidences,
        "chunks_in_chain_order": chunk_ids_int,
        "chunk_ids_unique": uniq_chunk_ids,
        "min_chunk_id": min_chunk_id,
        "max_chunk_id": max_chunk_id,
        "chunk_span": (max_chunk_id - min_chunk_id) if (min_chunk_id is not None and max_chunk_id is not None) else None,
        "chunk_order_monotonic_inc": nondecreasing,
        "chunk_order_monotonic_strict": strictly_increasing,
        "gold_chain_text": _chain_text_from_steps(steps),
        "chain_with_evidence": "\n".join(chain_with_evidence_lines),
        "cypher_visualize_full": item["cypher_visualize_full"],
    }


# =========================
# Prompt builders (task-specific)
# =========================

def reader_qa_zh(ctx: Dict[str, Any]) -> str:
    k = ctx["k"]
    final_ans = ctx["final_answer"]
    start_ent = ctx["start_entity"]
    chain_str = ctx["chain_with_evidence"]
    answer_max_chars = ctx["answer_max_chars"]
    min_q_len = ctx["min_question_len"]

    return f"""
你是长篇小说阅读类 benchmark 的出题员。现在给你一条从原文抽取出来的多跳线索与证据片段。
你的任务：写出一个“正常读者会问”的问题，以及一个简短回答。

链路长度：k={k}

严格要求：
1) 问题必须通过整合这条链路中的线索才能推出答案。
2) 若 k >= 2：问题必须要求跨多个线索推理（至少用到 2 跳信息，最好 k 跳都能用上）。
3) 问题里不能出现“hop/链路/图/关系类型/chunk/评测”等结构化词汇。
4) 问题中不得出现最终答案的字符串：{final_ans}
5) 只能使用给定 evidence 的信息，不要编造剧情；尽量自然转述，不要整句照搬。
6) 问题长度至少 {min_q_len} 个中文字符（避免过于直接）。
7) 回答要简短（<= {answer_max_chars} 个中文字符），并且必须包含最终答案实体：{final_ans}。

输出格式（只输出严格 JSON，不要输出任何多余文字）：
{{
  "question": "...",
  "answer": "..."
}}

起点实体（建议在问题中作为剧情锚点出现）：{start_ent}

线索与证据：
{chain_str}
""".strip()


def temporal_order_zh(ctx: Dict[str, Any]) -> str:
    """Ask for temporal ordering / relative sequence inferred via the chain.

    This template encourages a reader-style question that hinges on when/what
    happened first/earlier vs later along the implied narrative progression.
    """

    k = ctx["k"]
    final_ans = ctx["final_answer"]
    start_ent = ctx["start_entity"]
    chain_str = ctx["chain_with_evidence"]
    answer_max_chars = ctx["answer_max_chars"]
    min_q_len = ctx["min_question_len"]

    return f"""
你是长篇小说阅读类 benchmark 的出题员。现在给你一条从原文抽取出来的多跳线索与证据片段。
你的任务：写出一个读者会问的“时间顺序/先后关系”问题，以及一个简短回答。

链路长度：k={k}

严格要求：
1) 问题必须要求读者结合多条线索推断“先后/紧接/之前/之后”的关系，而不是单句就能回答。
2) 问题不能出现“hop/链路/图/关系类型/chunk/评测”等结构化词汇。
3) 问题中不得出现最终答案的字符串：{final_ans}
4) 只能使用给定 evidence 的信息，不要编造剧情；尽量自然转述，不要整句照搬。
5) 问题长度至少 {min_q_len} 个中文字符。
6) 回答要简短（<= {answer_max_chars} 个中文字符），并且必须包含最终答案实体：{final_ans}。

输出格式（只输出严格 JSON，不要输出任何多余文字）：
{{
  "question": "...",
  "answer": "..."
}}

起点实体（建议在问题中作为剧情锚点出现）：{start_ent}

线索与证据：
{chain_str}
""".strip()


PROMPT_BUILDERS = {
    "reader_qa_zh": reader_qa_zh,
    "temporal_order_zh": temporal_order_zh,
}
