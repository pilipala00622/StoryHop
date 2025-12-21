from __future__ import annotations

from typing import Any, Dict, List


def _chain_text_from_steps(steps: List[Dict[str, Any]]) -> str:
    names = [steps[0]["source"]["name"]]
    names += [st["target"]["name"] for st in steps]
    return " -> ".join(names)


def build_chain_features(item: Dict[str, Any], cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Extract task-agnostic features from a sampled k-hop chain.

    Updated to match the new chain JSONL schema:
      - labels removed from nodes
      - cypher_visualize_full renamed to full_query
      - meta removed; s_eid/t_eid are top-level fields
      - QA defaults pulled from cfg["run"]["qa"] if cfg is provided
    """
    chain = item["chain"]
    steps = chain["steps"]

    # Prefer actual hop count in steps; fall back to item["k"] (they should match)
    k = len(steps) if steps else int(item["k"])

    node_names: List[str] = [steps[0]["source"]["name"]] + [st["target"]["name"] for st in steps]
    rel_types: List[str] = [st["relation"]["type"] for st in steps]
    evidences: List[str] = [st["evidence"] for st in steps]
    chunk_ids_in_order: List[Any] = [st["chunk_id"] for st in steps]

    # Normalize chunk ids to ints when possible (useful for ordering/span signals)
    chunk_ids_int: List[int] = [int(cid) for cid in chunk_ids_in_order]

    uniq_chunk_ids: List[int] = []
    for cid in chunk_ids_int:
        if cid not in uniq_chunk_ids:
            uniq_chunk_ids.append(cid)

    min_chunk_id = min(chunk_ids_int) if chunk_ids_int else None
    max_chunk_id = max(chunk_ids_int) if chunk_ids_int else None

    # Monotonicity in the order that evidence was traversed along the path
    nondecreasing = all(chunk_ids_int[i] <= chunk_ids_int[i + 1] for i in range(len(chunk_ids_int) - 1))
    strictly_increasing = all(chunk_ids_int[i] < chunk_ids_int[i + 1] for i in range(len(chunk_ids_int) - 1))

    chain_with_evidence_lines: List[str] = []
    for st in steps:
        hop = st["hop"]
        src = st["source"]["name"]
        rel = st["relation"]["type"]
        tgt = st["target"]["name"]
        ev = st["evidence"].replace("\n", " ").strip()
        cid = st["chunk_id"]
        chain_with_evidence_lines.append(f"- hop{hop}: {src} --[{rel}]--> {tgt} | evidence: {ev} | chunk_id: {cid}")

    # Defaults from config (new structure)
    qa_defaults: Dict[str, Any] = (cfg or {}).get("run", {}).get("qa", {})
    answer_max_chars = int(qa_defaults.get("answer_max_chars", 40))
    min_question_len = int(qa_defaults.get("min_question_len", 12))
    language = qa_defaults.get("language", "zh")

    return {
        # identity
        "book_id": item["book_id"],
        "chain_id": item["chain_id"],
        "k": int(item.get("k", k)),
        "final_answer": item["final_answer"],
        "s_eid": item.get("s_eid"),
        "t_eid": item.get("t_eid"),

        # chain content
        "start_entity": chain["start"]["name"],
        "end_entity": chain["end"]["name"],
        "node_names_in_order": node_names,
        "rel_types_in_order": rel_types,
        "evidences_in_order": evidences,
        "chunks_in_chain_order": chunk_ids_int,
        "chunk_ids_unique": uniq_chunk_ids,

        # simple signals
        "min_chunk_id": min_chunk_id,
        "max_chunk_id": max_chunk_id,
        "chunk_span": (max_chunk_id - min_chunk_id) if (min_chunk_id is not None and max_chunk_id is not None) else None,
        "chunk_order_monotonic_inc": nondecreasing,
        "chunk_order_monotonic_strict": strictly_increasing,

        # renderings
        "gold_chain_text": _chain_text_from_steps(steps),
        "chain_with_evidence": "\n".join(chain_with_evidence_lines),

        # query debug
        "full_query": item.get("full_query"),

        # QA defaults (so prompt builders don't need cfg)
        "language": language,
        "answer_max_chars": answer_max_chars,
        "min_question_len": min_question_len,
    }


# =========================
# Prompt builders (task-specific)
# =========================

def khop_qa_zh(ctx: Dict[str, Any]) -> str:
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

def single_span_qa_zh(ctx: Dict[str, Any]) -> str:
    anchor = ctx["anchor_entity"]
    target = ctx["final_answer"]
    ev = ctx["evidence"]
    min_q_len = ctx["min_question_len"]

    return f"""
你是长篇小说阅读类 benchmark 的出题员。给你一条原文证据片段（单条线索）。
任务：基于 evidence 生成一个自然问题，并给出可在 evidence 中直接找到的回答。

要求：
1) 问题与回答都必须只依赖这条 evidence。
2) 问题里不得出现答案字符串：{target}
3) 问题长度至少 {min_q_len} 个中文字符。
4) 回答必须包含答案字符串：{target}
5) 输出必须是严格 JSON，不要输出任何多余文字。

输出格式：
{{
  "question": "...",
  "answer": "..."
}}

剧情锚点（建议在问题中出现）：{anchor}

evidence：
{ev}
""".strip()

def attribute_lookup_zh(ctx: dict) -> str:
    """
    Expected ctx fields:
      - language (e.g., "zh")
      - witness: {entity, value, rel_type, evidence, evidence_chunk_id}
      - answer_max_chars
    Returns: a prompt that asks the LLM to output JSON: {"question": "..."}.
    """
    w = ctx["witness"]
    entity = w["entity"]
    value = w["value"]
    evidence = w["evidence"]
    rel_type = w["rel_type"]

    # Keep question anchored, and do not leak the answer value.
    # The answer is deterministic and will be stored by the runner.
    # We instruct the LLM not to include the exact value in the question.
    return f"""
你将基于小说原文片段生成一个“属性查询”问题。

要求：
- 只输出 JSON，对象包含字段：question
- question 必须是中文
- question 必须提到实体“{entity}”
- question 不得包含答案文本（即不得出现“{value}”）
- 问题应询问 {entity} 的身份/称号/职务/所在地等属性（根据片段语义自然选择）
- 片段中可以直接支持该问题

原文片段：
{evidence}

输出格式示例：
{{"question": "文中称{entity}是什么身份？"}}
""".strip()


def who_act_what_zh(ctx: Dict[str, Any]) -> str:
    ans = ctx["final_answer"]
    ev = ctx["evidence"]
    min_q_len = ctx["min_question_len"]

    return f"""
你是长篇小说阅读类 benchmark 的出题员。给你一条原文证据片段（单条线索）。
任务：基于 evidence 生成一个“谁说了/做了什么”的自然问题，并给出可从 evidence 直接找到的回答（执行者是谁）。

要求：
1) 问题与回答都必须只依赖这条 evidence。
2) 问题中不得出现执行者名字：{ans}
3) 问题长度至少 {min_q_len} 个中文字符。
4) 回答必须包含执行者名字：{ans}
5) 输出必须是严格 JSON，不要输出任何多余文字。

输出格式：
{{
  "question": "...",
  "answer": "..."
}}

evidence：
{ev}
""".strip()


def local_coref_zh(ctx: Dict[str, Any]) -> str:
    # required ctx keys
    evidence = ctx["evidence"]          # single chunk evidence text
    pronoun = ctx["pronoun"]            # e.g., "那人"/"他"/"她"/"其"
    ans = ctx["final_answer"]           # canonical entity name, must NOT appear in question
    min_q_len = ctx["min_question_len"]

    return f"""
你是长篇小说阅读类 benchmark 的出题员。现在给你一条原文证据片段（单个 chunk 内的上下文）。
你的任务：写一个“代词/指代消解”问题，要求回答者指出文中“{pronoun}”指的是谁。

严格要求：
1) 问题必须只依赖这条 evidence 就能回答，不需要多跳推理。
2) 问题中不得出现答案字符串：{ans}
3) 问题长度至少 {min_q_len} 个中文字符。
4) 输出必须是严格 JSON，不要输出任何多余文字。

输出格式：
{{
  "question": "...",
  "answer": "{ans}"
}}

evidence：
{evidence}
""".strip()

def alias_coref_zh(ctx: Dict[str, Any]) -> str:
    # required ctx keys
    alias = ctx["alias"]                # surface form used in question (e.g., "老夫人"/"谢周氏"/"她")
    evidence_pack = ctx["evidence_pack"]  # multi-chunk evidence text (you decide formatting)
    ans = ctx["final_answer"]           # canonical entity name
    min_q_len = ctx["min_question_len"]

    return f"""
你是长篇小说阅读类 benchmark 的出题员。现在给你若干条原文证据片段（来自同一条链路的多个 hop）。
你的任务：写一个“跨段落身份一致/别名消解”问题，问题中只使用别称/称谓“{alias}”，要求回答者指出其真实是谁。

严格要求：
1) 问题必须只依赖给定 evidence，就能得出答案，不使用常识补全。
2) 问题中不得出现答案字符串：{ans}
3) 问题长度至少 {min_q_len} 个中文字符。
4) 输出必须是严格 JSON，不要输出任何多余文字。

输出格式：
{{
  "question": "...",
  "answer": "{ans}"
}}

evidence：
{evidence_pack}
""".strip()

PROMPT_BUILDERS = {
    "khop_qa_zh": khop_qa_zh,
    "single_span_qa_zh": single_span_qa_zh,
    "attribute_lookup_zh": attribute_lookup_zh,
    "who_act_what_zh": who_act_what_zh,
    "local_coref_zh": local_coref_zh,
    "alias_coref_zh": alias_coref_zh,

}

