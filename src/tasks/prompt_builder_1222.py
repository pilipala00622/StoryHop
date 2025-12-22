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
你的任务：写一个"跨段落身份一致/别名消解"问题，问题中只使用别称/称谓"{alias}"，要求回答者指出其真实是谁。

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


# =========================
# 四种关系维度的 Prompt Builders
# =========================

def functional_relationship_zh(ctx: Dict[str, Any]) -> str:
    """
    功能性关系 (Functional Relationship)
    基于普罗普的"行动圈"理论，考察实体在推动情节中的作用。
    例如：谁是"施与者"给主角提供了"魔法道具"？
    """
    k = ctx["k"]
    final_ans = ctx["final_answer"]
    start_ent = ctx["start_entity"]
    chain_str = ctx["chain_with_evidence"]
    answer_max_chars = ctx["answer_max_chars"]
    min_q_len = ctx["min_question_len"]
    node_names = ctx["node_names_in_order"]
    rel_types = ctx["rel_types_in_order"]

    return f"""
你是长篇小说阅读类 benchmark 的出题员。现在给你一条从原文抽取出来的多跳线索与证据片段。
你的任务：基于"功能性关系"维度，写出一个考察实体在推动情节中作用的问题，以及一个简短回答。

关系维度说明：
功能性关系关注实体在叙事中的"行动圈"功能（如：施与者、助手、对手、受难者等）。
问题应考察某个实体如何通过提供道具/信息/帮助/阻碍等方式推动情节发展。

链路长度：k={k}
起点实体：{start_ent}
终点实体：{final_ans}
路径中的实体：{" -> ".join(node_names)}
关系类型序列：{" -> ".join(rel_types)}

严格要求：
1) 问题必须聚焦于"功能性作用"：考察某个实体（如{start_ent}或{final_ans}）在情节中扮演什么功能性角色。
2) 问题应体现"行动圈"理论：如"谁给主角提供了关键物品/信息？"、"谁在关键时刻帮助/阻碍了主角？"、"谁承担了受难者的角色？"等。
3) 问题必须通过整合这条链路中的多条线索才能推出答案，不能只依赖单条证据。
4) 问题要足够复杂：需要理解实体间的功能互动关系，而非简单的"谁是谁"。
5) 问题中不得出现最终答案的字符串：{final_ans}
6) 只能使用给定 evidence 的信息，不要编造剧情；尽量自然转述，不要整句照搬。
7) 问题长度至少 {min_q_len} 个中文字符，最好 20+ 字符以体现复杂度。
8) 回答要简短（<= {answer_max_chars} 个中文字符），并且必须包含最终答案实体：{final_ans}。
9) 问题里不能出现"hop/链路/图/关系类型/chunk/评测/功能性/行动圈"等结构化或理论性词汇。

输出格式（只输出严格 JSON，不要输出任何多余文字）：
{{
  "question": "...",
  "answer": "..."
}}

线索与证据：
{chain_str}
""".strip()


def structural_relationship_zh(ctx: Dict[str, Any]) -> str:
    """
    社会网络关系 (Structural/Network Relationship)
    将人物视为节点，互动视为边，考察关系的疏密、权力的中心化及跨群体联系。
    """
    k = ctx["k"]
    final_ans = ctx["final_answer"]
    start_ent = ctx["start_entity"]
    chain_str = ctx["chain_with_evidence"]
    answer_max_chars = ctx["answer_max_chars"]
    min_q_len = ctx["min_question_len"]
    node_names = ctx["node_names_in_order"]
    rel_types = ctx["rel_types_in_order"]

    return f"""
你是长篇小说阅读类 benchmark 的出题员。现在给你一条从原文抽取出来的多跳线索与证据片段。
你的任务：基于"社会网络关系"维度，写出一个考察人物间关系结构、权力中心或跨群体联系的问题，以及一个简短回答。

关系维度说明：
社会网络关系关注人物作为"节点"、互动作为"边"所形成的网络结构。
问题应考察：关系的疏密程度、权力的中心化（谁处于关系网络的中心）、跨群体/派系的桥接作用等。

链路长度：k={k}
起点实体：{start_ent}
终点实体：{final_ans}
路径中的实体：{" -> ".join(node_names)}
关系类型序列：{" -> ".join(rel_types)}

严格要求：
1) 问题必须聚焦于"关系结构"：考察人物间的社会网络特征，而非单纯的事实查询。
2) 问题应体现网络分析视角：如"谁与多个重要人物都有直接联系？"、"谁在某个群体中处于核心地位？"、"谁连接了两个不同的派系/群体？"、"谁的关系网络更密集？"等。
3) 问题必须通过整合这条链路中的多条线索才能推出答案，需要理解多个人物间的互动模式。
4) 问题要足够复杂：需要分析关系密度、中心度、桥接性等结构特征，而非简单的"A认识B"。
5) 问题中不得出现最终答案的字符串：{final_ans}
6) 只能使用给定 evidence 的信息，不要编造剧情；尽量自然转述，不要整句照搬。
7) 问题长度至少 {min_q_len} 个中文字符，最好 25+ 字符以体现复杂度。
8) 回答要简短（<= {answer_max_chars} 个中文字符），并且必须包含最终答案实体：{final_ans}。
9) 问题里不能出现"hop/链路/图/关系类型/chunk/评测/网络/节点/边/中心度/桥接"等结构化或理论性词汇。

输出格式（只输出严格 JSON，不要输出任何多余文字）：
{{
  "question": "...",
  "answer": "..."
}}

线索与证据：
{chain_str}
""".strip()


def causal_relationship_zh(ctx: Dict[str, Any]) -> str:
    """
    因果/逻辑关系 (Causal/Logical Relationship)
    序列逻辑（可能一过程—结果），考察实体状态的变化是否符合前因后果，是否存在逻辑矛盾。
    """
    k = ctx["k"]
    final_ans = ctx["final_answer"]
    start_ent = ctx["start_entity"]
    chain_str = ctx["chain_with_evidence"]
    answer_max_chars = ctx["answer_max_chars"]
    min_q_len = ctx["min_question_len"]
    node_names = ctx["node_names_in_order"]
    rel_types = ctx["rel_types_in_order"]

    return f"""
你是长篇小说阅读类 benchmark 的出题员。现在给你一条从原文抽取出来的多跳线索与证据片段。
你的任务：基于"因果/逻辑关系"维度，写出一个考察事件序列、前因后果或逻辑链条的问题，以及一个简短回答。

关系维度说明：
因果/逻辑关系关注事件或状态的序列逻辑（原因→过程→结果）。
问题应考察：某个结果的前因是什么？某个事件导致了什么后果？状态变化是否符合逻辑？是否存在逻辑矛盾？

链路长度：k={k}
起点实体：{start_ent}
终点实体：{final_ans}
路径中的实体：{" -> ".join(node_names)}
关系类型序列：{" -> ".join(rel_types)}

严格要求：
1) 问题必须聚焦于"因果链条"或"逻辑序列"：考察事件/状态之间的前因后果关系，而非单纯的事实查询。
2) 问题应体现序列逻辑：如"什么导致了某个结果？"、"某个事件引发了什么后续变化？"、"为什么会出现某种状态？"、"某个决定造成了什么影响？"等。
3) 问题必须通过整合这条链路中的多条线索才能推出答案，需要理解完整的因果链条。
4) 问题要足够复杂：需要追踪多步因果推理，理解"原因→中间过程→结果"的完整逻辑，而非简单的"A导致B"。
5) 问题中不得出现最终答案的字符串：{final_ans}
6) 只能使用给定 evidence 的信息，不要编造剧情；尽量自然转述，不要整句照搬。
7) 问题长度至少 {min_q_len} 个中文字符，最好 25+ 字符以体现复杂度。
8) 回答要简短（<= {answer_max_chars} 个中文字符），并且必须包含最终答案实体：{final_ans}。
9) 问题里不能出现"hop/链路/图/关系类型/chunk/评测/因果/逻辑/序列"等结构化或理论性词汇。

输出格式（只输出严格 JSON，不要输出任何多余文字）：
{{
  "question": "...",
  "answer": "..."
}}

线索与证据：
{chain_str}
""".strip()


def symbolic_relationship_zh(ctx: Dict[str, Any]) -> str:
    """
    象征/主题关系 (Symbolic/Thematic Relationship)
    实体对主题的支撑作用，考察特定道具或环境如何映射人物的内心状态或宏大叙事。
    """
    k = ctx["k"]
    final_ans = ctx["final_answer"]
    start_ent = ctx["start_entity"]
    chain_str = ctx["chain_with_evidence"]
    answer_max_chars = ctx["answer_max_chars"]
    min_q_len = ctx["min_question_len"]
    node_names = ctx["node_names_in_order"]
    rel_types = ctx["rel_types_in_order"]

    return f"""
你是长篇小说阅读类 benchmark 的出题员。现在给你一条从原文抽取出来的多跳线索与证据片段。
你的任务：基于"象征/主题关系"维度，写出一个考察实体（道具/环境/意象）如何映射人物内心状态或支撑宏大主题的问题，以及一个简短回答。

关系维度说明：
象征/主题关系关注实体对主题的支撑作用。
问题应考察：特定道具/环境/意象如何反映人物的内心状态？某个实体如何承载或象征更大的主题/叙事意义？

链路长度：k={k}
起点实体：{start_ent}
终点实体：{final_ans}
路径中的实体：{" -> ".join(node_names)}
关系类型序列：{" -> ".join(rel_types)}

严格要求：
1) 问题必须聚焦于"象征意义"或"主题映射"：考察实体如何承载深层含义，而非单纯的事实查询。
2) 问题应体现象征分析视角：如"某个道具/环境反映了人物怎样的内心状态？"、"某个意象象征了什么主题？"、"某个场景如何映射人物的情感/命运？"、"某个细节如何支撑更大的叙事主题？"等。
3) 问题必须通过整合这条链路中的多条线索才能推出答案，需要理解实体与主题/内心状态的映射关系。
4) 问题要足够复杂：需要理解象征层面的含义，而非简单的"这是什么物品"或"这是哪里"。
5) 问题中不得出现最终答案的字符串：{final_ans}
6) 只能使用给定 evidence 的信息，不要编造剧情；尽量自然转述，不要整句照搬。
7) 问题长度至少 {min_q_len} 个中文字符，最好 25+ 字符以体现复杂度。
8) 回答要简短（<= {answer_max_chars} 个中文字符），并且必须包含最终答案实体：{final_ans}。
9) 问题里不能出现"hop/链路/图/关系类型/chunk/评测/象征/主题/映射"等结构化或理论性词汇。

输出格式（只输出严格 JSON，不要输出任何多余文字）：
{{
  "question": "...",
  "answer": "..."
}}

线索与证据：
{chain_str}
""".strip()


def conflict_analysis_zh(ctx: Dict[str, Any]) -> str:
    """
    冲突分析 (Conflict Analysis)
    分析角色间产生摩擦、误解或背叛的频率和类型。
    """
    witness = ctx["witness"]
    conflict_type = witness["conflict_type"]
    source_entity = witness["source_entity"]
    target_entity = witness["target_entity"]
    evidence = witness["evidence"]
    all_conflicts = witness.get("all_conflicts", evidence)
    conflict_count = witness.get("conflict_count", 1)
    min_q_len = ctx["min_question_len"]
    answer_max_chars = ctx["answer_max_chars"]
    final_answer = ctx["final_answer"]

    return f"""
你是长篇小说阅读类 benchmark 的出题员。现在给你一条从原文抽取出来的多跳线索与证据片段。
你的任务：基于"冲突分析"维度，写出一个考察角色间摩擦、误解或背叛的问题，以及一个简短回答。

关系维度说明：
冲突分析关注角色间的人际关系紧张程度，包括：
- 摩擦：日常冲突、争执、对立
- 误解：误会、错怪、曲解
- 背叛：出卖、背弃、反叛

当前案例：
- 冲突类型：{conflict_type}
- 涉及角色：{source_entity} 与 {target_entity}
- 冲突次数：{conflict_count}次

严格要求：
1) 问题必须聚焦于"冲突关系"：考察角色间的摩擦、误解或背叛，而非单纯的事实查询。
2) 问题应体现冲突分析视角：如"谁与谁发生了摩擦/误解/背叛？"、"某个角色与哪些角色关系紧张？"、"角色间的主要冲突类型是什么？"等。
3) 问题必须通过整合这条链路中的多条线索才能推出答案，需要理解完整的冲突链条。
4) 问题要足够复杂：需要分析冲突的类型、频率、涉及的角色等多方面信息，而非简单的"谁和谁冲突"。
5) 问题中不得出现最终答案的字符串：{final_answer}
6) 只能使用给定 evidence 的信息，不要编造剧情；尽量自然转述，不要整句照搬。
7) 问题长度至少 {min_q_len} 个中文字符，最好 25+ 字符以体现复杂度。
8) 回答要简短（<= {answer_max_chars} 个中文字符），并且必须包含最终答案：{final_answer}。
9) 问题里不能出现"hop/链路/图/关系类型/chunk/评测/冲突/摩擦/误解/背叛"等结构化或理论性词汇。

输出格式（只输出严格 JSON，不要输出任何多余文字）：
{{
  "question": "...",
  "answer": "..."
}}

线索与证据：
{all_conflicts}
""".strip()


def character_consistency_zh(ctx: Dict[str, Any]) -> str:
    """
    人物一致性分析 (Character Consistency Analysis)
    分析人物的行动是否与其设定的属性、身份相符。
    """
    witness = ctx["witness"]
    entity = witness["entity"]
    attribute = witness["attribute"]
    action = witness["action"]
    is_consistent = witness["is_consistent"]
    evidence = witness["evidence"]
    min_q_len = ctx["min_question_len"]
    answer_max_chars = ctx["answer_max_chars"]
    final_answer = ctx["final_answer"]

    consistency_status = "一致" if is_consistent else "不一致"
    attribute_desc = f"{entity}的身份/属性是{attribute['value']}"
    action_desc = f"{entity}的行动是{action['rel_type']}（针对{action['target']}）"

    return f"""
你是长篇小说阅读类 benchmark 的出题员。现在给你一条从原文抽取出来的多跳线索与证据片段。
你的任务：基于"人物一致性"维度，写出一个考察人物行动是否与其身份/属性相符的问题，以及一个简短回答。

关系维度说明：
人物一致性关注角色的行为是否符合其设定的身份、地位、性格等属性。
问题应考察：某个角色的行动是否与其身份相符？是否存在行为与设定矛盾的情况？

当前案例：
- 人物：{entity}
- {attribute_desc}
- {action_desc}
- 一致性：{consistency_status}

严格要求：
1) 问题必须聚焦于"一致性分析"：考察人物行动与身份/属性的匹配程度，而非单纯的事实查询。
2) 问题应体现一致性分析视角：如"某个角色的行为是否符合其身份？"、"某个角色的行动是否与其设定矛盾？"、"某个角色的行为是否反常？"等。
3) 问题必须通过整合这条链路中的多条线索才能推出答案，需要同时理解人物的属性和行动。
4) 问题要足够复杂：需要对比分析人物的身份设定和实际行为，判断是否一致，而非简单的"这个人是谁"。
5) 问题中不得出现最终答案的字符串：{final_answer}
6) 只能使用给定 evidence 的信息，不要编造剧情；尽量自然转述，不要整句照搬。
7) 问题长度至少 {min_q_len} 个中文字符，最好 25+ 字符以体现复杂度。
8) 回答要简短（<= {answer_max_chars} 个中文字符），并且必须包含最终答案：{final_answer}。
9) 问题里不能出现"hop/链路/图/关系类型/chunk/评测/一致性/属性/身份"等结构化或理论性词汇。

输出格式（只输出严格 JSON，不要输出任何多余文字）：
{{
  "question": "...",
  "answer": "..."
}}

线索与证据：
{evidence}
""".strip()


PROMPT_BUILDERS = {
    "khop_qa_zh": khop_qa_zh,
    "single_span_qa_zh": single_span_qa_zh,
    "attribute_lookup_zh": attribute_lookup_zh,
    "who_act_what_zh": who_act_what_zh,
    "local_coref_zh": local_coref_zh,
    "alias_coref_zh": alias_coref_zh,
    # 四种关系维度的 prompt builders
    "functional_relationship_zh": functional_relationship_zh,
    "structural_relationship_zh": structural_relationship_zh,
    "causal_relationship_zh": causal_relationship_zh,
    "symbolic_relationship_zh": symbolic_relationship_zh,
    # 冲突分析和人物一致性分析
    "conflict_analysis_zh": conflict_analysis_zh,
    "character_consistency_zh": character_consistency_zh,
}

