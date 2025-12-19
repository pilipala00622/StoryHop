from __future__ import annotations

# Keep the style aligned with your existing prompts: "任务/要求/输出格式" + strict JSON.

LOW_LEVEL_EXTRACT_PROMPT = """
你是一个信息抽取系统。你必须只输出一个严格可解析的 JSON对象，除此之外不要输出任何文字。

任务：从以下小说片段中抽取【实体】与【关系】，用于构建混合图（人物-地点-组织-物品-事件-时间 + 因果/时序/隶属/互动等关系）。

要求：
1. 只能依据片段中的明示信息，不要引入常识、推测、补完或背景知识。
2. 实体类型（type）必须从以下集合中选择（全大写）：
   - PERSON, PLACE, ORGANIZATION, OBJECT, EVENT, TIME
3. 关系类型（type）必须从以下集合中选择（全大写）：
   - MENTIONS, LOCATED_IN, PART_OF, OWNS, INTERACTS_WITH, BEFORE, AFTER, CAUSES, CAUSED_BY, HAS_ATTRIBUTE, HAS_ALIAS
4. 每个实体给出一句简短描述（<=20字，尽量抽取原文信息），不要写长摘要。
5. 每条关系必须给出 evidence：从原文中复制一段能够“直接支持该关系”的短句（<=40字）。
6. 输出 JSON 必须遵循下面的 schema（字段名固定）：

JSON 格式如下：

{{
  "entities": [
    {{
      "id": "PERSON::某某",          // 你自己生成的稳定ID：推荐用 TYPE::NAME
      "name": "某某",
      "type": "PERSON",
      "description": "一句话描述"
    }},
  ],
  "relations": [
    {{
      "source": "PERSON::某某",      // 必须引用上面 entities 里的 id
      "type": "INTERACTS_WITH",
      "target": "PERSON::某某",
      "description": "一句话描述该关系",
      "evidence": "原文短句"
    }}
  ]
}}

小说片段：
{chunk}
"""
