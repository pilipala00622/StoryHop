"""Task-specific prompt builders.

Each prompt builder is a pure function:
    (ctx: Dict[str, Any], task_cfg: Dict[str, Any]) -> str

The chain/path parsing and QA generation loop live in `khop_llm_qa.py`.
"""

from .prompt_builders import PROMPT_BUILDERS  # re-export
