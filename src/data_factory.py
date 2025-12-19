from __future__ import annotations

from typing import Any, Dict

from src.llm_factory import build_llm_from_cfg_dict
from src.utils.config_utils import load_config
from src.utils.log_utils import log, set_debug
from src.tasks.prompt_builders import PROMPT_BUILDERS
from src.tasks.khop_llm_qa import run_task


def main() -> None:
    cfg: Dict[str, Any] = load_config()

    set_debug(cfg["run"]["debug"])

    if not cfg["data_factory"]["enabled"]:
        log("data_factory.enabled=false; exiting")
        return

    llm, llm_name = build_llm_from_cfg_dict(cfg)
    log(f"LLM initialized: {llm_name}")

    tasks = cfg["data_factory"]["tasks"]
    for task_cfg in tasks:
        name = task_cfg["name"]
        builder_key = task_cfg["prompt_builder"]
        prompt_builder = PROMPT_BUILDERS[builder_key]
        log("-" * 80)
        log(f"Running DataFactory task: {name}")
        run_task(cfg=cfg, task_cfg=task_cfg, prompt_builder=prompt_builder, llm=llm, llm_name=llm_name)


if __name__ == "__main__":
    main()
