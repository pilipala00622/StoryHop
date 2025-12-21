from __future__ import annotations

import importlib
from typing import Any, Dict

from src.llm_factory import build_llm_from_cfg_dict
from src.utils.config_utils import load_config
from src.utils.log_utils import log, set_debug
from src.tasks.prompt_builders import PROMPT_BUILDERS


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    cfg: Dict[str, Any] = load_config()
    set_debug(cfg["run"]["debug"])

    if not cfg["data_factory"]["enabled"]:
        log("data_factory.enabled=false; exiting")
        return

    mode: str = cfg["data_factory"]["mode"]

    llm = None
    llm_name = ""
    if mode in ("llm_qa", "all"):
        llm, llm_name = build_llm_from_cfg_dict(cfg)
        log(f"LLM initialized: {llm_name}")

    for task_cfg in cfg["data_factory"]["tasks"]:
        if int(task_cfg["limit_items"]) == 0:
            log(f"Skipping DataFactory task: {task_cfg['name']} (limit_items=0)")
            continue

        name = task_cfg["name"]
        log("-" * 80)
        log(f"Running DataFactory task: {name} (mode={mode})")

        # Option B: task references a named chain-gen config block
        if "chains_gen_cfg_key" in task_cfg:
            task_cfg["chains_gen_cfg"] = cfg["chain_gens"][task_cfg["chains_gen_cfg_key"]]

        # Make k available to llm_qa modules (khop_llm_qa expects task_cfg["k"])
        if "k" not in task_cfg and "chains_gen_cfg" in task_cfg and "k" in task_cfg["chains_gen_cfg"]:
            task_cfg["k"] = task_cfg["chains_gen_cfg"]["k"]

        if mode in ("chain_gen", "all"):
            chain_mod = importlib.import_module(f"src.tasks.chains_gen.{task_cfg['chains_gen']}")
            chain_mod.run_chain_gen(cfg=cfg, task_cfg=task_cfg)

        if mode in ("llm_qa", "all"):
            qa_mod = importlib.import_module(f"src.tasks.llm_qa.{task_cfg['llm_qa']}")
            prompt_builder = PROMPT_BUILDERS[task_cfg["prompt_builder"]]
            qa_mod.run_task(
                cfg=cfg,
                task_cfg=task_cfg,
                prompt_builder=prompt_builder,
                llm=llm,
                llm_name=llm_name,
            )


if __name__ == "__main__":
    main()
