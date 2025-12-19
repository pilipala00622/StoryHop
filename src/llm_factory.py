from __future__ import annotations

import os
from inspect import signature
from typing import Any, Dict, Tuple

from llama_index.llms.openai import OpenAI


def _filter_kwargs(callable_obj, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter kwargs to match a callable signature."""

    allowed = set(signature(callable_obj).parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed and v is not None}


def build_llm_from_cfg_dict(cfg: Dict[str, Any]) -> Tuple[Any, str]:
    """Build an OpenAI LLM client from config.yaml.

    Expected config:
      llm:
        openai:
          model: ...
          api_key: ...
          timeout_s: ...
          max_retries: ...
          max_output_tokens: ...
          additional_kwargs: null | {...}
    """

    openai_cfg = cfg["llm"]["openai"]

    # Make sure downstream libs see the key.
    os.environ["OPENAI_API_KEY"] = openai_cfg["api_key"]

    # LlamaIndex OpenAI wrapper parameter names vary by version.
    # We populate a superset and filter against the installed signature.
    kwargs: Dict[str, Any] = {
        "model": openai_cfg["model"],
        "api_key": openai_cfg["api_key"],
        "timeout": openai_cfg["timeout_s"],
        "request_timeout": openai_cfg["timeout_s"],
        "max_retries": openai_cfg["max_retries"],
        "max_tokens": openai_cfg["max_output_tokens"],
        "max_output_tokens": openai_cfg["max_output_tokens"],
        "additional_kwargs": openai_cfg["additional_kwargs"],
    }

    filtered = _filter_kwargs(OpenAI.__init__, kwargs)
    llm = OpenAI(**filtered)
    return llm, f"openai:{openai_cfg['model']}"
