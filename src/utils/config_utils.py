from __future__ import annotations

"""Configuration loading utilities (YAML + ${ENV} expansion).

Rules
- Default config path: ./config.yaml
- Override config path: environment variable STORYHOP_CONFIG
- Placeholder expansion: any string containing ${ENV_VAR} is expanded from os.environ

This file is intended to be the single source of truth for config loading.
"""

import os
import re
from typing import Any, Dict

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env_in_str(s: str) -> str:
    def repl(m: re.Match) -> str:
        key = m.group(1)
        val = os.environ.get(key)
        if val is None:
            raise RuntimeError(
                f"Missing environment variable: {key} (referenced in config via ${{{key}}})"
            )
        return val

    return _ENV_PATTERN.sub(repl, s)


def expand_env(obj: Any) -> Any:
    """Recursively expand ${ENV_VAR} placeholders in a nested structure."""
    if isinstance(obj, str):
        return _expand_env_in_str(obj)
    if isinstance(obj, list):
        return [expand_env(x) for x in obj]
    if isinstance(obj, dict):
        return {k: expand_env(v) for k, v in obj.items()}
    return obj


def load_config(path_override: str | None = None) -> Dict[str, Any]:
    """Load YAML config and expand env vars."""
    path = path_override or os.environ.get("STORYHOP_CONFIG", "config.yaml")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Config file not found: {path}. "
            f"Set STORYHOP_CONFIG=/path/to/config.yaml or create ./config.yaml"
        )

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise RuntimeError("config.yaml top-level must be a mapping/dict")

    return expand_env(cfg)
