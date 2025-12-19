from __future__ import annotations

import time
from typing import Optional

# Global debug switch (can be set by CLI args or config)
_DEBUG: bool = False


def set_debug(enabled: bool) -> None:
    global _DEBUG
    _DEBUG = bool(enabled)


def get_debug() -> bool:
    return _DEBUG


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _print(level: str, msg: str) -> None:
    # stdout logging via print, flush for server logs / slurm
    print(f"{_ts()} | {level:<5} | {msg}", flush=True)


def log(msg: str) -> None:
    _print("INFO", msg)


def warn(msg: str) -> None:
    _print("WARN", msg)


def err(msg: str) -> None:
    _print("ERROR", msg)


def debug(msg: str, enabled: Optional[bool] = None) -> None:
    """
    Debug log. If enabled is None, use global _DEBUG.
    This keeps compatibility with older calls like debug(msg, dbg).
    """
    if enabled is None:
        enabled = _DEBUG
    if enabled:
        _print("DEBUG", msg)
