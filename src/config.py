from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Neo4jConfig:
    """Typed container for Neo4j connectivity.

    config.yaml is the single source of truth. These dataclasses exist only to
    keep internal APIs explicit and type-checkable.
    """

    url: str
    user: str
    password: str
    database: str


@dataclass
class ChunkingConfig:
    """Chunking hyperparameters for sliding-window chunking."""

    chunk_chars: int
    chunk_overlap: int
