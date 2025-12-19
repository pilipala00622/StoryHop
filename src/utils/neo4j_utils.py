from __future__ import annotations

from typing import Any

from ..config import Neo4jConfig


def build_graph_store(cfg: Neo4jConfig):
    """Build LlamaIndex Neo4j property graph store.

    Requires:
      pip install llama-index-graph-stores-neo4j

    Notes
    - Different llama-index versions use slightly different parameter names;
      we filter by signature to keep this repo stable.
    """

    from inspect import signature

    from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

    kwargs = {
        "username": cfg.user,
        "password": cfg.password,
        "url": cfg.url,
        "database": cfg.database,
    }

    # Drop None and unknown params
    sig = signature(Neo4jPropertyGraphStore.__init__)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if v is not None and k in allowed}

    return Neo4jPropertyGraphStore(**filtered)


def reset_graph_store(graph_store: Any) -> None:
    """Destructively delete all nodes/edges."""

    graph_store.structured_query("MATCH (n) DETACH DELETE n")
