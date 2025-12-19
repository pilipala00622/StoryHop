from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from llama_index.core.schema import TextNode
from llama_index.core.graph_stores.types import EntityNode, Relation

from src.config import ChunkingConfig, Neo4jConfig
from src.llm_factory import build_llm_from_cfg_dict
from src.graph_prompts import LOW_LEVEL_EXTRACT_PROMPT
from src.utils.config_utils import load_config
from src.utils.json_utils import safe_json_loads
from src.utils.log_utils import debug, err, log, set_debug, warn
from src.utils.neo4j_utils import build_graph_store, reset_graph_store
from src.utils.text_utils import normalize_text, sliding_window_chunks


def main() -> None:
    cfg = load_config()

    run = cfg["run"]
    set_debug(run["debug"])

    input_path = run["input_path"]
    book_id = run["book_id"]
    reset = run["reset"]
    save_jsonl = run["save_jsonl"]
    limit_chunks = run["limit_chunks"]
    parse_max_retries = run["parse_max_retries"]
    retry_backoff_s = run["retry_backoff_s"]

    chunking = cfg["chunking"]
    chunk_cfg = ChunkingConfig(
        chunk_chars=chunking["chunk_chars"],
        chunk_overlap=chunking["chunk_overlap"],
    )

    neo = cfg["neo4j"]
    neo_cfg = Neo4jConfig(
        url=neo["uri"],
        user=neo["username"],
        password=neo["password"],
        database=neo["database"],
    )

    llm, llm_name = build_llm_from_cfg_dict(cfg)
    log(f"LLM initialized: {llm_name}")

    # Read + chunk
    raw = Path(input_path).read_text(encoding="utf-8")
    text = normalize_text(raw)
    chunks = sliding_window_chunks(text, chunk_cfg.chunk_chars, chunk_cfg.chunk_overlap)

    if limit_chunks is not None:
        chunks = chunks[: int(limit_chunks)]

    log("Starting low-level graph extraction")
    log(f"input_path={input_path}")
    log(f"book_id={book_id} | chunks={len(chunks)} | chunk_chars={chunk_cfg.chunk_chars} | overlap={chunk_cfg.chunk_overlap}")
    log(f"reset={reset} | save_jsonl={save_jsonl}")

    graph_store = build_graph_store(neo_cfg)
    if reset:
        log("Resetting Neo4j graph store (DELETE all nodes/edges)")
        reset_graph_store(graph_store)

    jsonl_f = None
    if save_jsonl:
        Path(save_jsonl).parent.mkdir(parents=True, exist_ok=True)
        jsonl_f = open(save_jsonl, "w", encoding="utf-8")

    try:
        for c in tqdm(chunks, desc="Extracting + Upserting"):
            debug(f"[chunk {c.chunk_id}] chars={len(c.text)}")

            prompt = LOW_LEVEL_EXTRACT_PROMPT.format(chunk=c.text)

            parsed = None
            last_e: Exception | None = None

            for attempt in range(parse_max_retries + 1):
                try:
                    resp = llm.complete(prompt)
                    parsed = safe_json_loads(str(resp))
                    break
                except Exception as e:
                    last_e = e
                    parsed = None
                    warn(f"[chunk {c.chunk_id}] parse attempt {attempt+1}/{parse_max_retries+1} failed: {repr(e)}")
                    if attempt < parse_max_retries:
                        time.sleep(retry_backoff_s)

            if parsed is None:
                raise RuntimeError(f"Failed to parse JSON for chunk {c.chunk_id}: {repr(last_e)}")

            if jsonl_f:
                jsonl_f.write(
                    json.dumps(
                        {
                            "book_id": book_id,
                            "chunk_id": c.chunk_id,
                            "char_start": c.char_start,
                            "char_end": c.char_end,
                            "extraction": parsed,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            entities = parsed.get("entities", []) or []
            relations = parsed.get("relations", []) or []

            # 1) upsert chunk/source node
            chunk_node = TextNode(
                text=c.text,
                metadata={
                    "book_id": book_id,
                    "chunk_id": c.chunk_id,
                    "char_start": c.char_start,
                    "char_end": c.char_end,
                    "source_path": os.path.abspath(input_path),
                },
            )
            graph_store.upsert_llama_nodes([chunk_node])

            # 2) entity nodes
            entity_nodes: Dict[str, EntityNode] = {}
            for ent in entities:
                ent_id = str(ent.get("id", "")).strip()
                ent_name = str(ent.get("name", "")).strip()
                ent_type = str(ent.get("type", "")).strip().upper()
                ent_desc = str(ent.get("description", "")).strip()

                if not ent_id or not ent_name or not ent_type:
                    continue

                node = EntityNode(
                    label=ent_type,
                    name=ent_id,
                    properties={
                        "display_name": ent_name,
                        "description": ent_desc,
                        "book_id": book_id,
                    },
                )
                entity_nodes[ent_id] = node

            if entity_nodes:
                graph_store.upsert_nodes(list(entity_nodes.values()))

            # 3) relations + mentions
            rel_objs: List[Relation] = []

            for ent_id, node in entity_nodes.items():
                rel_objs.append(
                    Relation(
                        label="MENTIONS",
                        source_id=chunk_node.node_id,
                        target_id=node.id,
                        properties={"chunk_id": c.chunk_id, "book_id": book_id},
                    )
                )

            for rel in relations:
                src = str(rel.get("source", "")).strip()
                typ = str(rel.get("type", "")).strip().upper()
                tgt = str(rel.get("target", "")).strip()
                desc = str(rel.get("description", "")).strip()
                ev = str(rel.get("evidence", "")).strip()

                if not src or not typ or not tgt:
                    continue
                if src not in entity_nodes or tgt not in entity_nodes:
                    continue

                rel_objs.append(
                    Relation(
                        label=typ,
                        source_id=entity_nodes[src].id,
                        target_id=entity_nodes[tgt].id,
                        properties={
                            "description": desc,
                            "evidence": ev,
                            "chunk_id": c.chunk_id,
                            "book_id": book_id,
                        },
                    )
                )

            if rel_objs:
                graph_store.upsert_relations(rel_objs)

        stats = graph_store.structured_query("MATCH (n) RETURN count(n) AS n_cnt")
        rel_stats = graph_store.structured_query("MATCH ()-[r]->() RETURN count(r) AS r_cnt")
        log(f"Neo4j stats: nodes={stats} rels={rel_stats}")

    finally:
        if jsonl_f:
            jsonl_f.close()


if __name__ == "__main__":
    main()
