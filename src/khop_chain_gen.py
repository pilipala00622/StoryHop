from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from neo4j import GraphDatabase

from src.utils.config_utils import load_config
from src.utils.log_utils import log, warn, debug, set_debug


# =========================
# Neo4j helpers
# =========================
def neo4j_driver(cfg: Dict[str, Any]):
    neo = cfg["neo4j"]
    return GraphDatabase.driver(neo["uri"], auth=(neo["username"], neo["password"]))


def run_cypher(session, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    res = session.run(query, params or {})
    return [r.data() for r in res]


# =========================
# Cypher: sample paths + checks
# =========================
def cypher_sample_khop_paths(k: int) -> str:
    return f"""
    MATCH p=(s)-[rels*{k}]->(t)
    WHERE
      elementId(s) <> elementId(t)
      AND s.book_id = $book_id
      AND t.book_id = $book_id
      AND ALL(
        r IN rels WHERE
          r.book_id = $book_id
          AND NOT type(r) IN $exclude_rel_types
          AND toInteger(r.chunk_id) IS NOT NULL
      )
      AND (
        size($start_labels) = 0 OR ANY(l IN labels(s) WHERE l IN $start_labels)
      )
      AND (
        size($end_labels) = 0 OR ANY(l IN labels(t) WHERE l IN $end_labels)
      )
    WITH p, s, t, rels,
         [r IN rels | toInteger(r.chunk_id)] AS chunk_ids
    WITH p, s, t, rels, chunk_ids,
         reduce(acc = [], x IN chunk_ids |
           CASE WHEN x IN acc THEN acc ELSE acc + x END
         ) AS uniq_chunk_ids
    WHERE size(uniq_chunk_ids) >= $min_distinct_chunks
    RETURN
      elementId(s) AS s_eid,
      elementId(t) AS t_eid,

      coalesce(s.display_name, s.name, s.id, elementId(s)) AS s_name,
      labels(s) AS s_labels,

      coalesce(t.display_name, t.name, t.id, elementId(t)) AS t_name,
      labels(t) AS t_labels,

      [n IN nodes(p) | coalesce(n.display_name, n.name, n.id, elementId(n))] AS node_names,
      [n IN nodes(p) | labels(n)] AS node_labels,

      [r IN rels | type(r)] AS rel_types,
      [r IN rels | coalesce(r.description, '')] AS rel_descs,
      [r IN rels | coalesce(r.evidence, '')] AS evidences,
      [r IN rels | toInteger(r.chunk_id)] AS chunk_ids
    ORDER BY rand()
    LIMIT $limit
    """


def cypher_exists_shorter_path(max_len: int) -> str:
    return f"""
    MATCH p=(s)-[rels*1..{max_len}]->(t)
    WHERE
      elementId(s) = $s_eid
      AND elementId(t) = $t_eid
      AND ALL(r IN rels WHERE r.book_id = $book_id AND NOT type(r) IN $exclude_rel_types)
    RETURN 1 AS exists
    LIMIT 1
    """


def cypher_count_khop_paths(k: int) -> str:
    return f"""
    MATCH p=(s)-[rels*{k}]->(t)
    WHERE
      elementId(s) = $s_eid
      AND elementId(t) = $t_eid
      AND ALL(r IN rels WHERE r.book_id = $book_id AND NOT type(r) IN $exclude_rel_types)
    WITH p LIMIT 2
    RETURN count(p) AS c
    """


# =========================
# Formatting: chain
# =========================
def build_chain_object(row: Dict[str, Any]) -> Dict[str, Any]:
    node_names = row["node_names"]
    node_labels = row["node_labels"]
    rel_types = row["rel_types"]
    rel_descs = row["rel_descs"]
    evidences = row["evidences"]
    chunk_ids = row["chunk_ids"]

    steps: List[Dict[str, Any]] = []
    for i in range(len(rel_types)):
        steps.append(
            {
                "hop": i + 1,
                "source": {"name": node_names[i], "labels": node_labels[i]},
                "relation": {"type": rel_types[i], "description": rel_descs[i]},
                "target": {"name": node_names[i + 1], "labels": node_labels[i + 1]},
                "evidence": evidences[i],
                "chunk_id": chunk_ids[i],
            }
        )

    ref_chunks: List[int] = []
    for cid in chunk_ids:
        if cid not in ref_chunks:
            ref_chunks.append(cid)

    return {
        "start": {"name": node_names[0], "labels": node_labels[0]},
        "end": {"name": node_names[-1], "labels": node_labels[-1]},
        "steps": steps,
        "ref_chunks": ref_chunks,
    }


# =========================
# Acceptance checks
# =========================
def has_shorter_path(session, book_id: str, exclude_rel_types: List[str], s_eid: str, t_eid: str, k: int) -> bool:
    if k <= 1:
        return False
    q = cypher_exists_shorter_path(k - 1)
    rows = run_cypher(
        session,
        q,
        {"book_id": book_id, "exclude_rel_types": exclude_rel_types, "s_eid": s_eid, "t_eid": t_eid},
    )
    return len(rows) > 0


def unique_khop(session, book_id: str, exclude_rel_types: List[str], s_eid: str, t_eid: str, k: int) -> bool:
    q = cypher_count_khop_paths(k)
    rows = run_cypher(
        session,
        q,
        {"book_id": book_id, "exclude_rel_types": exclude_rel_types, "s_eid": s_eid, "t_eid": t_eid},
    )
    return int(rows[0]["c"]) == 1


# =========================
# Cypher param injection for "complete query"
# =========================
def cypher_quote(s: str) -> str:
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def cypher_list_str(xs: List[str]) -> str:
    return "[" + ", ".join(cypher_quote(x) for x in xs) + "]"


def build_full_visualize_query(k: int, book_id: str, exclude_rel_types: List[str], s_eid: str, t_eid: str) -> str:
    return (
        f"MATCH p=(s)-[rels*{k}]->(t)\n"
        f"WHERE elementId(s) = {cypher_quote(s_eid)}\n"
        f"  AND elementId(t) = {cypher_quote(t_eid)}\n"
        f"  AND ALL(r IN rels WHERE r.book_id = {cypher_quote(book_id)} AND NOT type(r) IN {cypher_list_str(exclude_rel_types)})\n"
        f"RETURN p\n"
        f"LIMIT 1"
    )


# =========================
# Dedupe against existing JSONL
# =========================
def chain_signature_from_chain(chain: Dict[str, Any]) -> Tuple:
    """
    Content-based signature. Prevents repeating the same chain across runs even if IDs differ.
    Uses ordered hops: (src_name, rel_type, tgt_name, chunk_id) repeated over hops.
    """
    sig: List[Any] = []
    for st in chain["steps"]:
        sig.extend(
            [
                st["source"]["name"],
                st["relation"]["type"],
                st["target"]["name"],
                st["chunk_id"],
            ]
        )
    return tuple(sig)


def load_existing_state(output_jsonl: str, book_id: str):
    """
    Scan the existing JSONL once. Build per-k state:
      seen_pairs_by_k[k] = {(s_eid,t_eid),...}
      seen_sigs_by_k[k]  = {sig,...}
      next_index_by_k[k] = next chain_id suffix to use for that k
      existing_count_by_k[k] = count
    """
    seen_pairs_by_k: Dict[int, set] = {}
    seen_sigs_by_k: Dict[int, set] = {}
    max_suffix_by_k: Dict[int, int] = {}
    existing_count_by_k: Dict[int, int] = {}

    if not output_jsonl or not os.path.exists(output_jsonl):
        return seen_pairs_by_k, seen_sigs_by_k, {}, existing_count_by_k

    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            it = json.loads(line)
            if it["book_id"] != book_id:
                continue

            kk = int(it["k"])

            existing_count_by_k[kk] = existing_count_by_k.get(kk, 0) + 1
            seen_pairs_by_k.setdefault(kk, set())
            seen_sigs_by_k.setdefault(kk, set())
            max_suffix_by_k.setdefault(kk, -1)

            meta = it["meta"]
            s_eid = meta.get("s_eid")
            t_eid = meta.get("t_eid")
            if s_eid and t_eid:
                seen_pairs_by_k[kk].add((s_eid, t_eid))

            chain = it.get("chain")
            if chain:
                seen_sigs_by_k[kk].add(chain_signature_from_chain(chain))

            cid = it.get("chain_id", "")
            prefix = f"{book_id}::k{kk}::"
            if isinstance(cid, str) and cid.startswith(prefix):
                suf = cid[len(prefix):]
                max_suffix_by_k[kk] = max(max_suffix_by_k[kk], int(suf))

    next_index_by_k = {kk: max_suffix_by_k.get(kk, -1) + 1 for kk in max_suffix_by_k.keys()}
    return seen_pairs_by_k, seen_sigs_by_k, next_index_by_k, existing_count_by_k


def normalize_k_list(k_value: Any) -> List[int]:
    if isinstance(k_value, list):
        return [int(x) for x in k_value]
    return [int(k_value)]


# =========================
# main
# =========================
def main() -> None:
    cfg = load_config()
    set_debug(cfg["run"]["debug"])

    kh = cfg["khop"]
    if not kh["enabled"]:
        log("khop.enabled=false; exiting")
        return

    book_id = cfg["run"]["book_id"]
    k_list = normalize_k_list(kh["k"])
    num_chains = int(kh["num_chains"])  # per k

    candidate_limit = int(kh["candidate_limit"])
    max_sampling_tries = int(kh["max_sampling_tries"])

    enforce_no_shorter = bool(kh["enforce_no_shorter_path"])
    enforce_unique = bool(kh["enforce_unique_khop_path"])
    min_distinct_chunks = int(kh["min_distinct_chunks"])

    exclude_rel_types = kh["exclude_rel_types"]
    start_labels = kh["start_labels"]
    end_labels = kh["end_labels"]

    output_jsonl = kh["output_jsonl"]
    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)) or ".", exist_ok=True)

    random.seed(int(kh["seed"]))

    log("k-hop chain sampler (append mode, multi-k)")
    log(f"book_id={book_id} | k_list={k_list} | num_chains_per_k={num_chains}")
    log(f"candidate_limit={candidate_limit} | max_sampling_tries={max_sampling_tries}")
    log(f"exclude_rel_types={exclude_rel_types} | min_distinct_chunks={min_distinct_chunks}")
    log(f"start_labels={start_labels} | end_labels={end_labels}")
    log(f"enforce_no_shorter={enforce_no_shorter} | enforce_unique={enforce_unique}")
    log(f"output_jsonl={output_jsonl}")

    seen_pairs_by_k, seen_sigs_by_k, next_index_by_k, existing_count_by_k = load_existing_state(output_jsonl, book_id)
    seen_pairs_global = set().union(*seen_pairs_by_k.values()) if seen_pairs_by_k else set()

    for kk in k_list:
        log(f"[existing] k={kk} count={existing_count_by_k.get(kk, 0)} next_index={next_index_by_k.get(kk, 0)}")

    driver = neo4j_driver(cfg)
    database = cfg["neo4j"]["database"]

    accepted_all: List[Dict[str, Any]] = []

    with driver.session(database=database) as session:
        for k in k_list:
            seen_pairs_by_k.setdefault(k, set())
            seen_sigs_by_k.setdefault(k, set())
            next_index_by_k.setdefault(k, 0)

            sample_q = cypher_sample_khop_paths(k)
            accepted_new_k: List[Dict[str, Any]] = []

            tries = 0
            while len(accepted_new_k) < num_chains and tries < max_sampling_tries:
                tries += 1
                debug(f"[k={k}] [try {tries}/{max_sampling_tries}] sampling candidates...")

                rows = run_cypher(
                    session,
                    sample_q,
                    {
                        "book_id": book_id,
                        "exclude_rel_types": exclude_rel_types,
                        "start_labels": start_labels,
                        "end_labels": end_labels,
                        "min_distinct_chunks": min_distinct_chunks,
                        "limit": candidate_limit,
                    },
                )

                if not rows:
                    warn(f"[k={k}] No candidate paths found. Stopping this k.")
                    break

                random.shuffle(rows)

                for row in rows:
                    if len(accepted_new_k) >= num_chains:
                        break

                    s_eid = row.get("s_eid")
                    t_eid = row.get("t_eid")
                    if not s_eid or not t_eid:
                        continue

                    # DEDUPE: existing or already accepted in this run (for this k)
                    if (s_eid, t_eid) in seen_pairs_by_k[k]:
                    # if (s_eid, t_eid) in seen_pairs_global:
                        continue

                    if enforce_no_shorter and has_shorter_path(session, book_id, exclude_rel_types, s_eid, t_eid, k):
                        continue

                    if enforce_unique and (not unique_khop(session, book_id, exclude_rel_types, s_eid, t_eid, k)):
                        continue

                    chain = build_chain_object(row)

                    sig = chain_signature_from_chain(chain)
                    if sig in seen_sigs_by_k[k]:
                        continue

                    # Accept => update dedupe state immediately
                    seen_pairs_by_k[k].add((s_eid, t_eid))
                    seen_pairs_global.add((s_eid, t_eid))

                    seen_sigs_by_k[k].add(sig)

                    full_vis_q = build_full_visualize_query(
                        k=k,
                        book_id=book_id,
                        exclude_rel_types=exclude_rel_types,
                        s_eid=s_eid,
                        t_eid=t_eid,
                    )

                    chain_id = f"{book_id}::k{k}::{(next_index_by_k[k] + len(accepted_new_k)):06d}"
                    item = {
                        "task": "khop_chain",
                        "book_id": book_id,
                        "k": k,
                        "chain_id": chain_id,
                        "chain": chain,
                        "final_answer": chain["end"]["name"],
                        "meta": {
                            "s_eid": s_eid,
                            "t_eid": t_eid,
                            "s_name": row.get("s_name"),
                            "t_name": row.get("t_name"),
                            "s_labels": row.get("s_labels"),
                            "t_labels": row.get("t_labels"),
                        },
                        "cypher_visualize_full": full_vis_q,
                    }

                    accepted_new_k.append(item)
                    accepted_all.append(item)

                    log(f"[k={k}] [ACCEPT {len(accepted_new_k)}/{num_chains}] {item['meta']['s_name']} -> {item['meta']['t_name']}")
                    print("Gold chain:", " -> ".join([chain["start"]["name"]] + [st["target"]["name"] for st in chain["steps"]]))
                    print("Ref chunks:", chain["ref_chunks"])
                    print("---- FULL CYPHER (paste into Aura Browser) ----")
                    print(item["cypher_visualize_full"])
                    print("=" * 100)

            # bump next index for this k so future runs continue correctly
            next_index_by_k[k] = next_index_by_k[k] + len(accepted_new_k)
            log(f"[k={k}] done. new_accepted={len(accepted_new_k)} tries={tries}")

    driver.close()

    # APPEND mode
    if accepted_all:
        with open(output_jsonl, "a", encoding="utf-8") as f:
            for it in accepted_all:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        log(f"Appended {len(accepted_all)} chains total to: {output_jsonl}")
    else:
        log("No new chains accepted; nothing appended.")


if __name__ == "__main__":
    main()
