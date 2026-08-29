"""Legal neighbor enumeration for ordinal fields (proposal v2 §2)."""

from __future__ import annotations

import numpy as np


def adjacent_edges(cats: list[str]) -> list[tuple[int, int]]:
    """Synthesis edges: chain over the ordinal order, directed low->high."""
    return [(j, j + 1) for j in range(len(cats) - 1)]


def cycle_edges(cats: list[str]) -> list[tuple[int, int]]:
    """Diagnostic skip-one edges (A<->C triangles); never synthesis edges."""
    return [(j, j + 2) for j in range(len(cats) - 2)]


def neighbors_of(idx: int, cats: list[str]) -> list[dict]:
    """Legal neighbors of an Actor answer at category index ``idx``."""
    out = []
    if idx - 1 >= 0:
        out.append({"neighbor_id": f"nb_{idx - 1}", "edit_type": "adjacent_category",
                    "candidate": {None: cats[idx - 1]}, "cat_index": idx - 1})
    if idx + 1 < len(cats):
        out.append({"neighbor_id": f"nb_{idx + 1}", "edit_type": "adjacent_category",
                    "candidate": {None: cats[idx + 1]}, "cat_index": idx + 1})
    return out


def pick_edges_for_unit(actor_idx: int, cats: list[str], devil_ids: list[str],
                        rng: np.random.Generator) -> list[dict]:
    """Assemble the query set: devil-selected + 1 random + 1 cycle edge.

    Edges are ``(j, k)`` with j < k (directed low->high), tagged with source.
    """
    adj = adjacent_edges(cats)
    cyc = cycle_edges(cats)
    by_nb = {}
    for nb in neighbors_of(actor_idx, cats):
        j, k = sorted((actor_idx, nb["cat_index"]))
        by_nb[nb["neighbor_id"]] = (j, k)

    chosen: list[tuple[tuple[int, int], str]] = []
    for nid in devil_ids:
        if nid in by_nb and by_nb[nid] not in [e for e, _ in chosen]:
            chosen.append((by_nb[nid], "devil"))
    dev_e = {e for e, _ in chosen}
    rest_adj = [e for e in adj if e not in dev_e]
    if rest_adj:
        e = rest_adj[int(rng.integers(len(rest_adj)))]
        chosen.append((e, "random"))
    if cyc:
        e = cyc[int(rng.integers(len(cyc)))]
        chosen.append((e, "cycle"))
    seen = set()
    out = []
    for e, src in chosen:
        if e in seen:
            continue
        seen.add(e)
        out.append({"edge": e, "source": src})
    return out
