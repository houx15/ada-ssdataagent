"""Unit execution: Devil selection + Blind Arbiter forward/reversed calls."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np

from ssbench.ada.prompts import (
    ARBITER_SYSTEM,
    CHALLENGE_CODES,
    DEVIL_SYSTEM,
    arbiter_user_prompt,
    devil_user_prompt,
)
from ssbench.llm.client import LLMClient

MIN_PROB = 0.05
EPS = 1e-3


@dataclass
class UnitResult:
    ds: str
    field: str
    persona_id: int
    actor_idx: int
    edges: list[dict] = field(default_factory=list)  # query plan with source
    devil_valid: bool = False
    devil_error: str | None = None
    arbiter_valid: int = 0
    arbiter_total: int = 0
    g: dict = field(default_factory=dict)  # (j,k) -> antisym response
    o: dict = field(default_factory=dict)  # position bias per edge
    audit: list = field(default_factory=list)


def _parse_json(content: str):
    return json.loads(content)


def run_unit(
    client: LLMClient,
    ds: str,
    fld: str,
    persona_id: int,
    actor_idx: int,
    cats: list[str],
    historical_context: str,
    persona_json: dict,
    target_schema_json: dict,
    max_challenges: int,
    rng: np.random.Generator,
) -> UnitResult:
    from ssbench.ada.neighbors import neighbors_of, pick_edges_for_unit

    res = UnitResult(ds=ds, field=fld, persona_id=persona_id, actor_idx=actor_idx)
    first_profile = {fld: cats[actor_idx]}
    nbs = neighbors_of(actor_idx, cats)

    # ---- Devil ----
    devil_ids: list[str] = []
    try:
        r = client.chat(
            DEVIL_SYSTEM,
            devil_user_prompt(historical_context, persona_json, target_schema_json,
                              first_profile, nbs, max_challenges),
        )
        res.audit.append({"role": "devil", "content": r.content,
                          "finish_reason": r.finish_reason, "usage": r.usage,
                          "ts": time.time()})
        out = _parse_json(r.content)
        challenges = out.get("challenges", [])
        no_valid = bool(out.get("no_valid_challenge", False))
        legal_ids = {nb["neighbor_id"] for nb in nbs}
        priorities = []
        for ch in challenges:
            if ch.get("neighbor_id") not in legal_ids:
                continue
            if ch.get("challenge_code") not in CHALLENGE_CODES:
                continue
            if int(ch.get("changed_factor_count", 0)) != 1:
                continue
            priorities.append(ch.get("priority"))
            devil_ids.append(ch["neighbor_id"])
        mutually_exclusive = (len(challenges) == 0) == no_valid
        res.devil_valid = bool(mutually_exclusive and len(priorities) == len(set(priorities)))
        if not res.devil_valid:
            res.devil_error = f"validation_failed (n_challenges={len(challenges)}, no_valid={no_valid})"
        devil_ids = devil_ids[:max_challenges]
    except Exception as e:  # noqa: BLE001
        res.devil_error = f"{type(e).__name__}: {e}"
        res.audit.append({"role": "devil", "content": None, "error": str(e), "ts": time.time()})

    edges = pick_edges_for_unit(actor_idx, cats, devil_ids, rng)
    res.edges = edges
    if not edges:
        return res

    # ---- Build anonymized pair batch ----
    comps = []
    for t, e in enumerate(edges):
        j, k = e["edge"]
        flip = bool(rng.integers(2))
        a, b = (k, j) if flip else (j, k)
        comps.append({
            "comparison_id": f"cmp_{t:02d}",
            "candidate_A": {fld: cats[a]},
            "candidate_B": {fld: cats[b]},
            "_edge": (j, k),
            "_flip": flip,
        })
    order = rng.permutation(len(comps))
    fwd = [comps[i] | {"_pos": i} for i in order]

    def call_arbiter(batch, tag):
        payload = [
            {"comparison_id": c["comparison_id"],
             "candidate_A": c["candidate_A"], "candidate_B": c["candidate_B"]}
            for c in batch
        ]
        r = client.chat(
            ARBITER_SYSTEM,
            arbiter_user_prompt(historical_context, persona_json,
                                {"note": "两个候选除目标字段外完全相同"}, payload, MIN_PROB),
        )
        res.audit.append({"role": f"arbiter_{tag}", "content": r.content,
                          "finish_reason": r.finish_reason, "usage": r.usage,
                          "ts": time.time()})
        out = _parse_json(r.content)
        got = {}
        for c in out.get("comparisons", []):
            got[c.get("comparison_id")] = c
        return got

    try:
        fwd_resp = call_arbiter(fwd, "fwd")
        rev = [
            {**c, "candidate_A": c["candidate_B"], "candidate_B": c["candidate_A"]}
            for c in fwd
        ]
        order2 = rng.permutation(len(rev))
        rev = [rev[i] for i in order2]
        rev_resp = call_arbiter(rev, "rev")
    except Exception as e:  # noqa: BLE001
        res.audit.append({"role": "arbiter", "content": None, "error": str(e), "ts": time.time()})
        return res

    # ---- Antisymmetric responses ----
    for c in fwd:
        cid = c["comparison_id"]
        j, k = c["_edge"]
        f, rv = fwd_resp.get(cid), rev_resp.get(cid)
        if not f or not rv:
            continue
        try:
            if not (f.get("valid_comparison", True) and rv.get("valid_comparison", True)):
                continue
            pa_f = min(max(float(f["probability_A"]), MIN_PROB), 1 - MIN_PROB)
            pa_r = min(max(float(rv["probability_A"]), MIN_PROB), 1 - MIN_PROB)
            if abs(float(f["probability_A"]) + float(f["probability_B"]) - 1) > 0.02:
                continue
            if abs(float(rv["probability_A"]) + float(rv["probability_B"]) - 1) > 0.02:
                continue
        except (KeyError, TypeError, ValueError):
            continue
        res.arbiter_total += 1
        # log-odds of B relative to A, as presented
        l_f = np.log((1 - pa_f + EPS) / (pa_f + EPS))
        l_r = np.log((pa_r + EPS) / (1 - pa_r + EPS))
        g_ab = (l_f - l_r) / 2  # antisymmetric part, B vs A as presented
        o_ab = (l_f + l_r) / 2  # position bias (should be ~0)
        # convert to low->high direction
        g_jk = -g_ab if c["_flip"] else g_ab
        res.g[(j, k)] = float(g_jk)
        res.o[(j, k)] = float(o_ab)
        res.arbiter_valid += 1
    return res
