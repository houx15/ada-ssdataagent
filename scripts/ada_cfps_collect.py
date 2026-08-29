#!/usr/bin/env python3
"""Collect ADA-Observer measurements on CFPS — per-persona all-fields mode.

Per persona (3 LLM calls): 1 Devil (select <=8 edges across all fields) +
1 forward Arbiter batch (16 anonymized pairs) + 1 reversed batch.
Resumable: one JSONL line per persona.

Usage:
  uv run python scripts/ada_cfps_collect.py --n-personas 1000
  uv run python scripts/ada_cfps_collect.py --resume-dir runs/ada/cfps_<stamp>

Round-2 adaptive mode: measure the round-1-corrected population instead of the
direct Actor output, so Devil/Arbiter probe the (now diversified) neighbourhood
structure that round 1 could not see (e.g. minority event orders):

  uv run python scripts/ada_cfps_collect.py --n-personas 1000 \\
      --arbiter-scale logodds --grid nice --order-pairs \\
      --actor-sim runs/ada/<round1>/run_order/sim.csv \\
      --levels-from runs/ada/<round1>/levels.json

Levels must be reused from round 1 (identical node indexing) so that edges
from both rounds can be pooled (scripts/ada_pool_rounds.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ssbench.ada.prompts import (  # noqa: E402
    ARBITER_SYSTEM,
    ARBITER_SYSTEM_LO,
    CHALLENGE_CODES,
    DEVIL_SYSTEM,
    arbiter_user_prompt,
    arbiter_user_prompt_lo,
    devil_user_prompt,
)
from ssbench.evaluation.cleaning import prep_variable  # noqa: E402
from ssbench.llm.client import LLMClient  # noqa: E402
from ssbench.settings import get_settings  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUN = os.path.join(ROOT, "runs", "cfps", "direct", "20260817_164329_glm52")
HIST = ("China, CFPS-style life-course panel. A randomly sampled adult respondent "
        "followed from age 14 to 45.")
INPUT_COLS = ["gender", "minzu", "mother_education", "father_education"]
TEMPERATURE = 0.3
MAX_CHALLENGES = 8   # devil-selected adjacent edges across all fields
N_RANDOM = 4         # random adjacent edges
N_CYCLE = 4          # skip-one cycle diagnostic edges
MIN_PROB = 0.05
EPS = 1e-3
MAX_LOG_ODDS = 10.0

# T4 order measurement: pairwise swaps of the three event ages.
ORDER_FIELDS = {"E": "age_finished_education",
                "M": "age_at_first_marriage",
                "C": "age_at_first_child"}
ORDER_STATES = ["".join(p) for p in __import__("itertools").permutations("EMC")]
ORDER_FIELD = "__order__"
N_ORDER = 3  # all three pairwise swaps
MAX_LEVELS = 8
N_QUANT_BINS = 6
CALLS_PER_PERSONA = 3


def nice_edges(vals: np.ndarray, target_bins: int = 5) -> list[float]:
    """Real-blind fixed grid. Small spans (bounded scales): integer/half-integer
    edges; large spans: 1/2/5-mantissa round numbers. No dependence on sim shape."""
    lo, hi = float(np.min(vals)), float(np.max(vals))
    span = hi - lo
    if span <= 0:
        return []
    if span <= 2 * target_bins:          # narrow bounded scale (Likert, scores)
        step = 0.5
        cands = [round(lo + i * step, 2) for i in range(1, int(span / step) + 1)]
    elif span <= 24:                      # small scale (0-6, 0-10, log10 units)
        cands = [float(v) for v in range(int(np.floor(lo)) + 1, int(np.ceil(hi)))]
    else:                                 # wide scale (ages, incomes)
        cands = [k * 10.0 ** j for j in range(-6, 7)
                 for k in (1, 2, 2.5, 5, 7.5) if lo < k * 10.0 ** j < hi]
        cands = sorted(set(cands))
    cands = [v for v in cands if lo < v < hi]
    if len(cands) < 2:
        qs = np.quantile(vals, np.arange(1, target_bins) / target_bins)
        return [float(e) for e in np.unique(qs)]
    n = min(target_bins - 1, len(cands))
    idx = np.round(np.linspace(0, len(cands) - 1, n)).astype(int)
    return [float(cands[i]) for i in sorted(set(idx))]


def build_levels(var: str, vcfg: dict, sim_vals: np.ndarray, grid: str = "nice"):
    t = (vcfg.get("type") or "").lower()
    if t == "categorical":
        cats = [c for c in vcfg["allowed"] if (sim_vals == c).any() or True]
        return cats, "categorical", {}
    vals = np.sort(np.unique(sim_vals))
    if len(vals) <= MAX_LEVELS:
        labels = [float(v) for v in vals]
        return labels, "numeric_unique", {"values": labels}
    if grid == "nice":
        edges = nice_edges(vals, N_QUANT_BINS)
    else:
        edges = list(np.unique(np.quantile(vals, np.arange(1, N_QUANT_BINS) / N_QUANT_BINS)))
    if len(edges) < 2:
        return None, None, {}
    full = np.concatenate([[vals.min() - 1e-9], edges, [np.inf]])
    labels = []
    for b in range(len(full) - 1):
        in_bin = vals[(vals > full[b]) & (vals <= full[b + 1])]
        raw = float(in_bin.mean()) if len(in_bin) else float((full[b] + full[b + 1]) / 2)
        mag = abs(raw)
        nd = 2 if mag >= 10 else (2 if mag >= 1 else 3)
        labels.append(round(raw, nd))
    return labels, "numeric_bin", {"edges": [float(e) for e in edges]}


def level_index(value, labels, kind: str, meta: dict):
    if kind == "categorical":
        return labels.index(value) if value in labels else None
    v = float(value)
    if kind == "numeric_unique":
        arr = np.asarray(labels)
        i = int(np.argmin(np.abs(arr - v)))
        return i if abs(arr[i] - v) < 1e-9 else None
    full = [-np.inf] + list(meta["edges"]) + [np.inf]
    for i in range(len(labels)):
        if full[i] < v <= full[i + 1]:
            return i
    return None


def fmt_val(v):
    if isinstance(v, float) and v == int(v):
        return int(v)
    return v


def legal_neighbors(actor_idxs: dict, levels: dict) -> list[dict]:
    """Flatten per-field adjacent neighbors of this persona's Actor answers."""
    nbs = []
    for var, idx in actor_idxs.items():
        lm = levels[var]
        labels = lm["labels"]
        m = len(labels)
        if idx - 1 >= 0:
            nbs.append({"neighbor_id": f"{var}__{idx - 1}", "field": var,
                        "edit_type": "adjacent_category",
                        "candidate": {var: labels[idx - 1]}, "cat_index": idx - 1})
        if idx + 1 < m:
            nbs.append({"neighbor_id": f"{var}__{idx + 1}", "field": var,
                        "edit_type": "adjacent_category",
                        "candidate": {var: labels[idx + 1]}, "cat_index": idx + 1})
    return nbs


def order_state(ages: dict) -> int | None:
    """Index of the order state (events sorted by age) from {"E":age,...}; None if tie/missing."""
    vals = {k: float(v) for k, v in ages.items()}
    if any(pd.isna(v) for v in vals.values()):
        return None
    a = sorted(vals.values())
    if any(abs(a[i] - a[i + 1]) < 1e-9 for i in range(len(a) - 1)):
        return None  # tie: swapping is identity
    s = "".join(k for k, _ in sorted(vals.items(), key=lambda kv: kv[1]))
    return ORDER_STATES.index(s)


def order_state_presentable(ages: dict) -> int | None:
    """Order state as the Arbiter will SEE it (ages rounded to 1 decimal).

    Round-1 reallocation jitters numeric values by <=1e-3, which can split an
    exact age tie (e.g. E==M==22) into a spurious sub-millipoint order. Derive
    the state from the presented (rounded) ages and require all three to be
    distinct, so no ambiguous swap pair is ever sent."""
    vals = {}
    for k, v in ages.items():
        if v is None or pd.isna(v):
            return None
        vals[k] = round(float(v), 1)
    a = sorted(vals.values())
    if any(abs(a[i] - a[i + 1]) < 1e-9 for i in range(len(a) - 1)):
        return None
    s = "".join(k for k, _ in sorted(vals.items(), key=lambda kv: kv[1]))
    return ORDER_STATES.index(s)


def swapped_state(state: str, x: str, y: str) -> str:
    t = list(state)
    i, j = t.index(x), t.index(y)
    t[i], t[j] = t[j], t[i]
    return "".join(t)


def edge_plan(actor_idxs: dict, levels: dict, devil_ids: list[str],
              rng: np.random.Generator,
              n_dev: int = None, n_rand: int = None, n_cycle: int = None) -> list[dict]:
    """(field, j, k) query set: devil adjacent + random adjacent + cycle skip-one."""
    n_dev = MAX_CHALLENGES if n_dev is None else n_dev
    n_rand = N_RANDOM if n_rand is None else n_rand
    n_cycle = N_CYCLE if n_cycle is None else n_cycle
    # map neighbor_id -> edge
    by_id = {}
    for nb in legal_neighbors(actor_idxs, levels):
        j, k = sorted((actor_idxs[nb["field"]], nb["cat_index"]))
        by_id[nb["neighbor_id"]] = (nb["field"], j, k)
    chosen, seen = [], set()
    for nid in devil_ids:
        if nid in by_id and by_id[nid] not in seen and len(chosen) < n_dev:
            chosen.append({"key": by_id[nid], "source": "devil"})
            seen.add(by_id[nid])
    rest = [v for v in by_id.values() if v not in seen]
    rng.shuffle(rest)
    for key in rest[:n_rand]:
        chosen.append({"key": key, "source": "random"})
        seen.add(key)
    cyc = []
    for var, idx in actor_idxs.items():
        m = len(levels[var]["labels"])
        for d in (-2, 2):
            t = idx + d
            if 0 <= t < m:
                j, k = sorted((idx, t))
                if (var, j, k) not in seen:
                    cyc.append((var, j, k))
    rng.shuffle(cyc)
    for key in cyc[:n_cycle]:
        chosen.append({"key": key, "source": "cycle"})
        seen.add(key)
    return chosen


def run_persona(client, pid, actor_idxs, persona, schema, levels, rng,
                scale: str = "prob", order: dict | None = None) -> dict:
    first_profile = {v: levels[v]["labels"][i] for v, i in actor_idxs.items()}
    nbs = legal_neighbors(actor_idxs, levels)
    rec = {"persona_id": pid, "devil_valid": False, "devil_error": None,
           "arb_valid": 0, "arb_total": 0, "edges": [], "g": {}, "audit": []}

    # ---- Devil ----
    devil_ids: list[str] = []
    try:
        r = client.chat(
            DEVIL_SYSTEM,
            devil_user_prompt(HIST, persona, schema, first_profile, nbs, MAX_CHALLENGES),
        )
        rec["audit"].append({"role": "devil", "content": r.content,
                             "finish_reason": r.finish_reason, "usage": r.usage})
        out = json.loads(r.content)
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
        rec["devil_valid"] = bool(
            (len(challenges) == 0) == no_valid and len(priorities) == len(set(priorities)))
        if not rec["devil_valid"]:
            rec["devil_error"] = f"validation_failed (n={len(challenges)}, no_valid={no_valid})"
        devil_ids = devil_ids[:MAX_CHALLENGES]
    except Exception as e:  # noqa: BLE001
        rec["devil_error"] = f"{type(e).__name__}: {e}"
        rec["audit"].append({"role": "devil", "content": None, "error": str(e)})
        return rec

    use_order = order is not None and order.get("state_idx") is not None
    budgets = (MAX_CHALLENGES - 1, N_RANDOM - 1, N_CYCLE - 1) if use_order else (None, None, None)
    plan = edge_plan(actor_idxs, levels, devil_ids, rng, *budgets)
    rec["edges"] = [{"field": k[0], "edge": [k[1], k[2]], "source": s}
                    for k, s in ((p["key"], p["source"]) for p in plan)]

    # ---- Arbiter forward ----
    comps = []
    for t, p in enumerate(plan):
        var, j, k = p["key"]
        labels = levels[var]["labels"]
        flip = bool(rng.integers(2))
        a, b = (k, j) if flip else (j, k)
        comps.append({"comparison_id": f"cmp_{t:02d}",
                      "candidate_A": {var: labels[a]},
                      "candidate_B": {var: labels[b]},
                      "_key": (var, j, k), "_flip": flip})
    order_comps = []
    if use_order:
        s0 = ORDER_STATES[order["state_idx"]]
        ages = dict(order["ages"])
        i0 = ORDER_STATES.index(s0)
        for x, y in (("E", "M"), ("M", "C"), ("E", "C")):
            s1 = swapped_state(s0, x, y)
            i1 = ORDER_STATES.index(s1)
            age1 = dict(ages)
            age1[x], age1[y] = ages[y], ages[x]
            by_state = {i0: ages, i1: age1}
            j, k = sorted((i0, i1))
            t = len(comps) + len(order_comps)
            flip = bool(rng.integers(2))
            fmt = lambda a: {ORDER_FIELDS[e]: round(float(a[e]), 1) for e in "EMC"}
            order_comps.append({
                "comparison_id": f"cmp_{t:02d}",
                "cand_j": fmt(by_state[j]), "cand_k": fmt(by_state[k]),
                "_key": (ORDER_FIELD, j, k), "_flip": flip})
    all_comps = comps + order_comps
    if not all_comps:
        return rec
    perm = rng.permutation(len(all_comps))
    fwd = [all_comps[i] for i in perm]

    def pair_payload(c, reverse: bool = False):
        if c["_key"][0] == ORDER_FIELD:
            cj, ck = c["cand_j"], c["cand_k"]
            a, b = (ck, cj) if c["_flip"] else (cj, ck)
        else:
            a, b = c["candidate_A"], c["candidate_B"]
        if reverse:
            a, b = b, a
        return {"comparison_id": c["comparison_id"],
                "candidate_A": a, "candidate_B": b}

    note = ("每对候选只在单一因素上不同：或单个字段的取值，"
            "或三个事件年龄中两个的先后顺序互换（年龄数值多重集不变）；其余完全相同"
            if use_order else "每对候选除单个目标字段外完全相同")

    def call_arbiter(batch, tag, reverse=False):
        payload = [pair_payload(c, reverse) for c in batch]
        if scale == "logodds":
            r = client.chat(ARBITER_SYSTEM_LO,
                            arbiter_user_prompt_lo(HIST, persona, {"note": note},
                                                    payload, MAX_LOG_ODDS))
        else:
            r = client.chat(ARBITER_SYSTEM,
                            arbiter_user_prompt(HIST, persona, {"note": note},
                                                payload, MIN_PROB))
        rec["audit"].append({"role": f"arbiter_{tag}", "content": r.content,
                             "finish_reason": r.finish_reason, "usage": r.usage})
        out = json.loads(r.content)
        return {c.get("comparison_id"): c for c in out.get("comparisons", [])}

    try:
        fwd_resp = call_arbiter(fwd, "fwd")
        rev_resp = call_arbiter([fwd[i] for i in rng.permutation(len(fwd))],
                                "rev", reverse=True)
    except Exception as e:  # noqa: BLE001
        rec["audit"].append({"role": "arbiter", "content": None, "error": str(e)})
        return rec

    # ---- antisymmetric edge responses ----
    for c in fwd:
        f, rv = fwd_resp.get(c["comparison_id"]), rev_resp.get(c["comparison_id"])
        if not f or not rv:
            continue
        try:
            if not (f.get("valid_comparison", True) and rv.get("valid_comparison", True)):
                continue
            if scale == "logodds":
                lo_f = float(f["log_odds_A_vs_B"])
                lo_r = float(rv["log_odds_A_vs_B"])
                if not (np.isfinite(lo_f) and np.isfinite(lo_r)):
                    continue
                lo_f = min(max(lo_f, -MAX_LOG_ODDS), MAX_LOG_ODDS)
                lo_r = min(max(lo_r, -MAX_LOG_ODDS), MAX_LOG_ODDS)
                # log-odds of B relative to A as presented (matches prob branch)
                l_f, l_r = -lo_f, -lo_r
            else:
                pa_f = float(f["probability_A"])
                pa_r = float(rv["probability_A"])
                if abs(pa_f + float(f["probability_B"]) - 1) > 0.02:
                    continue
                if abs(pa_r + float(rv["probability_B"]) - 1) > 0.02:
                    continue
                pa_f = min(max(pa_f, MIN_PROB), 1 - MIN_PROB)
                pa_r = min(max(pa_r, MIN_PROB), 1 - MIN_PROB)
                l_f = np.log((1 - pa_f + EPS) / (pa_f + EPS))
                l_r = np.log((pa_r + EPS) / (1 - pa_r + EPS))
        except (KeyError, TypeError, ValueError):
            continue
        rec["arb_total"] += 1
        g_ab = (l_f - l_r) / 2
        var, j, k = c["_key"]
        g_jk = -g_ab if c["_flip"] else g_ab
        rec["g"][f"{var}|{j},{k}"] = float(g_jk)
        rec["arb_valid"] += 1
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-personas", type=int, default=1000)
    ap.add_argument("--run-dir", default=RUN,
                    help="direct seed run containing sim.csv/real.csv")
    ap.add_argument("--fields", default=None, help="comma list; default all T1 fields")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--resume-dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arbiter-scale", choices=["prob", "logodds"], default="prob",
                    help="Arbiter output scale: probability pair (old, compressed) "
                         "or direct log-odds (new, de-saturated)")
    ap.add_argument("--grid", choices=["nice", "quantile"], default="nice",
                    help="numeric binning: fixed round-number grid (default) or sim quantiles (old)")
    ap.add_argument("--order-pairs", action="store_true",
                    help="add 3 event-order swap pairs per batch (T4 measurement), "
                         "reducing field pairs 16->13")
    ap.add_argument("--actor-sim", default=None,
                    help="sim.csv whose per-persona answers are the measured profile "
                         "(default: direct Actor run). Round-2 adaptive mode: point "
                         "this at round-1 run_order/sim.csv")
    ap.add_argument("--levels-from", default=None,
                    help="load levels.json from an existing ADA dir instead of "
                         "rebuilding from sim (REQUIRED with --actor-sim so edge "
                         "indices stay poolable across rounds)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build levels/persona specs and exit (no LLM calls)")
    args = ap.parse_args()
    run_dir = os.path.abspath(args.run_dir)

    settings = get_settings()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.resume_dir or args.outdir or os.path.join(ROOT, "runs", "ada", f"cfps_{stamp}")
    if not args.dry_run:
        os.makedirs(outdir, exist_ok=True)
    units_path = os.path.join(outdir, "units.jsonl")

    with open(os.path.join(ROOT, "configs", "eval", "cfps.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.actor_sim:
        sim_raw = pd.read_csv(args.actor_sim, low_memory=False)
        base = pd.read_csv(os.path.join(run_dir, "sim.csv"), low_memory=False)
        if "profile_id" not in sim_raw.columns:
            # reallocation/swap outputs preserve row order but drop the id column;
            # verify positional alignment on passthrough columns before re-attaching
            if len(sim_raw) != len(base):
                raise SystemExit(f"--actor-sim has {len(sim_raw)} rows, base run has {len(base)}")
            for c in ("gender", "birth_year", "minzu", "mother_education"):
                if (sim_raw[c].astype(str).values != base[c].astype(str).values).any():
                    raise SystemExit(f"--actor-sim rows not positionally aligned "
                                     f"with base run on passthrough column '{c}'")
            sim_raw.insert(0, "profile_id", base["profile_id"].values)
        sim = sim_raw.set_index("profile_id")
        print(f"actor-sim: {args.actor_sim} ({len(sim)} personas)")
    else:
        sim = pd.read_csv(os.path.join(run_dir, "sim.csv"), low_memory=False).set_index("profile_id")
    sample = pd.read_csv(os.path.join(ROOT, "data", "processed", "cfps", "sample.csv"))

    wanted = args.fields.split(",") if args.fields else list(cfg["t1"]["variables"])

    if args.actor_sim and not args.levels_from:
        raise SystemExit("--actor-sim requires --levels-from (edge indices must stay "
                         "identical to round 1 for pooling)")

    levels, schema_fields = {}, {}
    if args.levels_from:
        lv_prev = json.load(open(args.levels_from, encoding="utf-8"))
        for var, lm in lv_prev.items():
            if var == ORDER_FIELD:
                continue
            levels[var] = {"labels": lm["labels"], "kind": lm["kind"],
                           "meta": lm.get("meta", {})}
        print(f"levels-from: {args.levels_from} ({len(levels)} fields)")
    else:
        for var in wanted:
            vcfg = dict(cfg["t1"]["variables"][var])
            s = prep_variable(sim.reset_index(), var, vcfg).dropna()
            min_values = min(100, max(2, len(sim) // 2))
            if len(s) < min_values:
                print(f"[skip] {var}: too few sim values ({len(s)} < {min_values})")
                continue
            labels, kind, meta = build_levels(var, vcfg, s.to_numpy(), grid=args.grid)
            if labels is None or len(labels) < 2:
                print(f"[skip] {var}: degenerate levels")
                continue
            levels[var] = {"labels": [fmt_val(x) if isinstance(x, float) else x for x in labels],
                           "kind": kind, "meta": meta}
    for var in levels:
        vcfg = dict(cfg["t1"]["variables"][var])
        sch = {"field": var, "description": vcfg.get("description", ""),
               "type": vcfg.get("type")}
        if levels[var]["kind"] == "categorical":
            sch["allowed"] = levels[var]["labels"]
        else:
            sch["allowed"] = vcfg.get("allowed", {})
        schema_fields[var] = sch
    print(f"fields: {len(levels)}")

    schema = {"fields": schema_fields,
              "note": "每个字段独立取值；邻居只改动一个字段一个等级"}

    # ---- persona specs ----
    specs = []
    rng = np.random.default_rng(args.seed)
    pids = list(sample["profile_id"])
    rng.shuffle(pids)
    for pid in pids[:args.n_personas]:
        row = sample[sample.profile_id == pid]
        if row.empty or pid not in sim.index:
            continue
        row = row.iloc[0]
        actor_idxs = {}
        for var in levels:
            vcfg = dict(cfg["t1"]["variables"][var])
            val = prep_variable(sim.loc[[pid]], var, vcfg).iloc[0]
            if pd.isna(val):
                continue
            idx = level_index(val, levels[var]["labels"],
                              levels[var]["kind"], levels[var]["meta"])
            if idx is None and levels[var]["kind"] == "numeric_unique":
                # round-1 reallocation adds <=1e-3 jitter to break cross-field
                # ties; snap to the nearest label when unambiguous
                arr = np.asarray(levels[var]["labels"], float)
                near = int(np.argmin(np.abs(arr - float(val))))
                if abs(arr[near] - float(val)) < 0.05:
                    idx = near
            if idx is not None:
                actor_idxs[var] = idx
        if not actor_idxs:
            continue
        persona = {c: (None if pd.isna(row[c]) else str(row[c])) for c in INPUT_COLS}
        order_spec = None
        if args.order_pairs:
            ages = {}
            for e, fld in ORDER_FIELDS.items():
                vcfg = dict(cfg["t1"]["variables"][fld])
                val = prep_variable(sim.loc[[pid]], fld, vcfg).iloc[0]
                ages[e] = None if pd.isna(val) else float(val)
            st = order_state_presentable(ages)
            if st is not None:
                order_spec = {"ages": ages, "state_idx": st}
        specs.append({"pid": int(pid), "actor_idxs": actor_idxs,
                      "persona": persona, "order": order_spec})
    n_order = sum(1 for s in specs if s["order"])
    n_calls = len(specs) * CALLS_PER_PERSONA
    pairs_desc = (f"{MAX_CHALLENGES}D+{N_RANDOM}R+{N_CYCLE}C" if not args.order_pairs
                  else f"{MAX_CHALLENGES - 1}D+{N_RANDOM - 1}R+{N_CYCLE - 1}C+{N_ORDER}O")
    print(f"personas: {len(specs)} ({n_order} with order pairs) | ~{n_calls} calls "
          f"({pairs_desc} = {MAX_CHALLENGES + N_RANDOM + N_CYCLE} pairs/batch)")
    if args.order_pairs:
        states = np.bincount([s["order"]["state_idx"] for s in specs if s["order"]],
                             minlength=len(ORDER_STATES))
        print("actor order-state mix:",
              {st: int(c) for st, c in zip(ORDER_STATES, states) if c})

    # canonical levels payload (identical content across resumes for same config)
    lv = {k: {"labels": v["labels"], "kind": v["kind"], "meta": v["meta"]}
          for k, v in levels.items()}
    if args.order_pairs:
        lv[ORDER_FIELD] = {"labels": ORDER_STATES, "kind": "order",
                           "meta": {"events": ORDER_FIELDS}}
    lv_hash = hashlib.md5(
        json.dumps(lv, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()
    actor_sim_rel = os.path.relpath(args.actor_sim, ROOT) if args.actor_sim else None

    done = set()
    if os.path.exists(units_path):
        prot = os.path.join(outdir, "protocol.json")
        if os.path.exists(prot):
            prev = json.load(open(prot, encoding="utf-8"))
            if prev.get("arbiter_scale") != args.arbiter_scale or prev.get("grid") != args.grid \
                    or prev.get("order_pairs") != args.order_pairs:
                raise SystemExit(f"resume dir uses arbiter_scale={prev.get('arbiter_scale')}, "
                                 f"refusing to mix with --arbiter-scale {args.arbiter_scale}")
            if prev.get("levels_hash") not in (None, lv_hash):
                raise SystemExit("resume dir was collected on a DIFFERENT levels grid; "
                                 "edges would not be poolable — refusing")
            if prev.get("actor_sim") not in (None, actor_sim_rel):
                raise SystemExit(f"resume dir measured actor_sim={prev.get('actor_sim')} but "
                                 f"this invocation uses {actor_sim_rel} — refusing")
        with open(units_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # only count personas that produced measurements; network-failed
                # units (empty g) are retried automatically on resume
                if rec.get("g"):
                    done.add(rec["persona_id"])
    todo = [s for s in specs if s["pid"] not in done]
    print(f"resume: {len(done)} done, {len(todo)} to run")

    if args.dry_run:
        # per-field actor coverage sanity (catches level-mapping regressions)
        cov = {}
        for s in specs:
            for var in s["actor_idxs"]:
                cov[var] = cov.get(var, 0) + 1
        miss = {v: len(specs) - c for v, c in cov.items() if c < len(specs) * 0.95}
        print(f"dry-run: {len(specs)} specs, {len(cov)} fields covered"
              + (f", LOW COVERAGE: {miss}" if miss else ""))
        return

    with open(os.path.join(outdir, "levels.json"), "w", encoding="utf-8") as f:
        json.dump(lv, f, ensure_ascii=False, indent=1, default=str)

    with open(os.path.join(outdir, "protocol.json"), "w", encoding="utf-8") as f:
        json.dump({"arbiter_scale": args.arbiter_scale, "grid": args.grid,
                   "order_pairs": args.order_pairs, "levels_hash": lv_hash,
                   "actor_sim": actor_sim_rel}, f)

    client = LLMClient(base_url=settings.llm_base_url, api_key=settings.llm_api_key,
                       model=settings.llm_model, temperature=TEMPERATURE, top_p=1.0,
                       max_tokens=4096, json_mode=True)
    audit_f = open(os.path.join(outdir, "responses.jsonl"), "a", encoding="utf-8")
    out_f = open(units_path, "a", encoding="utf-8")
    lock = threading.Lock()

    def _run(sp):
        rng = np.random.default_rng(abs(hash((sp["pid"], args.seed))) % (2 ** 32))
        return run_persona(client, sp["pid"], sp["actor_idxs"], sp["persona"],
                           schema, levels, rng, scale=args.arbiter_scale,
                           order=sp["order"] if args.order_pairs else None)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=settings.llm_concurrency) as ex:
        futs = {ex.submit(_run, sp): sp for sp in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            sp = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001
                rec = {"persona_id": sp["pid"], "devil_valid": False,
                       "devil_error": f"unit_failed: {e}", "arb_valid": 0,
                       "arb_total": 0, "edges": [], "g": {}, "audit": []}
            with lock:
                out_f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                out_f.flush()
                for a in rec.pop("audit", []):
                    audit_f.write(json.dumps({"persona_id": rec["persona_id"], **a},
                                             ensure_ascii=False, default=str) + "\n")
                audit_f.flush()
            if i % 50 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}] elapsed {time.time() - t0:.0f}s", flush=True)
    out_f.close()
    audit_f.close()
    with open(os.path.join(outdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"model": settings.llm_model, "temperature": TEMPERATURE,
                    "source_run": run_dir,
                    "max_challenges": MAX_CHALLENGES, "n_random": N_RANDOM,
                    "n_cycle": N_CYCLE, "n_personas": len(specs),
                     "arbiter_scale": args.arbiter_scale, "grid": args.grid,
                     "order_pairs": args.order_pairs,
                     "actor_sim": actor_sim_rel,
                     "levels_from": (os.path.relpath(args.levels_from, ROOT)
                                     if args.levels_from else None),
                    "seed": args.seed, "elapsed_sec": round(time.time() - t0, 1)},
                   f, indent=2)
    print(f"[ada-collect] done -> {outdir}")


if __name__ == "__main__":
    main()
