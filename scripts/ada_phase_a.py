#!/usr/bin/env python3
"""ADA-Observer Phase A: single-field prototype on ordinal fields (v2 §17).

Actor = existing direct-run outputs. Per (persona, field): Devil selects <=2
adjacent-category edges, +1 random adjacent +1 cycle (skip-one) edge; Blind
Arbiter scores anonymized A/B forward + reversed. Signal-level analysis:
legality, position bias, echo test, Hodge potentials, GSS->CFPS innovation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ssbench.ada.calls import UnitResult, run_unit  # noqa: E402
from ssbench.ada.signals import aggregate_edges, clr_with_pseudo, hodge, softmax  # noqa: E402
from ssbench.evaluation.cleaning import prep_variable  # noqa: E402
from ssbench.llm.client import LLMClient  # noqa: E402
from ssbench.settings import get_settings  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS = {
    "gss": os.path.join(ROOT, "runs", "gss", "direct", "20260817_163059_glm52"),
    "cfps": os.path.join(ROOT, "runs", "cfps", "direct", "20260817_164329_glm52"),
}
HIST = {
    "gss": ("United States, General Social Survey style cross-section. "
            "A randomly sampled adult respondent."),
    "cfps": ("China, CFPS-style life-course panel. A randomly sampled adult "
             "respondent followed from age 14 to 45."),
}
PER_UNIT_FIELD = {"gss": 12, "cfps": 30}
TEMPERATURE = 0.3
MAX_CHALLENGES = 2


ORDINAL_FIELDS = {
    "gss": ["education", "income", "wealth", "health", "isolated", "lonely",
            "happy", "political_view", "satisfy_job", "work_hard"],
    "cfps": ["highest_education", "self_rated_health"],
}


def ordinal_fields(ds: str, cfg: dict) -> dict:
    out = {}
    for var in ORDINAL_FIELDS[ds]:
        vcfg = cfg["t1"]["variables"][var]
        assert (vcfg.get("type") or "") == "categorical" and len(vcfg["allowed"]) >= 3
        out[var] = list(vcfg["allowed"])
    return out


def persona_json_for(ds: str, row) -> dict:
    inputs = {
        "gss": ["age", "gender", "race", "immigrant_status", "mother_education",
                "father_education", "mother_occupation", "father_occupation"],
        "cfps": ["gender", "minzu", "mother_education", "father_education"],
    }[ds]
    out = {}
    for c in inputs:
        v = row.get(c)
        out[c] = None if pd.isna(v) else (int(v) if isinstance(v, (np.integer,)) and c == "age" else str(v))
    return out


def clean_series(series: pd.Series, vcfg: dict) -> pd.Series:
    s = prep_variable(pd.DataFrame({vcfg.get("_name", "v"): series}), "v", vcfg)
    s.index = series.index
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 personas per field")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    settings = get_settings()
    client = LLMClient(
        base_url=settings.llm_base_url, api_key=settings.llm_api_key,
        model=settings.llm_model, temperature=TEMPERATURE, top_p=1.0,
        max_tokens=2048, json_mode=True,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir or os.path.join(ROOT, "runs", "ada", f"{stamp}_glm52")
    os.makedirs(outdir, exist_ok=True)

    cfgs, reals, sims, samples = {}, {}, {}, {}
    for ds in RUNS:
        with open(os.path.join(ROOT, "configs", "eval", f"{ds}.yaml"), encoding="utf-8") as f:
            cfgs[ds] = yaml.safe_load(f)
        reals[ds] = pd.read_csv(os.path.join(RUNS[ds], "real.csv"), low_memory=False)
        sims[ds] = pd.read_csv(os.path.join(RUNS[ds], "sim.csv"), low_memory=False)
        samples[ds] = pd.read_csv(os.path.join(ROOT, "data", "processed", ds, "sample.csv"))

    fields = {ds: ordinal_fields(ds, cfgs[ds]) for ds in RUNS}
    print("ordinal fields:", {ds: list(f) for ds, f in fields.items()})

    # ---- build units ----
    units = []
    for ds in RUNS:
        per_field = 2 if args.smoke else PER_UNIT_FIELD[ds]
        sim = sims[ds].set_index("profile_id")
        for fld, cats in fields[ds].items():
            vcfg = dict(cfgs[ds]["t1"]["variables"][fld])
            taken = 0
            for _, row in samples[ds].iterrows():
                if taken >= per_field:
                    break
                pid = int(row["profile_id"])
                if pid not in sim.index:
                    continue
                ans = sim.loc[pid, fld]
                ans = prep_variable(pd.DataFrame({"v": [ans]}), "v", vcfg).iloc[0]
                if pd.isna(ans) or ans not in cats:
                    continue
                units.append({
                    "ds": ds, "field": fld, "persona_id": pid,
                    "actor_idx": cats.index(ans), "cats": cats,
                    "persona": persona_json_for(ds, row),
                    "schema": {"field": fld, "description": vcfg.get("description", ""),
                               "allowed": cats},
                })
                taken += 1
    print(f"units: {len(units)} -> ~{3 * len(units)} calls")

    def _run(u):
        rng = np.random.default_rng(abs(hash((u["ds"], u["field"], u["persona_id"]))) % (2**32))
        return run_unit(
            client, u["ds"], u["field"], u["persona_id"], u["actor_idx"], u["cats"],
            HIST[u["ds"]], u["persona"], u["schema"], MAX_CHALLENGES, rng,
        )

    results: list[UnitResult] = []
    t0 = time.time()
    audit_f = open(os.path.join(outdir, "responses.jsonl"), "a", encoding="utf-8")
    lock = __import__("threading").Lock()
    with ThreadPoolExecutor(max_workers=settings.llm_concurrency) as ex:
        futs = {ex.submit(_run, u): u for u in units}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001
                u = futs[fut]
                r = UnitResult(ds=u["ds"], field=u["field"], persona_id=u["persona_id"],
                               actor_idx=u["actor_idx"], devil_error=str(e))
            results.append(r)
            with lock:
                for a in r.audit:
                    audit_f.write(json.dumps({
                        "ds": r.ds, "field": r.field, "persona_id": r.persona_id, **a,
                    }, ensure_ascii=False) + "\n")
                audit_f.flush()
            if i % 20 == 0 or i == len(units):
                print(f"[{i}/{len(units)}] elapsed {time.time() - t0:.0f}s", flush=True)
    audit_f.close()

    pd.DataFrame([{
        "ds": r.ds, "field": r.field, "persona_id": r.persona_id, "actor_idx": r.actor_idx,
        "devil_valid": r.devil_valid, "devil_error": r.devil_error,
        "n_g": len(r.g), "arb_valid": r.arbiter_valid, "arb_total": r.arbiter_total,
    } for r in results]).to_csv(os.path.join(outdir, "units.csv"), index=False)

    # ---- per-field signals ----
    rows = []
    phi_store, xq_store, xp_store, q_store, p_store, cat_store = {}, {}, {}, {}, {}, {}
    for ds in RUNS:
        for fld, cats in fields[ds].items():
            rs = [r for r in results if r.ds == ds and r.field == fld and r.g]
            if not rs:
                continue
            gbar, counts, src_g = aggregate_edges(rs, cats)
            phi, resid = hodge(gbar, cats)
            vcfg = dict(cfgs[ds]["t1"]["variables"][fld])
            rc = prep_variable(reals[ds], fld, vcfg).dropna()
            sc = prep_variable(sims[ds], fld, vcfg).dropna()
            cats_eff = [c for c in cats if (rc == c).any() or (sc == c).any()]
            idx = [cats.index(c) for c in cats_eff]
            pc = np.array([(rc == c).sum() for c in cats_eff], float)
            qc = np.array([(sc == c).sum() for c in cats_eff], float)
            x_p = clr_with_pseudo(pc)
            x_q = clr_with_pseudo(qc)
            r_ada = phi[idx] - x_q
            cycle_resid = [abs(v) for e, v in resid.items() if e[1] - e[0] == 2]
            pos_bias = [abs(v) for r in rs for v in r.o.values()]
            rows.append({
                "ds": ds, "field": fld, "m": len(cats), "n_units": len(rs),
                "n_edges": len(gbar),
                "devil_valid_rate": float(np.mean([r.devil_valid for r in
                                                   [x for x in results if x.ds == ds and x.field == fld]])),
                "arb_valid_rate": (sum(r.arbiter_valid for r in rs) /
                                   max(sum(r.arbiter_total for r in rs), 1)),
                "pos_bias_mean": float(np.mean(pos_bias)) if pos_bias else np.nan,
                "cycle_inconsistency": float(np.mean(cycle_resid)) if cycle_resid else np.nan,
                "echo_corr": float(np.corrcoef(phi[idx], x_q)[0, 1]),
                "cos_rADA_vs_PQ": float(
                    (r_ada @ (x_p - x_q)) / (np.linalg.norm(r_ada) * np.linalg.norm(x_p - x_q) + 1e-12)),
                "rADA_norm": float(np.linalg.norm(r_ada)),
                "PQ_norm": float(np.linalg.norm(x_p - x_q)),
            })
            phi_store[(ds, fld)] = phi[idx]
            xq_store[(ds, fld)] = x_q
            xp_store[(ds, fld)] = x_p
            q_store[(ds, fld)] = qc / qc.sum()
            p_store[(ds, fld)] = pc / pc.sum()
            cat_store[(ds, fld)] = cats_eff

    sig = pd.DataFrame(rows)
    sig.to_csv(os.path.join(outdir, "field_signals.csv"), index=False)
    print("\n===== field signals =====")
    print(sig.round(3).to_string(index=False))

    # ---- GSS -> CFPS innovation transfer (beta fitted on GSS only) ----
    tr = [k for k in phi_store if k[0] == "gss"]
    te = [k for k in phi_store if k[0] == "cfps"]
    if tr and te:
        num = sum(float((phi_store[k] - xq_store[k]) @ (xp_store[k] - xq_store[k])) for k in tr)
        den = sum(float(np.linalg.norm(phi_store[k] - xq_store[k]) ** 2) for k in tr)
        beta = num / den if den > 0 else 0.0
        trows = []
        for k in te:
            p, q = p_store[k], q_store[k]
            r_hat = softmax(xq_store[k] + beta * (phi_store[k] - xq_store[k]))
            trows.append({
                "field": k[1],
                "tv_raw": 0.5 * np.abs(p - q).sum(),
                "tv_ada": 0.5 * np.abs(p - r_hat).sum(),
                "beta_gss": beta,
            })
        tr_df = pd.DataFrame(trows)
        tr_df.to_csv(os.path.join(outdir, "innovation_transfer.csv"), index=False)
        print("\n===== GSS->CFPS innovation (beta from GSS only) =====")
        print(tr_df.round(3).to_string(index=False))

    meta = {
        "model": settings.llm_model, "temperature_devil_arbiter": TEMPERATURE,
        "min_prob": 0.05, "max_challenges": MAX_CHALLENGES,
        "n_units": len(units), "elapsed_sec": round(time.time() - t0, 1),
        "per_unit_field": PER_UNIT_FIELD, "smoke": args.smoke,
    }
    with open(os.path.join(outdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[ada] outputs -> {outdir}")


if __name__ == "__main__":
    main()
