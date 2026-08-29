"""Evaluate one simulation run directory against SSDataBench T1–T5."""

from __future__ import annotations

import json
import os

import pandas as pd
import yaml

from ssbench.evaluation.bootstrap import BootstrapConfig
from ssbench.settings import get_settings

_RUNNERS = {}


def _register(fn):
    _RUNNERS[fn.__name__.replace("run_", "")] = fn
    return fn


def _load_eval_config(dataset: str) -> dict:
    path = os.path.join(get_settings().configs_dir, "eval", f"{dataset}.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@_register
def run_t1(df_real, df_sim, cfg, boot):
    from ssbench.evaluation.t1_univariate import run_t1 as f
    return f(df_real, df_sim, cfg["variables"], boot)


@_register
def run_t2(df_real, df_sim, cfg, boot):
    from ssbench.evaluation.t2_bivariate import run_t2 as f
    return f(df_real, df_sim, cfg["variables"], boot)


@_register
def run_t3(df_real, df_sim, cfg, boot):
    from ssbench.evaluation.t3_regression import run_t3 as f
    return f(df_real, df_sim, cfg["responses"], cfg["predictors"], cfg["model_type"], boot)


@_register
def run_t4(df_real, df_sim, cfg, boot):
    from ssbench.evaluation.t4_event_order import run_t4 as f
    return f(df_real, df_sim, cfg["events"], boot)


@_register
def run_t5(df_real, df_sim, cfg, boot):
    from ssbench.evaluation.t5_event_covariate import run_t5 as f
    return f(df_real, df_sim, cfg["events"], cfg["predictors"], boot)


def evaluate_run(
    run_dir: str,
    types: list[str] | None = None,
    B: int | None = None,
    sample_n: int | None = None,
    seed: int | None = None,
    output_dir: str | None = None,
) -> dict:
    run_dir = os.path.abspath(run_dir)
    with open(os.path.join(run_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    dataset = meta["dataset"]

    eval_cfg = _load_eval_config(dataset)
    boot = BootstrapConfig.from_dict(eval_cfg["bootstrap"])
    if B is not None:
        boot.B = B
    if sample_n is not None:
        boot.sample_n = sample_n
    if seed is not None:
        boot.seed = seed

    types = types or eval_cfg["types"]
    unknown = [t for t in types if t not in eval_cfg or t not in _RUNNERS]
    if unknown:
        raise ValueError(f"Types {unknown} not configured/implemented for dataset '{dataset}'")

    df_real = pd.read_csv(os.path.join(run_dir, "real.csv"), low_memory=False)
    df_sim = pd.read_csv(os.path.join(run_dir, "sim.csv"), low_memory=False)

    out_dir = os.path.abspath(output_dir) if output_dir else os.path.join(run_dir, "evaluation")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for t in types:
        print(f"\n================ {t.upper()} ================\nBootstrap {boot.B} | "
              f"alpha={boot.alpha} | sample_n={boot.sample_n}")
        res = _RUNNERS[t](df_real.copy(), df_sim.copy(), eval_cfg[t], boot)
        res["summary_df"].to_csv(os.path.join(out_dir, f"summary_{t}.csv"), index=False)
        for name, extra_df in (res.get("extra") or {}).items():
            extra_dir = os.path.join(out_dir, f"data_{t}")
            os.makedirs(extra_dir, exist_ok=True)
            extra_df.to_csv(os.path.join(extra_dir, f"{name}.csv"), index=False)
        rows.append({"type": t, "avg_insignificant_rate": res["avg_insignificant_rate"]})

    summary = pd.DataFrame(rows)
    overall = float(summary["avg_insignificant_rate"].mean(skipna=True)) if len(summary) else float("nan")
    summary = pd.concat([
        summary,
        pd.DataFrame([{"type": "overall", "avg_insignificant_rate": overall}]),
    ], ignore_index=True)
    summary.to_csv(os.path.join(out_dir, "overall_summary.csv"), index=False)

    print("\n================ EVAL SUMMARY ================")
    print(summary.to_string(index=False))
    print(f"[eval] written -> {out_dir}")
    return {"overall": overall, "per_type": rows, "output_dir": out_dir,
            "bootstrap_seed": boot.seed}
