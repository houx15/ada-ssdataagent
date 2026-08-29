"""Run orchestration: sample inputs -> generate -> postprocess -> persist artifacts."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

from ssbench.datasets.registry import load_dataset
from ssbench.llm.client import LLMClient
from ssbench.preprocessing.prepare import processed_sample_path
from ssbench.settings import get_settings
from ssbench.simulation.checkpoint import CheckpointStore, ResponseAuditLog
from ssbench.simulation.lifecycle import apply_postprocess
from ssbench.simulation.methods import create_method

DEFAULT_MAX_TOKENS = {"gss": 16384, "cfps": 32768}


@dataclass
class RunResult:
    run_dir: str
    sim_csv: str
    real_csv: str
    n_rows: int
    n_complete: int
    elapsed_sec: float


def run_simulation(
    dataset: str,
    method: str = "direct",
    n: int = 1000,
    model: Optional[str] = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: Optional[int] = None,
    max_attempts: int = 4,
    seed: int = 42,
    runs_root: Optional[str] = None,
    tag: Optional[str] = None,
    resume_dir: Optional[str] = None,
    run_dir: Optional[str] = None,
) -> RunResult:
    """Run one simulation. Pass ``resume_dir`` (an existing run directory) to continue it."""
    settings = get_settings()
    spec = load_dataset(dataset)

    sample_path = processed_sample_path(dataset)
    if not os.path.exists(sample_path):
        raise FileNotFoundError(
            f"{sample_path} not found — run `python scripts/prepare_data.py --dataset {dataset}` first"
        )
    processed = pd.read_csv(sample_path, low_memory=False)
    n_eff = min(n, len(processed))
    inputs_df = processed.sample(n=n_eff, replace=False, random_state=seed).reset_index(drop=True)
    inputs_df["profile_id"] = range(len(inputs_df))

    model = model or settings.llm_model
    max_tokens = max_tokens or DEFAULT_MAX_TOKENS.get(dataset, 16384)
    client = LLMClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    sampling_parameters_sent = os.getenv(
        "SSBENCH_LLM_OMIT_SAMPLING_PARAMS", ""
    ).strip().lower() not in {"1", "true", "yes", "on"}
    gen = create_method(method, client=client, max_attempts=max_attempts)

    if resume_dir and run_dir:
        raise ValueError("resume_dir and run_dir are mutually exclusive")
    if resume_dir:
        run_dir = os.path.abspath(resume_dir)
        if not os.path.exists(os.path.join(run_dir, "meta.json")):
            raise FileNotFoundError(f"{resume_dir} is not a run directory (no meta.json)")
        with open(os.path.join(run_dir, "meta.json"), "r", encoding="utf-8") as f:
            old_meta = json.load(f)
        print(f"[run] resuming {run_dir} (previous model={old_meta.get('model')}, "
              f"n={old_meta.get('n')}, seed={old_meta.get('seed')})")
        if old_meta.get("seed") != seed:
            print(f"[run] WARNING: seed differs from original run "
                  f"({seed} vs {old_meta.get('seed')}); input rows may not match.")
    elif run_dir:
        run_dir = os.path.abspath(run_dir)
        if os.path.exists(os.path.join(run_dir, "meta.json")):
            raise FileExistsError(
                f"{run_dir} is already a completed run; use resume_dir to continue it"
            )
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + (f"_{tag}" if tag else "")
        run_dir = os.path.join(runs_root or settings.runs_dir, dataset, method, run_id)
    os.makedirs(run_dir, exist_ok=True)

    checkpoint = CheckpointStore(run_dir)
    audit = ResponseAuditLog(run_dir)
    failure_logger = audit.log

    start = time.time()
    sim_df = gen.generate(spec, inputs_df, failure_logger=failure_logger, checkpoint=checkpoint)
    elapsed = time.time() - start

    if spec.postprocess_modules:
        sim_df = apply_postprocess(sim_df, spec.postprocess_modules)

    check_cols = [c for c in spec.static_outputs + spec.postprocess_modules if c in sim_df.columns]
    n_complete = int((~sim_df[check_cols].isna().any(axis=1)).sum()) if len(sim_df) else 0

    real_csv = os.path.join(run_dir, "real.csv")
    sim_csv = os.path.join(run_dir, "sim.csv")
    inputs_df.to_csv(real_csv, index=False)
    sim_df.to_csv(sim_csv, index=False)

    meta = {
        "dataset": dataset,
        "method": method,
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "sampling_parameters_sent": sampling_parameters_sent,
        "max_tokens": max_tokens,
        "n": len(inputs_df),
        "seed": seed,
        "max_attempts": max_attempts,
        "elapsed_sec": round(elapsed, 1),
        "n_complete": n_complete,
        "n_checkpointed": len(checkpoint),
        "resumed_from": resume_dir,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[run] dataset={dataset} method={method} model={model}")
    print(f"[run] rows={len(sim_df)} complete={n_complete} elapsed={elapsed:.1f}s")
    print(f"[run] artifacts -> {run_dir}")
    return RunResult(run_dir, sim_csv, real_csv, len(sim_df), n_complete, elapsed)
