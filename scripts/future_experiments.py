#!/usr/bin/env python3
"""Plan, run, resume, and summarize the frozen FUTURE_EXPERIMENTS P0 matrix."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "experiments" / "future_p0.yaml"
DEFAULT_RUNS = ROOT / "runs" / "experiments"

CONDITION_COMPONENTS = {
    "direct": (),
    "marginal_only": ("marginal",),
    "ada_only": ("ada",),
    "population_without_ada": ("marginal", "dependence", "r2", "event_order"),
    "full_without_marginal": ("ada", "dependence", "r2", "event_order"),
    "full_without_dependence": ("ada", "marginal", "r2", "event_order"),
    "full_without_r2": ("ada", "marginal", "dependence", "event_order"),
    "full_without_event_order": ("ada", "marginal", "dependence", "r2"),
    "full": ("ada", "marginal", "dependence", "r2", "event_order"),
}

# Measured on the existing GLM-5.2 1,000-person artifacts, plus a conservative
# allowance for the consolidated T1/T2/T3/T4 population probes.
TOKEN_ESTIMATE_PER_FULL_SEED = {"prompt": 15_500_000, "completion": 23_350_000}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-").lower()


def load_config(path: str | os.PathLike[str]) -> dict:
    with open(path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    unknown = [c for c in cfg.get("conditions", []) if c not in CONDITION_COMPONENTS]
    if unknown:
        raise SystemExit(f"unknown conditions: {unknown}")
    return cfg


def selected_models(cfg: dict, names: str | None, provider: str | None) -> list[tuple[str, dict]]:
    wanted = {x.strip() for x in names.split(",")} if names else None
    rows = []
    for name, model in cfg["models"].items():
        if wanted and name not in wanted:
            continue
        if provider and model["provider"] != provider:
            continue
        rows.append((name, model))
    if wanted:
        missing = wanted - {name for name, _ in rows}
        if missing:
            raise SystemExit(f"unknown or provider-mismatched models: {sorted(missing)}")
    if not rows:
        raise SystemExit("no models selected")
    return rows


def model_cost(model: dict, n_seeds: int) -> float:
    t = TOKEN_ESTIMATE_PER_FULL_SEED
    per = (t["prompt"] * float(model["input_usd_per_m"])
           + t["completion"] * float(model["output_usd_per_m"])) / 1_000_000
    return per * n_seeds


def print_plan(cfg: dict, models: list[tuple[str, dict]], seeds: list[int],
               conditions: list[str], population_n: int | None = None,
               estimate_scale: float = 1.0) -> None:
    paid = 0.0
    print(f"protocol={cfg['name']} version={cfg['protocol_version']}")
    print(f"n={population_n or cfg['population_n']} seeds={seeds} conditions={conditions}")
    for name, model in models:
        cost = (model_cost(model, len(seeds)) * estimate_scale
                if any(c != "direct" for c in conditions) else 0.0)
        paid += cost
        print(f"{name:20s} {model['model']:35s} {model['category']:24s} est=${cost:.2f}")
    print(f"estimated paid total=${paid:.2f}; hard budget=${float(cfg['budget_usd']):.2f}")
    if paid > float(cfg["budget_usd"]):
        print("WARNING: estimate exceeds the configured hard budget")


class UnitRunner:
    def __init__(self, cfg: dict, suite: Path, model_name: str, model: dict,
                 seed: int, n: int, ada_personas: int, reps: int,
                 concurrency: int, eval_seeds: list[int], eval_b: int,
                 eval_sample_n: int, conditions: list[str]):
        self.cfg = cfg
        self.suite = suite
        self.model_name = model_name
        self.model = model
        self.seed = seed
        self.n = n
        self.ada_personas = min(ada_personas, n)
        self.reps = reps
        self.concurrency = concurrency
        self.eval_seeds = eval_seeds
        self.eval_b = eval_b
        self.eval_sample_n = eval_sample_n
        self.conditions = conditions
        self.unit = suite / slug(model_name) / f"seed_{seed}"
        self.logs = self.unit / "logs"
        self.state_path = self.unit / "state.json"
        self.ledger = suite / "cost_ledger.jsonl"
        self.state = self._load_state()
        self.env = self._environment()

    def _load_state(self) -> dict:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"model_name": self.model_name, "model": self.model["model"],
                    "seed": self.seed, "stages": {}, "created": utcnow()}

    def _save_state(self) -> None:
        self.unit.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2, ensure_ascii=False),
                                   encoding="utf-8")

    def _environment(self) -> dict[str, str]:
        load_dotenv(ROOT / ".env")
        env = dict(os.environ)
        provider = self.cfg["providers"][self.model["provider"]]
        if provider["kind"] == "openrouter":
            key = env.get(provider["api_key_env"])
            if not key:
                raise SystemExit(
                    f"{provider['api_key_env']} is not available; add it to the remote .env "
                    "or export it before starting the paid suite"
                )
            env["SDTL_LLM_BASE_URL"] = provider["base_url"]
            env["SDTL_LLM_API_KEY"] = key
            env["SSBENCH_LLM_EXTRA_BODY"] = json.dumps(provider.get("extra_body", {}))
        env["SDTL_LLM_MODEL"] = self.model["model"]
        env["SDTL_LLM_CONCURRENCY"] = str(self.concurrency)
        env["SSBB_MAX_WORKERS"] = str(self.concurrency)
        env["SSBENCH_COST_LEDGER"] = str(self.ledger)
        env["SSBENCH_BUDGET_USD"] = str(self.cfg["budget_usd"])
        env["SSBENCH_INPUT_USD_PER_M"] = str(self.model["input_usd_per_m"])
        env["SSBENCH_OUTPUT_USD_PER_M"] = str(self.model["output_usd_per_m"])
        return env

    def stage(self, name: str, command: list[str], outputs: list[Path]) -> None:
        old = self.state["stages"].get(name, {})
        if old.get("status") == "completed" and all(p.exists() for p in outputs):
            print(f"[skip] {self.model_name}/seed={self.seed} {name}", flush=True)
            return
        self.logs.mkdir(parents=True, exist_ok=True)
        log = self.logs / f"{name}.log"
        self.state["stages"][name] = {
            "status": "running", "started": utcnow(), "command": command,
            "outputs": [str(p) for p in outputs], "log": str(log),
        }
        self._save_state()
        print(f"[run] {self.model_name}/seed={self.seed} {name}", flush=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[{utcnow()}] {' '.join(command)}\n")
            handle.flush()
            proc = subprocess.run(command, cwd=ROOT, env=self.env,
                                  stdout=handle, stderr=subprocess.STDOUT)
        entry = self.state["stages"][name]
        entry["finished"] = utcnow()
        entry["exit_code"] = proc.returncode
        entry["status"] = "completed" if proc.returncode == 0 else "failed"
        self._save_state()
        if proc.returncode:
            raise SystemExit(f"stage {name} failed; inspect {log}")
        missing = [str(p) for p in outputs if not p.exists()]
        if missing:
            raise SystemExit(f"stage {name} did not create expected outputs: {missing}")

    def py(self, script: str, *args: object) -> list[str]:
        return [sys.executable, str(ROOT / "scripts" / script), *map(str, args)]

    def run(self) -> None:
        direct = self.unit / "direct"
        self.stage("direct", self.py(
            "simulate.py", "--dataset", self.cfg["dataset"], "--method", "direct",
            "--n", self.n, "--model", self.model["model"], "--seed", self.seed,
            "--tag", f"{self.model_name}-s{self.seed}", "--run-dir", direct,
        ), [direct / "meta.json", direct / "sim.csv", direct / "responses.jsonl"])

        need_population = any(c != "direct" for c in self.conditions)
        if need_population:
            self._build_targets_and_ada(direct)
        for condition in self.conditions:
            final_run = self._build_condition(condition, direct)
            self._validate_and_evaluate(condition, final_run)
        self.state["status"] = "completed"
        self.state["finished"] = utcnow()
        self._save_state()

    def _build_targets_and_ada(self, direct: Path) -> None:
        targets = self.unit / "targets"
        t1 = targets / "t1_marg.jsonl"
        t2 = targets / "t2_full.jsonl"
        t2_compiled = targets / "t2_targets.json"
        t3 = targets / "t3_r2.jsonl"
        t4 = targets / "t4_order.jsonl"
        t4_target = targets / "t4_target.json"
        round1 = self.unit / "ada_round1"
        round2 = self.unit / "ada_round2"
        pooled = self.unit / "ada_pooled"

        self.stage("ada_round1", self.py(
            "ada_cfps_collect.py", "--run-dir", direct, "--n-personas", self.ada_personas,
            "--outdir", round1, "--seed", self.seed, "--arbiter-scale", "logodds",
            "--grid", "nice", "--order-pairs",
        ), [round1 / "meta.json", round1 / "levels.json", round1 / "units.jsonl"])
        self.stage("ada_round1_load", self.py(
            "ada_make_sim.py", "--dir", round1, "--run-dir", direct,
            "--beta-modes", "one",
        ), [round1 / "run_one" / "sim.csv", round1 / "run_one" / "meta.json"])
        self.stage("ada_round2", self.py(
            "ada_cfps_collect.py", "--run-dir", direct, "--n-personas", self.ada_personas,
            "--outdir", round2, "--seed", self.seed + 100_000,
            "--arbiter-scale", "logodds", "--grid", "nice", "--order-pairs",
            "--actor-sim", round1 / "run_one" / "sim.csv",
            "--levels-from", round1 / "levels.json",
        ), [round2 / "meta.json", round2 / "units.jsonl"])
        self.stage("ada_pool", self.py(
            "ada_pool_rounds.py", "--dirs", round1, round2, "--out", pooled,
        ), [pooled / "meta.json", pooled / "units.jsonl"])
        self.stage("ada_pool_load", self.py(
            "ada_make_sim.py", "--dir", pooled, "--run-dir", direct,
            "--beta-modes", "one",
        ), [pooled / "run_one" / "sim.csv", pooled / "run_one" / "meta.json"])

        self.stage("probe_t1", self.py(
            "ada_t1_probe.py", "--out", t1, "--reps", self.reps,
            "--concurrency", self.concurrency,
        ), [t1])
        self.stage("probe_t2", self.py(
            "ada_t2_probe_full.py", "--out", t2, "--reps", self.reps,
            "--concurrency", self.concurrency,
        ), [t2])
        self.stage("compile_t2", self.py(
            "ada_t2_compile.py", "--probe", t2, "--out", t2_compiled,
        ), [t2_compiled])
        self.stage("probe_t3", self.py(
            "ada_t3_probe.py", "--out", t3, "--reps", self.reps,
            "--concurrency", self.concurrency,
        ), [t3])
        self.stage("probe_t4", self.py(
            "ada_t4_probe.py", "--out", t4, "--target-out", t4_target,
            "--reps", self.reps, "--concurrency", self.concurrency,
            "--zero-c-first",
        ), [t4, t4_target])

    def _build_condition(self, condition: str, direct: Path) -> Path:
        components = CONDITION_COMPONENTS[condition]
        current = direct
        condition_dir = self.unit / "conditions" / condition
        condition_dir.mkdir(parents=True, exist_ok=True)
        if "ada" in components:
            current = self.unit / "ada_pooled" / "run_one"
        if "marginal" in components:
            out = condition_dir / "01_marginal"
            self.stage(f"{condition}_marginal", self.py(
                "ada_t1_load.py", "--run-dir", current,
                "--marg", self.unit / "targets" / "t1_marg.jsonl", "--out", out,
            ), [out / "sim.csv", out / "meta.json"])
            current = out
        if "dependence" in components:
            out = condition_dir / "02_dependence"
            self.stage(f"{condition}_dependence", self.py(
                "ada_t2_load.py", "--run-dir", current,
                "--targets-json", self.unit / "targets" / "t2_targets.json", "--out", out,
            ), [out / "sim.csv", out / "meta.json"])
            current = out
        if "r2" in components:
            out = condition_dir / "03_r2"
            self.stage(f"{condition}_r2", self.py(
                "ada_t3_adjust.py", "--run-dir", current,
                "--r2", self.unit / "targets" / "t3_r2.jsonl",
                "--targets", self.unit / "targets" / "t2_targets.json",
                "--marg", self.unit / "targets" / "t1_marg.jsonl", "--out", out,
            ), [out / "sim.csv", out / "meta.json"])
            current = out
        if "event_order" in components:
            out = condition_dir / "04_event_order"
            self.stage(f"{condition}_event_order", self.py(
                "ada_order_post.py", "--run-dir", current,
                "--p-target", self.unit / "targets" / "t4_target.json", "--out", out,
            ), [out / "sim.csv", out / "meta.json"])
            current = out
        manifest = condition_dir / "condition.json"
        manifest.write_text(json.dumps({
            "condition": condition, "components": components,
            "final_run": str(current), "model": self.model["model"], "seed": self.seed,
        }, indent=2), encoding="utf-8")
        return current

    def _validate_and_evaluate(self, condition: str, run_dir: Path) -> None:
        base = self.unit / "conditions" / condition
        validation = base / "validation"
        self.stage(f"{condition}_validate", self.py(
            "validate_run.py", "--run-dir", run_dir, "--out-dir", validation,
        ), [validation / "consistency.json"])
        if condition != "direct":
            diagnostics = base / "target_diagnostics"
            self.stage(f"{condition}_targets", self.py(
                "diagnose_targets.py", "--run-dir", run_dir,
                "--t1", self.unit / "targets" / "t1_marg.jsonl",
                "--t2", self.unit / "targets" / "t2_targets.json",
                "--t3", self.unit / "targets" / "t3_r2.jsonl",
                "--t4", self.unit / "targets" / "t4_target.json",
                "--out-dir", diagnostics,
            ), [diagnostics / "target_achieved.csv",
                diagnostics / "target_achieved_summary.csv"])
        for eval_seed in self.eval_seeds:
            out = base / "evaluation" / f"seed_{eval_seed}"
            self.stage(f"{condition}_eval_{eval_seed}", self.py(
                "evaluate.py", "--run-dir", run_dir, "--B", self.eval_b,
                "--sample-n", self.eval_sample_n, "--seed", eval_seed,
                "--output-dir", out,
            ), [out / "overall_summary.csv"])


def summarize(suite: Path) -> None:
    eval_rows = []
    for manifest in suite.glob("*/seed_*/conditions/*/condition.json"):
        info = json.loads(manifest.read_text(encoding="utf-8"))
        model_name = manifest.parents[3].name
        seed = int(manifest.parents[2].name.split("_", 1)[1])
        for summary in manifest.parent.glob("evaluation/seed_*/overall_summary.csv"):
            eval_seed = int(summary.parent.name.split("_", 1)[1])
            df = pd.read_csv(summary)
            for row in df.to_dict("records"):
                eval_rows.append({"model_name": model_name, "model": info["model"],
                                  "seed": seed, "condition": info["condition"],
                                  "evaluator_seed": eval_seed, **row})
    out = suite / "summary"
    out.mkdir(parents=True, exist_ok=True)
    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(out / "evaluator_runs.csv", index=False)
    e2e = pd.DataFrame()
    if not eval_df.empty:
        per_population = eval_df.groupby(
            ["model_name", "model", "seed", "condition", "type"], as_index=False
        )["avg_insignificant_rate"].agg(["mean", "std"]).reset_index()
        per_population.to_csv(out / "per_population.csv", index=False)
        e2e = per_population.groupby(
            ["model_name", "model", "condition", "type"], as_index=False
        )["mean"].agg(["mean", "std", "min", "max"]).reset_index()
        e2e.to_csv(out / "end_to_end.csv", index=False)
    costs = []
    ledger = suite / "cost_ledger.jsonl"
    if ledger.exists():
        data = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        if data:
            cdf = pd.DataFrame(data)
            costs = cdf.groupby("requested_model", as_index=False).agg(
                calls=("cost_usd", "size"), prompt_tokens=("prompt_tokens", "sum"),
                completion_tokens=("completion_tokens", "sum"), cost_usd=("cost_usd", "sum"),
            ).to_dict("records")
            pd.DataFrame(costs).to_csv(out / "costs.csv", index=False)
    payload = {"suite": str(suite), "n_evaluator_rows": len(eval_df),
               "n_end_to_end_rows": len(e2e), "costs": costs, "created": utcnow()}
    (out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"summary -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["plan", "run", "summarize"])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--suite", default=None,
                    help="suite directory; required for deterministic resume/summarize")
    ap.add_argument("--provider", choices=["local", "openrouter"], default=None)
    ap.add_argument("--models", default=None, help="comma-separated model aliases")
    ap.add_argument("--seeds", default=None, help="comma-separated integer seeds")
    ap.add_argument("--conditions", default=None, help="comma-separated conditions")
    ap.add_argument("--smoke", action="store_true",
                    help="n=20, one probe/evaluator repeat, B=5")
    args = ap.parse_args()
    cfg = load_config(args.config)
    suite = Path(args.suite) if args.suite else DEFAULT_RUNS / cfg["name"]
    if args.command == "summarize":
        summarize(suite)
        return
    models = selected_models(cfg, args.models, args.provider)
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else list(cfg["seeds"])
    conditions = args.conditions.split(",") if args.conditions else list(cfg["conditions"])
    unknown = [c for c in conditions if c not in CONDITION_COMPONENTS]
    if unknown:
        raise SystemExit(f"unknown conditions: {unknown}")
    effective_n = 20 if args.smoke else int(cfg["population_n"])
    print_plan(cfg, models, seeds, conditions, population_n=effective_n,
               estimate_scale=(effective_n / int(cfg["population_n"])))
    if args.command == "plan":
        return

    n = effective_n
    el = cfg["elicitation"]
    ev = cfg["evaluator"]
    ada_personas = 20 if args.smoke else int(el["ada_personas"])
    reps = 1 if args.smoke else int(el["probe_reps"])
    eval_seeds = [int(ev["bootstrap_seeds"][0])] if args.smoke else list(ev["bootstrap_seeds"])
    eval_b = 5 if args.smoke else int(ev["B"])
    eval_sample_n = 20 if args.smoke else int(ev["sample_n"])
    suite.mkdir(parents=True, exist_ok=True)
    snapshot = suite / "suite_config.json"
    if not snapshot.exists():
        snapshot.write_text(json.dumps({"config": cfg, "smoke": args.smoke,
                                        "effective_population_n": n,
                                        "conditions": conditions, "created": utcnow()},
                                       indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    for model_name, model in models:
        for seed in seeds:
            UnitRunner(cfg, suite, model_name, model, seed, n, ada_personas, reps,
                       int(el["concurrency"]), eval_seeds, eval_b, eval_sample_n,
                       conditions).run()
    summarize(suite)


if __name__ == "__main__":
    main()
