#!/usr/bin/env python3
"""Run 5 independent evaluations of a run dir, report per-test means."""
import sys, json, os
import numpy as np
sys.path.insert(0, "src")
from ssbench.evaluation.runner import evaluate_run

run_dir = sys.argv[1]
types = sys.argv[2].split(",") if len(sys.argv) > 2 else None
names = types or ["t1","t2","t3","t4","t5"]
scores = {t: [] for t in names}
for k in range(5):
    res = evaluate_run(run_dir, types=types)
    for row in res["per_type"]:
        scores[row["type"]].append(row["avg_insignificant_rate"])
        print(f"  rep{k+1} {row['type']}: {row['avg_insignificant_rate']:.4f}", flush=True)
out = {t: float(np.mean(v)) for t, v in scores.items() if v}
out["overall"] = float(np.mean(list(out.values())))
print("MEANS:", json.dumps(out))
json.dump(out, open(os.path.join(run_dir, "eval5_means.json"), "w"))
