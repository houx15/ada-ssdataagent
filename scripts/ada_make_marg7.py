"""Build marg7.jsonl = marg6 + child_number co-residence marginal.

Content rule (frozen): the child_number entry is the mean over the 6
reps of probe14 Part C (retr6.jsonl, kind=p, var=child_coreside),
keys 0..5 renormalised.  Everything else is copied verbatim from
marg6.jsonl.  No real data is read.
"""
from __future__ import annotations

import json

import numpy as np

OUT = "runs/ada/t1_probe/marg7.jsonl"

ds = []
for line in open("runs/ada/t1_probe/retr6.jsonl"):
    r = json.loads(line)
    if r["parse_ok"] and r["var"] == "child_coreside":
        ds.append(r["val"])
P = {k: float(np.mean([d.get(k, d.get(str(k), 0.0)) for d in ds]))
     for k in range(6)}
tot = sum(P.values())
P = {k: v / tot for k, v in P.items()}

rows = [json.loads(l) for l in open("runs/ada/t1_probe/marg6.jsonl")]
rows.append(dict(var="child_number", rep=1, kind="atom", parse_ok=True,
                 val={str(float(k)): p for k, p in P.items()}))
with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{OUT}: {[r['var'] for r in rows]}; child={P}")
