"""Build marg6.jsonl: pooled retrieval-battery marginals for the three
fields whose T1 credit became positive (§7.7).

Content rule (frozen): marg6 = mean over reps of the LLM outputs only.
  occupation_30_40 : probe12 (retr4.jsonl) 9-class civilian-employed
                     framing, averaged over reps, renormalised
  fixed/growth_mindset : probe11 (retr3.jsonl phase B) 9-key atom grid
                     (1..5 step .5), keys converted float("3.0")-style,
                     averaged over reps, renormalised
No real data is read anywhere in this script.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

OUT = "runs/ada/t1_probe/marg6.jsonl"

rows = []

# occupation (probe12 / retr4)
g = []
for line in open("runs/ada/t1_probe/retr4.jsonl"):
    r = json.loads(line)
    if r["var"] == "occupation_30_40" and r["parse_ok"]:
        g.append(r["val"])
keys = list(g[0].keys())
M = {k: float(np.mean([d.get(k, 0.0) for d in g])) for k in keys}
tot = sum(M.values())
rows.append(dict(var="occupation_30_40", rep=1, kind="cat",
                 parse_ok=True, val={k: v / tot for k, v in M.items()}))

# mindsets (probe11 / retr3 phase B): float() key conversion is the
# whole point -- JSON dict keys are strings ("3.0"), the loader sorts
# them as floats
gm = defaultdict(list)
for line in open("runs/ada/t1_probe/retr3.jsonl"):
    r = json.loads(line)
    if (r["phase"] == "B" and r["parse_ok"]
            and r["var"] in ("fixed_mindset", "growth_mindset")):
        gm[r["var"]].append({float(k): float(v)
                             for k, v in r["val"].items()})
for v in ["fixed_mindset", "growth_mindset"]:
    ds = gm[v]
    keys = sorted({k for d in ds for k in d})
    P = {k: float(np.mean([d.get(k, 0.0) for d in ds])) for k in keys}
    tot = sum(P.values())
    rows.append(dict(var=v, rep=1, kind="atom", parse_ok=True,
                     val={str(float(k)): p / tot for k, p in P.items()}))

with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{OUT}: {len(rows)} entries "
      f"({', '.join(r['var'] for r in rows)})")
