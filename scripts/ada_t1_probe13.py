"""T1 probe13 (v4): population-feedback sequential persona sampling.

Instrument (frozen before any evaluation of outputs):
  1000 virtual respondents generated in 100 blocks of 10 parallel
  calls.  EVERY respondent sees the exact answer histograms of all
  previous respondents (updated per block) and then answers a
  17-question CFPS-style questionnaire as a randomly drawn new
  45-55-year-old Chinese respondent (persona self-imagined).

  temperature 0.8 -- for a SAMPLING instrument the diversity is the
  instrument (declared deviation from the 0.3 elicitation standard).

  Fields (respondent wording -> schema mapping, declared 1:1):
    与人相处/理解/表达  3 items each, 1-7   -> mean = interpersonal/
        comprehension/expression (real: multi-item mean atoms)
    算术 24 题答对数     0-24              -> math_cognitive
    识字读词 34 题答对数 0-34              -> verbal_cognitive
    自评健康 很好/好/一般/不好/很不好       -> very healthy/fairly
        healthy/somewhat unhealthy/unhealthy/very unhealthy
    观念 2+2 items, 1-5                     -> fixed/growth mean
        (real: 9 atoms, 0.5 grid; item-2 wording is ours)
    30-40 岁主要职业(含 无业)               -> ISCO map + unemployed
    一共生过的孩子数 0-8                    -> child_number

Selection rule (frozen): per field, adopt persona marginal over the
incumbent instrument iff evaluator-scored rate improves by >= .01.

Output: runs/ada/t1_probe/retr5.jsonl  (append-only, resumable).
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ssbench.llm.client import LLMClient  # noqa: E402
from ssbench.settings import get_settings  # noqa: E402

OUT = "runs/ada/t1_probe/retr5.jsonl"
N = 1000
BLOCK = 10
FIELDS = ["a_soc", "a_und", "a_exp",      # ability items x3 (1-7)
          "math", "word", "health",
          "f1", "f2", "g1", "g2",
          "job", "kids"]

SYS = ("你将扮演一位随机抽中的中国 CFPS 住户调查受访者（45-55 岁），"
       "自行想象你的性别、城乡、学历、职业、家庭等背景，然后按你自己"
       "的真实情况作答。注意：你是具体的某个人，不是'平均人'；受访者"
       "之间的差异很大。只输出符合指定格式的 JSON，不要输出理由。")

Q = """CFPS 问卷（已访 {n} 位受访者，他们的回答分布附后供你了解样本人群）：

1-3. 请给你自己在以下三方面的能力打分（1 很差 … 7 很好，各一项整数）：
    a_soc 与人相处/人际交往；a_und 理解报纸电视广播的内容；a_exp 把自己的想法清楚地说出来
4. math：算术测试 24 题，你答对几题（0-24 整数）
5. word：识字读词测试 34 题，你读对几题（0-34 整数）
6. health：你的总体健康状况（很好/好/一般/不好/很不好 之一）
7-8. f1/f2：对下列陈述的同意程度（1 很不同意 … 5 很同意）：
    f1「人的聪明才智基本是天生的，后天很难改变」
    f2「有些人怎么学都学不会，主要是天分不够」
9-10. g1/g2：同意程度（1-5）：
    g1「只要努力，能力是可以提高的」
    g2「多花时间练习，就能把不擅长的事做好」
11. job：你 30-40 岁期间的主要职业（选一）：
    负责人 / 专业技术人员 / 技术员或助理人员 / 办事人员 / 商业服务业人员 /
    农林牧渔劳动者 / 技工工匠 / 机械操作员 / 普通非技术工人 / 军人 / 无业（含家务、退休）
12. kids：你一生共生过几个孩子（活产数，整数 0-8）

在你之前的 {n} 位受访者的回答分布（直方图，0 计数省略）：
{hist}

现在请你作答。输出 JSON：
{{"a_soc":1,"a_und":1,"a_exp":1,"math":0,"word":0,"health":"好",
  "f1":1,"f2":1,"g1":1,"g2":1,"job":"…","kids":0}}"""


def compose_hist(done):
    """exact atom counts per composite field, nonzero only."""
    if not done:
        return "（你是最早的受访者之一）"
    lines = []
    # abilities: ONE item per field (single items keep full inter-person
    # spread; 3-item averaging would shrink variance toward the middle --
    # the exact failure mode all previous instruments showed)
    for name, key in [("与人相处", "a_soc"), ("理解", "a_und"),
                      ("表达", "a_exp")]:
        vc = {}
        for d in done:
            vc[d[key]] = vc.get(d[key], 0) + 1
        lines.append(f"{name}: " + "|".join(
            f"{k}:{v}" for k, v in sorted(vc.items(), key=lambda kv: float(kv[0]))))
    for name, key in [("math", "math"), ("word", "word"), ("kids", "kids")]:
        vc = {}
        for d in done:
            vc[d[key]] = vc.get(d[key], 0) + 1
        lines.append(f"{name}: " + "|".join(
            f"{k}:{v}" for k, v in sorted(vc.items(), key=lambda kv: float(kv[0]))))
    for name in ["health", "job"]:
        vc = {}
        for d in done:
            vc[d[name]] = vc.get(d[name], 0) + 1
        lines.append(f"{name}: " + "|".join(
            f"{k}:{v}" for k, v in sorted(vc.items(), key=lambda kv: -kv[1])))
    for name, k1, k2 in [("观念-天生(f1f2均分)", "f1", "f2"),
                         ("观念-努力(g1g2均分)", "g1", "g2")]:
        vc = {}
        for d in done:
            m = round((float(d[k1]) + float(d[k2])) / 2 * 2) / 2
            vc[m] = vc.get(m, 0) + 1
        lines.append(f"{name}: " + "|".join(
            f"{k}:{v}" for k, v in sorted(vc.items())))
    return "\n".join(lines)


INT_KEYS = ["a_soc", "a_und", "a_exp", "math", "word", "f1", "f2",
            "g1", "g2", "kids"]
BOUNDS = {"a_soc": (1, 7), "a_und": (1, 7), "a_exp": (1, 7),
          "math": (0, 24), "word": (0, 34), "f1": (1, 5), "f2": (1, 5),
          "g1": (1, 5), "g2": (1, 5), "kids": (0, 8)}
HEALTH = {"很好", "好", "一般", "不好", "很不好"}
JOBS = {"负责人", "专业技术人员", "技术员或助理人员", "办事人员",
        "商业服务业人员", "农林牧渔劳动者", "技工工匠", "机械操作员",
        "普通非技术工人", "军人", "无业"}


def parse(c):
    try:
        m = re.search(r"\{.*\}", c, re.S)
        d = json.loads(m.group(0))
        out = {}
        for k in INT_KEYS:
            v = int(round(float(d[k])))
            lo, hi = BOUNDS[k]
            out[k] = max(lo, min(hi, v))
        h = str(d["health"]).strip()
        if h not in HEALTH:
            raise ValueError("health")
        out["health"] = h
        j = str(d["job"]).strip()
        if j not in JOBS:
            raise ValueError("job")
        out["job"] = j
        return out, True
    except Exception:
        return None, False


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = {}
    try:
        for line in open(OUT):
            r = json.loads(line)
            if r.get("parse_ok"):
                done[r["i"]] = r["val"]
    except FileNotFoundError:
        pass
    todo = [i for i in range(N) if i not in done]
    print(f"answered: {len(done)}  remaining: {len(todo)}", flush=True)
    if not todo:
        return
    st = get_settings()
    cli = LLMClient(base_url=st.llm_base_url, api_key=st.llm_api_key,
                    model=st.llm_model, temperature=0.8, max_tokens=4096,
                    json_mode=True)
    lock = threading.Lock()
    fout = open(OUT, "a")
    # blocks aligned to indices for per-block feedback
    for b0 in range(0, N, BLOCK):
        idxs = [i for i in range(b0, b0 + BLOCK) if i in set(todo)]
        if not idxs:
            continue
        vals = [done[i] for i in sorted(done)]
        hist = compose_hist(vals)
        user = Q.format(n=len(vals), hist=hist)

        def run(i):
            r = cli.chat(SYS, user)
            c = r.content or ""
            v, ok = parse(c)
            with lock:
                fout.write(json.dumps(dict(i=i, parse_ok=ok, val=v,
                                           content=c),
                                      ensure_ascii=False) + "\n")
                fout.flush()
            if ok:
                done[i] = v
            return ok

        with ThreadPoolExecutor(len(idxs)) as ex:
            for f in as_completed([ex.submit(run, i) for i in idxs]):
                f.result()
        nb = sum(1 for i in range(b0, b0 + BLOCK) if i in done)
        print(f"block {b0//BLOCK+1}/100 done ({nb}/{BLOCK} ok, "
              f"total {len(done)})", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
