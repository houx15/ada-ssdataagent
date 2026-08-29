"""T1 probe17 (v8): deliberation-with-critique revision loops.

User-directed follow-up to probe13 (§7.8).  Two mechanisms, two arms,
shared round 0:

  Round 0 (shared): N=600 personas answer the 12-item questionnaire
    with NO feedback (unbiased sampling, temperature 0.8).

  Arm A (neutral deliberation): 2 revision rounds.  Each persona sees
    THEIR OWN answers + the current arm-A population histograms and a
    neutral question: revise anything or keep as is.

  Arm B (critique deliberation): before each revision round, a
    DEMOGRAPHER role critiques the current population histograms
    ("do these distributions look like the real 45-55 Chinese
    population? which levels are over/under-represented?") at
    temperature 0.3 -- the evaluation role that §7.6 showed reaches
    corpus knowledge where estimation does not.  Each persona then
    sees own answers + histograms + the pooled critique, and decides
    whether to revise.

Mechanism predictions (frozen before running):
  A: self-consistency vs conformity -- revision rate moderate, drift
     toward the middle (probe13's failure mode) or no change.
  B: critique may reweight the population if the evaluator role
     accesses different knowledge than the respondent role.

Selection rule (frozen): per field, candidate = (arm, round) marginal;
adopt iff evaluator-scored insignificant rate >= incumbent + .01.
Incumbents: abilities/comprehension/expression/math/health .000,
verbal .005, occupation .013, fixed .095, growth .105, child .853.

Output: runs/ada/t1_probe/retr9.jsonl  (append-only, resumable).
Records: {arm:'A'|'B'|'S', i, round, val, parse_ok}; critique records
{arm:'B', i:-1, round, val:{critique}}.
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

OUT = "runs/ada/t1_probe/retr9.jsonl"
N = 600
ROUNDS = [1, 2]

SYS_P = ("你将扮演一位随机抽中的中国 CFPS 住户调查受访者（45-55 岁），"
         "自行想象你的性别、城乡、学历、职业、家庭等背景，然后按你自己"
         "的真实情况作答。注意：你是具体的某个人，不是'平均人'；受访者"
         "之间的差异很大。只输出符合指定格式的 JSON，不要输出理由。")

Q0 = """CFPS 问卷（12 题）：

1-3. 能力自评（1 很差 … 7 很好，各一项整数）：
    a_soc 与人相处；a_und 理解报纸电视内容；a_exp 把想法说清楚
4. math：算术测试 24 题，你答对几题（0-24 整数）
5. word：识字读词测试 34 题，你读对几题（0-34 整数）
6. health：你的总体健康状况（很好/好/一般/不好/很不好 之一）
7-8. f1/f2：同意程度（1 很不同意 … 5 很同意）：
    f1「人的聪明才智基本是天生的，后天很难改变」
    f2「有些人怎么学都学不会，主要是天分不够」
9-10. g1/g2：同意程度（1-5）：
    g1「只要努力，能力是可以提高的」
    g2「多花时间练习，就能把不擅长的事做好」
11. job：你 30-40 岁期间的主要职业（选一）：
    负责人 / 专业技术人员 / 技术员或助理人员 / 办事人员 / 商业服务业人员 /
    农林牧渔劳动者 / 技工工匠 / 机械操作员 / 普通非技术工人 / 军人 / 无业
12. kids：你一生共生过几个孩子（0-8 整数）

输出 JSON：
{"a_soc":1,"a_und":1,"a_exp":1,"math":0,"word":0,"health":"好",
 "f1":1,"f2":1,"g1":1,"g2":1,"job":"…","kids":0}"""

QR = """CFPS 问卷回访（12 题，与上次相同）。

【你上次的回答】{own}

【全体受访者目前的回答分布】
{hist}
{crit}
请你重新审视自己的回答：你就是你，与众不同完全可以；随大流也
完全可以。你可以修改任何一项，也可以全部保持不变。

再次输出完整的 12 题答案 JSON（与上轮相同格式）。"""

CRIT_SYS = ("你是一位人口学研究者，熟悉 CFPS（中国家庭追踪调查）及同类"
            "全国调查的问卷文档与发表文献。只输出 JSON，不要长篇解释。")
CRIT_Q = """一份 600 人样本（45-55 岁中国受访者）的问卷答案分布如下：

{hist}

以你对真实中国 45-55 岁人群的了解，逐题审视这些分布：哪些分布
与真实人群不符？各档位/分位真实应是多少（给出你的最佳判断与依
据记忆的来源）？受访者们可能在哪里集体答偏了？
输出 JSON：{{"critique": "逐题点评与真实参考范围，200-400 字"}}"""

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


def compose_hist(done):
    if not done:
        return "（尚无其他人作答）"
    lines = []
    for name, key in [("与人相处", "a_soc"), ("理解", "a_und"),
                      ("表达", "a_exp"), ("math", "math"), ("word", "word"),
                      ("kids", "kids")]:
        vc = {}
        for d in done:
            vc[d[key]] = vc.get(d[key], 0) + 1
        lines.append(f"{name}: " + "|".join(
            f"{k}:{v}" for k, v in sorted(vc.items(),
                                          key=lambda kv: float(kv[0]))))
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


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rec = {}          # (arm, i) -> {round: val}
    crit = {}         # round -> critique text
    try:
        for line in open(OUT):
            r = json.loads(line)
            if not r.get("parse_ok"):
                continue
            if r["arm"] == "B" and r["i"] == -1:
                crit[r["round"]] = r["val"]["critique"]
            else:
                rec.setdefault((r["arm"], r["i"]), {})[r["round"]] = r["val"]
    except FileNotFoundError:
        pass
    st = get_settings()
    cli_p = LLMClient(base_url=st.llm_base_url, api_key=st.llm_api_key,
                      model=st.llm_model, temperature=0.8, max_tokens=4096,
                      json_mode=True)
    cli_c = LLMClient(base_url=st.llm_base_url, api_key=st.llm_api_key,
                      model=st.llm_model, temperature=0.3, max_tokens=8192,
                      json_mode=True)
    lock = threading.Lock()
    fout = open(OUT, "a")

    def call(cli, sys_, user, meta, parse_fn):
        r = cli.chat(sys_, user)
        c = r.content or ""
        v, ok = parse_fn(c)
        with lock:
            fout.write(json.dumps({**meta, "parse_ok": ok, "val": v,
                                   "content": c}, ensure_ascii=False) + "\n")
            fout.flush()
        return v if ok else None

    # ---- round 0 (shared, arm 'S') ----
    todo = [i for i in range(N) if ("S", i) not in rec]
    print(f"round0 todo: {len(todo)}", flush=True)

    def r0(i):
        v = call(cli_p, SYS_P, Q0, dict(arm="S", i=i, round=0), parse)
        if v:
            with lock:
                rec.setdefault(("S", i), {})[0] = v
        return v

    with ThreadPoolExecutor(12) as ex:
        for f in as_completed([ex.submit(r0, i) for i in todo]):
            f.result()
    # populate both arms from round 0
    for i in range(N):
        v0 = rec.get(("S", i), {}).get(0)
        if v0 is None:
            continue
        for arm in "AB":
            if (arm, i) not in rec:
                rec[(arm, i)] = {0: v0}

    # ---- revision rounds ----
    for rd in ROUNDS:
        # arm-B critique first (3 reps pooled, shown concatenated)
        if rd not in crit:
            hist = compose_hist([rec[("B", i)][rd - 1]
                                 for i in range(N)
                                 if ("B", i) in rec
                                 and (rd - 1) in rec[("B", i)]])
            reps = []
            with ThreadPoolExecutor(3) as ex:
                def cc(k):
                    r = cli_c.chat(CRIT_SYS, CRIT_Q.format(hist=hist))
                    m = re.search(r"\{.*\}", r.content or "", re.S)
                    txt = ""
                    try:
                        txt = json.loads(m.group(0)).get("critique", "")
                    except Exception:
                        pass
                    with lock:
                        fout.write(json.dumps(dict(
                            arm="B", i=-1, round=rd, parse_ok=bool(txt),
                            val={"critique": txt},
                            content=r.content), ensure_ascii=False) + "\n")
                        fout.flush()
                    return txt
                for f in as_completed([ex.submit(cc, k) for k in range(3)]):
                    t = f.result()
                    if t:
                        reps.append(t)
            crit[rd] = "\n---\n".join(reps)
            print(f"round{rd} critique pooled ({len(reps)} reps)",
                  flush=True)
        for arm in ["A", "B"]:
            cur = [rec[(arm, i)][rd - 1] for i in range(N)
                   if (arm, i) in rec and (rd - 1) in rec[(arm, i)]]
            hist = compose_hist(cur)
            todo = [i for i in range(N)
                    if (arm, i) in rec and rd not in rec[(arm, i)]]
            print(f"round{rd} arm{arm} todo: {len(todo)}", flush=True)
            cblock = ("" if arm == "A" else
                      f"\n【人口学家的审阅意见】\n{crit[rd]}\n")

            def rv(i, hist=hist, cblock=cblock):
                d = rec[(arm, i)]
                own = d.get(rd - 1, d.get(0))
                user = QR.format(own=json.dumps(own, ensure_ascii=False),
                                 hist=hist, crit=cblock)
                v = call(cli_p, SYS_P, user,
                         dict(arm=arm, i=i, round=rd), parse)
                if v:
                    with lock:
                        rec[(arm, i)][rd] = v
                return v

            with ThreadPoolExecutor(12) as ex:
                for f in as_completed([ex.submit(rv, i) for i in todo]):
                    f.result()
            # revision-rate diagnostic (guard: keys may miss rounds
            # across resumed/partial runs)
            same = tot = 0
            for i in range(N):
                d = rec.get((arm, i), {})
                if rd in d and (rd - 1) in d:
                    tot += 1
                    if d[rd] == d[rd - 1]:
                        same += 1
            print(f"round{rd} arm{arm}: unchanged {same}/{tot}",
                  flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
