# ssbench 使用指南

SSDataBench（GSS + CFPS）独立复现与扩展实验框架的操作手册。

## 目录

- [环境](#环境)
- [项目结构](#项目结构)
- [完整流程](#完整流程)
- [命令详解](#命令详解)
- [断点续跑](#断点续跑)
- [产物说明](#产物说明)
- [结果对照](#结果对照)
- [常见问题](#常见问题)

## 环境

```bash
cd /home/chengkaiyue/code/ky/ssdatabench
uv sync    # 用 .venv（清华源已在 pyproject.toml 里配置）
```

注意：本机代理会拦截清华源，如果 `uv sync` 超时，用：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY uv sync
```

LLM 端点在 `.env` 里配置：

```ini
SDTL_LLM_BASE_URL=http://localhost:39001/v1   # vLLM OpenAI 兼容端点
SDTL_LLM_API_KEY=not-needed
SDTL_LLM_MODEL=glm-5.2
SDTL_LLM_CONCURRENCY=30                       # 仿真并发数
```

## 项目结构

```
configs/
  datasets/{gss,cfps}.yaml     # 仿真规格：变量、prompt 上下文、预处理指令
  eval/{gss,cfps}.yaml         # 评估规格：T1–T5 变量清单（从原版忠实移植）
src/ssbench/
  datasets/                    # 规格加载
  preprocessing/               # 真实数据清洗、派生、采样
  llm/                         # OpenAI 兼容客户端（重试/退避）
  simulation/
    methods/                   # 生成方法，可插拔（direct = 论文基线）
    prompts.py                 # Prompt 构建（逐字移植原版）
    lifecycle.py               # 生命历程派生变量
    checkpoint.py              # 逐条落盘 + 断点续跑
    runner.py                  # run 编排
  evaluation/                  # T1–T5 评估（忠实移植 SSDataBench）
scripts/
  prepare_data.py              # 数据预处理
  simulate.py                  # 仿真
  evaluate.py                  # 评估
  probe_truncation.py          # 截断探测（诊断用）
data/
  real_data/                   # 原始数据（gss2018.csv / cfps_2010_2022.csv）
  processed/                   # 预处理后的参照样本
runs/                          # 实验产物，一个 run 一个目录
```

## 完整流程

### 0. 数据预处理（一次性）

```bash
uv run python scripts/prepare_data.py --dataset all
```

- GSS：2347 行 → 采样 1000（seed=42），31 列
- CFPS：58474 行 → 完整输入过滤（33141）→ 派生 `occupation_30_40` / `mean_income_30_40` → 采样 1000，25 列
- 产物：`data/processed/<dataset>/sample.csv`

### 1. 仿真（放 tmux 里跑）

```bash
tmux new -s ssb
cd /home/chengkaiyue/code/ky/ssdatabench

# GSS：1000 次短调用，约 10-20 分钟
uv run python scripts/simulate.py --dataset gss --n 1000 --tag glm52

# CFPS：1000 次长轨迹调用（14→45 岁完整人生），约 1-1.5 小时
uv run python scripts/simulate.py --dataset cfps --n 1000 --tag glm52
```

- 离开按 `Ctrl-b d`，回来 `tmux attach -t ssb`
- 进度：stdout 每 10 个 profile 打一条，带速率和 ETA
- 结束后 stdout 会打印 run 目录路径，例如
  `runs/gss/direct/20260817_160835_glm52`

### 2. 评估

```bash
# GSS：T1–T3
uv run python scripts/evaluate.py --run-dir runs/gss/direct/<run_id>

# CFPS：T1–T5，约 20-40 分钟
uv run python scripts/evaluate.py --run-dir runs/cfps/direct/<run_id>
```

结果：`runs/<ds>/direct/<run_id>/evaluation/overall_summary.csv`

### 3. 查看结果

```bash
cat runs/cfps/direct/<run_id>/evaluation/overall_summary.csv
```

```
type,avg_insignificant_rate
t1,0.xxx
t2,0.xxx
...
overall,0.xxx
```

## 命令详解

### simulate.py

```bash
uv run python scripts/simulate.py --dataset cfps --method direct --n 1000 \
    --model glm-5.2 --temperature 1.0 --top-p 1.0 \
    --max-attempts 4 --seed 42 --tag glm52
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--dataset` | 必填 | `gss` / `cfps` |
| `--method` | `direct` | 生成方法（新方法在 `src/ssbench/simulation/methods/` 加文件夹注册） |
| `--n` | 1000 | 生成个体数 |
| `--model` | .env 的 SDTL_LLM_MODEL | 模型名 |
| `--temperature` / `--top-p` | 1.0 / 1.0 | 采样参数（对齐论文） |
| `--max-tokens` | gss 16384 / cfps 32768 | 单次响应预算 |
| `--max-attempts` | 4 | 单个 profile 的最大重试次数 |
| `--seed` | 42 | 输入采样种子 |
| `--tag` | 无 | 目录名后缀 |
| `--resume-dir` | 无 | 断点续跑，见下文 |

临时调整并发：`SSBB_MAX_WORKERS=50 uv run python scripts/simulate.py ...`

### evaluate.py

```bash
uv run python scripts/evaluate.py --run-dir <run_dir> [--types t1,t2] [--B 30] [--sample-n 500]
```

| 参数 | 说明 |
|---|---|
| `--types` | 只跑部分评估（默认按数据集配置：GSS t1-t3，CFPS t1-t5） |
| `--B` | bootstrap 轮数（默认 100；调参时用 30 快速看分） |
| `--sample-n` | 每次 bootstrap 采样数（默认 500） |

耗时参考（n=1000，B=100）：GSS 全量约 10 秒，CFPS 全量约 10 秒。热路径为
numpy 向量化实现（解析梯度 Cramér's V、闭式 η²、bincount 列联表、
multinomial 重采样），与原版 pandas/autograd/statsmodels 实现统计语义一致
（检验函数对拍 max |Δp| < 1e-14）。

### probe_truncation.py（诊断）

```bash
uv run python scripts/probe_truncation.py --dataset cfps --profile 0 --dump /tmp/raw.json
```

检查单次 CFPS 轨迹生成是否截断：finish_reason、token 用量、14–45 岁覆盖、JSON 完整性。

## 断点续跑

仿真过程中每个完成的 profile 会即时写入 `partials.jsonl`。中断后：

```bash
uv run python scripts/simulate.py --dataset cfps --n 1000 \
    --resume-dir runs/cfps/direct/<run_id>
```

- 已完成的 profile_id 自动跳过，只补缺失的
- seed 与原 run 不一致会打 WARNING（输入行可能对不上）
- 评估是纯本地计算，无需续跑

## 产物说明

每个 `runs/<dataset>/<method>/<run_id>/`：

| 文件 | 内容 |
|---|---|
| `real.csv` | 条件输入（真实参照行，含 profile_id） |
| `sim.csv` | 合成数据（CFPS 含逐年展开列 + 派生列） |
| `partials.jsonl` | checkpoint：每个完成 profile 一条，崩溃安全的逐条落盘 |
| `responses.jsonl` | 每次 LLM 调用的完整审计：时间戳、finish_reason、usage、完整 raw 响应 |
| `meta.json` | run 配置、耗时、n_complete、n_checkpointed、resumed_from |
| `evaluation/overall_summary.csv` | T1–T5 分数 + 总平均 |
| `evaluation/summary_t*.csv` | 各类型的逐变量明细 |
| `evaluation/data_t*/` | 附加数据（熵、事件顺序熵等） |

## 结果对照

| 参考 | 分数 |
|---|---|
| real vs real 自检（CFPS，B=30） | T1 0.96 / T2 0.95 / T3 0.80 / T4 1.0 / T5 0.98 |
| SSDataBench 论文 15 模型最好水平 | 单数据集约 0.3–0.5，T4/T5 普遍 <0.1 |
| glm-5.2 direct 基线（n=5 冒烟） | 低分属正常（5 vs 1000 做检验） |

指标含义：avg_insignificant_rate = bootstrap 检验 p>0.05 的比例，**越高越好**，1.0 表示合成分布与真实统计不可区分。

## 常见问题

**Q：评估分数为 nan？**
n 太小（如冒烟 n=5）时部分变量/组合无有效样本，属预期；n=1000 不会出现。

**Q：CFPS 会不会截断？**
glm-5.2 实测单条约 3.3–3.7k completion tokens（预算 32768），finish_reason=stop，32 年全覆盖。可随时用 `probe_truncation.py` 复查。

**Q：怎么换模型对比？**
`--model glm-5.3 --tag glm53`，各 run 目录独立，评估逐个跑即可。

**Q：加新方法？**
在 `src/ssbench/simulation/methods/` 下新建文件，实现 `generate(spec, inputs_df, failure_logger, checkpoint) -> DataFrame` 并 `@register_method`，然后 `--method <name>` 调用。

**Q：bootstrap 分数有波动？**
评估随机性无固定 seed（与原版一致），B=100 下约 ±0.02–0.05。
