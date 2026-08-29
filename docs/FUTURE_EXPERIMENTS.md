# 下一阶段实验清单

## 0. 当前边界

本文档记录完成当前论文版本后再开展的实验。现阶段的优先任务是让同事能够准确理解并讨论现有 pipeline：输入是什么、LLM 在每个阶段回答什么问题、回答如何转化为总体目标、loading operator 如何修改合成表，以及现有结果能够支持什么结论。

以下实验暂不进入当前正文结果，也不影响当前版本的写作完成。开始新实验前，应先冻结当前论文中使用的 prompt、解析规则、target pooling、operator 顺序、超参数和评测口径。

## 1. P0：多模型验证

### 1.1 研究问题

当前结果只来自 GLM-5.2。多模型实验需要回答：

1. population elicitation and loading 的提升是否可以跨模型复现？
2. 不同模型的 direct seed、ADA judgments 和 population-level estimates 分别有多大差异？
3. 模型的直接生成能力与其总体统计判断能力是否一致？
4. pipeline 的收益主要来自较好的 seed，还是较好的 elicited targets？

### 1.2 模型选择原则

至少覆盖以下模型类型：

| 类型 | 目的 | 模型名处理方式 |
|---|---|---|
| 当前模型 | 作为已有结果和复现实验的基准 | GLM-5.2 |
| 强闭源通用模型 | 检查 frontier reasoning/generation 能力是否改善总体统计判断 | 实验启动时冻结具体版本 |
| 中国语境能力较强的模型 | 检查中文人口知识与 CFPS 场景的影响 | 实验启动时冻结具体版本 |
| 开源或开放权重模型 | 检查方法是否依赖专有模型，并支持可复现性 | 记录精确 checkpoint |
| 中小规模模型 | 检查 pipeline 是否能补偿较弱的直接生成能力 | 记录参数规模和 checkpoint |

第一轮建议使用 **4–6 个模型**。所有模型必须记录精确名称、版本或 checkpoint、访问日期、API 参数和上下文长度，不能只写模型家族名称。

### 1.3 最小多模型实验矩阵

第一轮只运行两个系统：

| 条件 | Seed model | Elicitation model | Loading | 目的 |
|---|---|---|---|---|
| Direct | 模型 $M$ | 无 | 仅标准后处理 | 得到每个模型的直接生成基线 |
| Full pipeline | 模型 $M$ | 同一个模型 $M$ | 冻结后的完整 operators | 检验同模型端到端提升 |

每个模型至少运行 3 个端到端随机种子；资源允许时使用 5 个。每个端到端结果再运行固定次数的 evaluator bootstrap。报告时必须区分：

- 不同 synthetic population 之间的 end-to-end variation；
- 同一 synthetic population 上的 evaluator variation。

### 1.4 第二轮模型解耦实验

第一轮确认方法可以跨模型运行后，再区分 seed quality 与 elicitation quality：

| 条件 | Seed | Elicitation targets | 目的 |
|---|---|---|---|
| Same-model | $M_i$ | $M_i$ | 标准完整 pipeline |
| Fixed-seed elicitor comparison | 固定同一 seed table | $M_1,\ldots,M_k$ | 单独比较不同模型的总体统计判断 |
| Cross-model loading | $M_i$ seed | $M_j$ targets | 检查 targets 是否可以跨生成模型迁移 |

该实验可以回答“某模型直接生成很强，是否也更会估计 marginals/relationships”以及“较强 elicitor 能否修正较弱 generator”。

## 2. P0：完整方法的消融实验

消融必须围绕最终冻结版本设计，不能把研究过程中的旧目录直接当作消融结果。

建议的最小条件为：

1. Direct seed generation。
2. Seed + direct marginal prompts/loading。
3. Seed + ADA only。
4. Seed + population-level prompts without ADA。
5. Full pipeline without marginal loaders。
6. Full pipeline without dependence loader。
7. Full pipeline without $R^2$ loader。
8. Full pipeline without event-order loader。
9. Complete full pipeline。

如果成本需要压缩，优先保留 1、2、3、4、6、8、9。每个条件使用相同 conditioning profiles、相同 seed table 和相同 evaluator 配置，从而把差异限制在被移除的组件上。

主要回答：

- ADA 是否比直接询问 category shares 提供额外价值？
- Devil selection 是否优于随机 legal-neighbor selection？
- forward/reverse arbitration 是否降低位置偏差？
- Hodge aggregation 是否优于直接平均或直接比例询问？
- 每类 loader 是否主要改善其对应的 SSDataBench type？

## 3. P0：端到端稳定性

当前五次结果是同一张 final table 的 evaluator reruns。下一阶段需要独立重复整个过程：

1. 重新生成 1,000-row seed population；
2. 重新运行 ADA；
3. 重新运行全部 population probes；
4. 重新构造 targets；
5. 重新运行所有 loading operators；
6. 重新评测 final table。

每次运行必须使用稳定、可记录的随机种子。至少报告：

- T1–T5 和 Overall 的均值、标准差和区间；
- 每类 elicited target 的跨运行差异；
- target-versus-achieved error；
- LLM parse failure、retry 和有效回答比例；
- 运行时间、token 使用量和成本。

## 4. P0：最终表的一致性与有效性检查

当前 final artifact 是 population-calibrated analysis table。若论文希望进一步称其为 respondent-level synthetic microdata，需要增加以下检查。

### 4.1 Schema validity

- 每个字段是否处于允许范围或允许类别中；
- integer/atom-valued 字段是否仍落在合法网格上；
- missingness 是否被意外改变；
- clipping 后是否出现不合理的边界堆积。

### 4.2 Cross-field consistency

- highest education 与 education-completion age 是否相容；
- education、occupation、income 和 employment 是否出现明显冲突；
- child number 与 first-child age 是否相容；
- ever divorced 与 marriage trajectory 是否相容。

### 4.3 Longitudinal consistency

- summary event ages 是否与 annual trajectory 中的事件发生年龄一致；
- event-order loading 后，education/marriage/childbirth summaries 是否需要同步写回 trajectory；
- annual education 是否单调；
- child number 是否随年龄非递减；
- occupation、employment 与 income 是否满足基本状态约束。

报告每项规则的违反率，并同时给出 direct seed 与 final table，避免只报告最终绝对值。

## 5. P1：target-versus-achieved diagnostics

SSDataBench 汇总分数之外，还应报告各阶段是否成功装载了自己声称的目标：

| 模块 | 建议指标 |
|---|---|
| Marginals | 分类 total variation；数值 KS/Wasserstein；分位数绝对误差 |
| ADA | graph coverage；Hodge residual；forward/reverse disagreement |
| Relationships | target/achieved correlation scatter；matrix MAE；PSD projection displacement |
| $R^2$ | target、loader 后 achieved、最终表 achieved 三组数值 |
| Event order | target/achieved six-state distribution；total variation；被修改行数和修改成本 |

这些指标需要同时按变量或变量对报告分布，不能只报告一个平均值。

## 6. P1：event-order 假设检验

最终版本用三条 precedence probabilities 加上

\[
p_{CEM}=p_{CME}=0
\]

识别六类 event-order target。下一阶段需要比较：

1. 当前 zero-C-first constraint；
2. Plackett–Luce factorization；
3. 直接询问六类 pathway shares；
4. pairwise + direct-six-state 的融合目标。

比较标准包括 T4、T5、target self-consistency、跨 prompt 重复稳定性，以及是否需要把 0 概率改为小的平滑概率。

## 7. P1：跨数据集验证

多模型验证之后，在另一项 SSDataBench 数据集上冻结迁移方法。优先选择同时具有：

- 清楚的 conditioning variables；
- 多种连续和分类 outcomes；
- 可评估的 dependence structure；
- 与 CFPS 不同的国家、年份或调查领域。

需要区分两种迁移：

1. **Protocol transfer**：保留 elicitation 和 loading 规则，只替换 schema 与人口描述。
2. **Full retuning**：允许重新选择 prompt 与 operator 参数。

论文中最有说服力的是 protocol transfer，因为它检验方法而非针对一个 benchmark 的重新开发。

## 8. 推荐执行顺序

完成当前论文版本并与同事确认后，按以下顺序推进：

1. 冻结当前 pipeline specification 与实验配置。
2. 建立 schema、cross-field 和 trajectory consistency tests。
3. 在 GLM-5.2 上完成 3–5 个端到端种子。
4. 完成最小 final-method ablation。
5. 运行第一轮 4–6 模型 direct/full matrix。
6. 根据第一轮结果决定是否开展 cross-model seed/elicitor 解耦。
7. 补充 target-versus-achieved diagnostics。
8. 比较 event-order target specifications。
9. 最后开展跨数据集 protocol transfer。

## 9. 当前版本完成的判定标准

进入新实验前，当前论文版本至少应达到：

- 同事能够仅凭 Figure 1、Method 和 Appendix 复述完整 pipeline；
- 每个 LLM call 的输入、角色、输出和 pooling rule 都有文字定义；
- 每个 loading operator 的输入、公式、保留项和副作用都清楚；
- 现有 0.197→0.653 结果的评测口径没有歧义；
- proof-of-concept 结论与尚未完成的实验边界分开；
- 需要讨论的方法假设已经显式列出，尤其是 ADA typicality-to-prevalence 与 event-order zero-C-first constraint。

## 10. 已实现的实验入口（protocol v1）

`scripts/future_experiments.py` 已把 P0 第一轮收敛为可续跑的实验单元。每个单元固定
`model × end-to-end seed`，先生成 direct seed，再独立运行两轮 ADA、T1 marginal、
T2 dependence、T3 R² 和 T4 event-order probes，按 marginal → dependence → R² →
event-order 的顺序装载，并使用固定 evaluator seeds 重复评测。每次调用记录 requested
model、provider 实际返回的 resolved model、token 和 cost；OpenRouter 套件共用美元账本，
达到预算后在下一次请求前停止。

冻结配置为 `configs/experiments/future_p0.yaml`。默认第一轮只运行 `direct` 和 `full`；
runner 同时支持 `marginal_only`、`ada_only`、`population_without_ada`、
`full_without_marginal`、`full_without_dependence`、`full_without_r2` 和
`full_without_event_order`，用于随后启动最小消融矩阵。

```bash
# 查看模型、seed 和保守成本估计
uv run python scripts/future_experiments.py plan --provider openrouter

# 远端本地端点 smoke（20 人、每类 probe 1 次、B=5）
uv run python scripts/future_experiments.py run \
  --provider local --models local_glm52 --seeds 4101 \
  --conditions direct,full --smoke \
  --suite runs/experiments/future_p0_local_smoke_v1

# OpenRouter 正式矩阵；OPENROUTER_API_KEY 只放在远端 .env
uv run python scripts/future_experiments.py run \
  --provider openrouter --conditions direct,full \
  --suite runs/experiments/future_p0_openrouter_v1
```

每个 condition 同时产出：五类 benchmark 结果、固定 bootstrap seed 的 evaluator variation、
schema/cross-field/trajectory consistency、target-versus-achieved diagnostics。套件完成后，
`summary/` 汇总 evaluator runs、每张 synthetic population、跨 end-to-end seeds 和模型成本。
