# ssbench

SSDataBench（GSS + CFPS）的独立复现与扩展实验框架。

**操作手册见 [docs/HOWTO.md](docs/HOWTO.md)** —— 完整流程、命令详解、断点续跑、产物说明、常见问题。

## 结构

```
configs/datasets/    数据集仿真规格（变量、prompt 上下文）
configs/eval/        评估规格（T1–T5 变量清单）
src/ssbench/         核心包
  datasets/            数据集规格加载
  preprocessing/       真实数据清洗与采样
  llm/                 OpenAI 兼容客户端
  simulation/          仿真（方法可插拔：methods/）
  evaluation/          T1–T5 评估（忠实移植 SSDataBench）
scripts/             命令行入口（薄封装）
data/processed/      预处理后数据
runs/                实验产物（每个 run 一个目录）
```

## 快速开始

```bash
uv sync                                            # 环境
uv run python scripts/prepare_data.py --all        # 数据预处理
uv run python scripts/simulate.py --dataset cfps --method direct --n 5
uv run python scripts/evaluate.py --run-dir runs/cfps/direct/<run_id>
```
