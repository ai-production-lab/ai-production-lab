# ai-production-lab

**AI实战落地** — 从模型到场景的工程化实践（可复现实验与配置）

组织：[https://github.com/ai-production-lab](https://github.com/ai-production-lab)

> 能力是长期的，场景是可替换的。本仓库按**系列文章 + 实验轮次 + 生产配置**组织，避免多篇文章共用一个扁平目录。

## 系列目录

| # | 主题 | 文章 | 实验与生产 |
|---|------|------|-----------|
| 1 | 推理基座：Qwen3.6-27B @ 双 A100（**两轮调优**） | [掘金](TODO_JUEJIN_URL) · [全文](docs/posts/01-qwen36-a100-vllm-tuning.md) | [experiments/qwen36-27b-a100/](experiments/qwen36-27b-a100/) |
| 2 | 运营 Agent MVP | 计划中 | — |

## 第 1 篇：两轮调优结论

| 轮次 | 做什么 | 结论入口 |
|------|--------|---------|
| **第一轮** | 标准路径：64K→128K/200K/256K，扫 prefix/MTP/FP8 | [round1 分析](experiments/qwen36-27b-a100/round1-standard-path/analysis/ANALYSIS-optimal-config.md) |
| **第二轮** | 128K 上 14 变体矩阵（R2-A0～C3） | [summary.csv](experiments/qwen36-27b-a100/round2-matrix/results/summary.csv) |

**生产上线（R2-A1）**：

```bash
git clone https://github.com/ai-production-lab/ai-production-lab.git
cd ai-production-lab
bash production/qwen36-27b-128k/docker-run.sh
```

| 项 | 值 |
|----|-----|
| max_model_len | 131072 (128K) |
| KV | BF16 |
| prefix + MTP×2 | 开启 |
| 相对第一轮主力配置 B2 | **+20%** tok/s |

## 仓库结构

```text
ai-production-lab/
├── README.md
├── docs/posts/                    # 系列文章（MD）
│   └── 01-qwen36-a100-vllm-tuning.md
├── production/                    # 当前线上配置（与实验分离）
│   └── qwen36-27b-128k/
│       └── docker-run.sh
└── experiments/                   # 按模型/课题分目录
    └── qwen36-27b-a100/            # 第 1 篇全部实验
        ├── README.md
        ├── round1-standard-path/  # 第一轮
        │   ├── configs/
        │   ├── scripts/
        │   └── analysis/
        └── round2-matrix/           # 第二轮
            ├── configs/
            ├── scripts/
            └── results/
                ├── summary.csv
                └── ANALYSIS-r2-summary.md
```

第 2 篇起建议新增 `experiments/<课题名>/`，**不要**堆在 `qwen36-27b-a100/` 下。

## 复现第二轮矩阵

```bash
cd experiments/qwen36-27b-a100/round2-matrix
pip install -r ../../../requirements.txt
bash scripts/run_round2_full_auto.sh
python3 scripts/summarize_r2.py
```

## 许可

代码与文档：MIT。模型权重遵循 Qwen 官方许可。
