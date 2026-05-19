# ai-production-lab

**AI实战落地** — 从模型到场景的工程化实践（可复现实验与配置）

组织主页：[https://github.com/ai-production-lab](https://github.com/ai-production-lab)

> 能力是长期的，场景是可替换的。本仓库沉淀部署脚本、benchmark 数据与系列文档。

## 系列目录

| # | 主题 | 文档 | 代码与数据 |
|---|------|------|-----------|
| 1 | 推理基座：Qwen3.6-27B @ 双 A100 | [掘金](TODO_JUEJIN_URL) · [仓库全文](docs/posts/01-qwen36-a100-vllm-tuning.md) | 本仓库 `configs/` `scripts/` `results/r2/` |
| 2 | 运营 Agent MVP | 计划中 | — |

## 第 1 篇：推荐配置（R2-A1）

**环境**：2 × A100 80GB · vLLM 0.19.1 · Qwen3.6-27B BF16 · TP=2

```bash
git clone https://github.com/ai-production-lab/ai-production-lab.git
cd ai-production-lab
export VLLM_TUNING_ROOT="$(pwd)"   # 脚本默认使用仓库根目录

bash configs/docker-run-main-128k-r2-final.sh
```

| 项 | 值 |
|----|-----|
| max_model_len | 131072 (128K) |
| KV | BF16（不启用 `--kv-cache-dtype fp8`） |
| prefix caching | 开启 |
| MTP | `num_speculative_tokens=2` |
| 相对 T7b 基准 B2 吞吐 | **+20%** |

### Benchmark 摘要

| 变体 | 变更 | B2 tok/s | B3(s) | B6 tok/s |
|------|------|----------|-------|----------|
| R2-A0 | T7b 基准（FP8 KV） | 70.95 | 7.41 | 73.08 |
| **R2-A1** | **BF16 KV（推荐）** | **85.01** | **6.96** | — |
| R2-A2 | 无 MTP | 47.49 | 7.45 | 46.35 |
| R2-A4 | MTP×4 | 80.17 | 7.05 | 82.95 |

完整数据：[results/r2/summary.csv](results/r2/summary.csv) · 分析：[results/r2/ANALYSIS-r2-summary.md](results/r2/ANALYSIS-r2-summary.md)

## 仓库结构

```text
ai-production-lab/
├── README.md
├── requirements.txt
├── configs/
│   └── docker-run-main-128k-r2-final.sh    # 推荐生产启动（R2-A1）
├── scripts/
│   ├── benchmark_r2.py                   # 统一基准 B1–B6
│   ├── run_round2_matrix.sh              # P0/P1/P2 矩阵
│   ├── run_round2_variant.sh             # 单变体
│   ├── run_round2_full_auto.sh           # 全自动 P0→P1→P2
│   ├── summarize_r2.py                   # JSON → summary.csv
│   ├── r2_status.sh                      # 状态快照
│   └── start_round2_tmux.sh
├── results/r2/
│   ├── summary.csv                       # 14 变体实测汇总
│   └── ANALYSIS-r2-summary.md
└── docs/posts/
    └── 01-qwen36-a100-vllm-tuning.md     # 系列第 1 篇全文
```

## 复现第二轮调优

```bash
export VLLM_TUNING_ROOT="$(pwd)"
pip install -r requirements.txt

# 单变体冒烟
python3 scripts/benchmark_r2.py --label smoke

# 全自动（约数小时，会反复重建容器）
bash scripts/run_round2_full_auto.sh
# 或 tmux：bash scripts/start_round2_tmux.sh P0

python3 scripts/summarize_r2.py
bash scripts/r2_status.sh
```

将 `configs/docker-run-main-128k-r2-final.sh` 中的模型挂载路径改为你本机的权重目录。

## 验证 API

```bash
curl -s http://127.0.0.1:50600/v1/models | head -c 300
```

## 许可

代码与文档：MIT（见 [LICENSE](LICENSE)）。模型权重请遵循 [Qwen](https://github.com/Qwen) 官方许可。

---

**下一篇**：基于本推理基座的运营 Agent MVP（系列第 2 篇）。
