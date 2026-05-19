# Qwen3.6-27B @ 双 A100 — 实验目录

系列第 1 篇对应的**全部可复现资产**，分两轮调优，不要与「仅 R2」混淆。

| 轮次 | 目录 | 目标 |
|------|------|------|
| **第一轮** | [round1-standard-path/](round1-standard-path/) | 64K 合同 → 128K/200K/256K 档位；prefix / MTP / FP8 探索（标准路径 12 步） |
| **第二轮** | [round2-matrix/](round2-matrix/) | 在 128K T7b 基准上 14 变体单因子矩阵（R2-A0～C3） |

**当前生产配置**（两轮结论合并）：[../../production/qwen36-27b-128k/docker-run.sh](../../production/qwen36-27b-128k/docker-run.sh)（R2-A1：BF16 KV + prefix + MTP×2）

文章全文：[../../docs/posts/01-qwen36-a100-vllm-tuning.md](../../docs/posts/01-qwen36-a100-vllm-tuning.md)
