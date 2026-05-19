# 第二轮调优：128K 细粒度矩阵（R2）

在第一轮 **T7b（R2-A0 对照）** 上，单因子扫描 **14 个变体**（P0～P2），统一基准 B1～B6。

## 使用

```bash
cd experiments/qwen36-27b-a100/round2-matrix

python3 scripts/benchmark_r2.py --label smoke
bash scripts/run_round2_full_auto.sh    # P0→P1→P2
# 或：bash scripts/start_round2_tmux.sh P0

python3 scripts/summarize_r2.py
bash scripts/r2_status.sh
```

## 目录

| 路径 | 说明 |
|------|------|
| `configs/docker-run-t7b-baseline-r2-a0.sh` | R2-A0 = 第一轮 T7b |
| `configs/docker-run-r2-a1-recommended.sh` | R2-A1 推荐（= 生产配置） |
| `scripts/benchmark_r2.py` | 统一基准 |
| `results/summary.csv` | 14 变体数据 |
| `results/ANALYSIS-r2-summary.md` | 第二轮结论 |

## 推荐变体

**R2-A1**：BF16 KV + prefix + MTP×2 → B2 相对 A0 **+20%**
