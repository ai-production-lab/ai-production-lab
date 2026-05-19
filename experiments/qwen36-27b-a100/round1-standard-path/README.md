# 第一轮调优：标准路径（standard_path）

在 64K 合同验收通过后，自动化跑 **12 个阶段**（T1～T9），回答：

- 128K / 200K / 256K 能否稳定启动？
- prefix cache、MTP、FP8 KV 是否值得默认开启？
- 第一轮收敛主力：**T7b**（128K + FP8 KV + prefix + MTP×2）

## 使用

```bash
cd experiments/qwen36-27b-a100/round1-standard-path
bash scripts/run_standard_path.sh
# 或：bash scripts/start_standard_path_tmux.sh
```

## 目录

| 路径 | 说明 |
|------|------|
| `configs/docker-run-baseline-64k.sh` | 64K 合同基线 |
| `configs/docker-run-main-128k-recommended.sh` | 第一轮推荐 T7b（第二轮 R2-A0 对照基准） |
| `scripts/run_standard_path.sh` | 标准路径主编排 |
| `analysis/ANALYSIS-optimal-config.md` | 第一轮结论摘要 |

第二轮在此基础上做细粒度扫描，见 [../round2-matrix/README.md](../round2-matrix/README.md)。
