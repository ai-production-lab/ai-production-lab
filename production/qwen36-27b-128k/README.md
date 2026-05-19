# 生产配置：Qwen3.6-27B · 128K（R2-A1）

两轮调优后的**当前上线**脚本，与实验目录解耦，避免误跑矩阵变体。

```bash
bash production/qwen36-27b-128k/docker-run.sh
```

等价于 `experiments/.../round2-matrix/configs/docker-run-r2-a1-recommended.sh`。
