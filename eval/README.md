# Qwen3.6-27B 生产评测（两套独立）

对已部署的 vLLM 服务做评测时，**性能**与**能力**分开跑、分开存档，不要混在一张表。

| 套系 | 目录 | 回答的问题 | 主要工具 |
|------|------|-----------|----------|
| **套系 A：性能** | [performance/](performance/) | 吞吐、延迟、长 prefill、prefix、MTP | `benchmark_r2.py` + EvalScope perf |
| **套系 B：能力** | [capability/](capability/) | MMLU/C-Eval/GSM8K 等准确率 | EvalScope eval（OpenAI API） |

## 前置条件

1. vLLM 已按 [production/qwen36-27b-128k/docker-run.sh](../production/qwen36-27b-128k/docker-run.sh) 启动
2. `curl http://<HOST>:50600/v1/models` 返回 200
3. 评测期间 **GPU 独占**（能力评测耗时长，勿与压测并行）

## 快速开始

```bash
# 1. 复制并编辑环境变量
cp eval/config.env.example eval/config.env
# 编辑 VLLM_BASE_URL、VLLM_MODEL_ID

# 2. 安装依赖（建议在独立 venv）
pip install -r eval/requirements-eval.txt

# 3. 套系 A：性能（在 GPU 服务器上跑，约 5～15 分钟）
bash eval/performance/run_deploy_benchmark.sh
bash eval/performance/run_evalscope_perf.sh      # 可选，通用 serving 指标

# 4. 套系 B：能力（可远程 API，quick 约 30～60 分钟，视 limit 而定）
bash eval/capability/run_evalscope_capability.sh quick
# 完整评测：bash eval/capability/run_evalscope_capability.sh standard
```

## 结果目录

```text
eval/results/
├── performance/
│   ├── deploy/          # benchmark_r2.py JSON（可与 R2-A1 对照）
│   └── evalscope/       # EvalScope perf 输出
└── capability/
    └── evalscope/       # EvalScope eval 报告
```

原始结果默认 **不入库**（见 `.gitignore`）。请把摘要抄入 `eval/results/BASELINE.md` 或 Wiki。

## 参照基线（R2-A1，双 A100）

| 指标 | 参考值 | 来源 |
|------|--------|------|
| 512 token 生成 (B2) | **~85 tok/s** | round2 summary.csv |
| 长文 prefill (B3) | **~6.96 s** | 同上 |
| 2048 token 生成 (B6) | —（R2-A1 未跑 B6） | A0 约 73 tok/s |
| prefix 降幅 (B4) | **~52%** | 同上 |

## Qwen3.6 思考模型注意

- **性能套系 B2/B6**：脚本已用官方思考采样（temp=1.0, top_p=0.95, top_k=20）
- **能力套系**：客观题（GSM8K、C-Eval）默认同样思考采样；若与官方榜数字对比，需确认榜单是否关思考，必要时在 `task_*.yaml` 中加 `enable_thinking: false`
- **能力 ≠ 性能**：能力分下降不说明吞吐变差，反之亦然

## 与调优实验的关系

| 工具 | 用途 |
|------|------|
| `experiments/.../benchmark_r2.py` | 第二轮 14 变体矩阵（改 docker 参数时用） |
| `eval/performance/run_deploy_benchmark.sh` | **当前生产配置**快照，label=`prod-local-*` |
| EvalScope | 对外可引用的通用 perf + 能力榜 |
