# A100 Qwen3.6-27B 调优结论（基于 standard_path 实测）

数据来源：`standard_path-20260519-032450.log`、`t1a-*.log`，标准路径全自动跑通，无 OOM/失败。

---

## 结论摘要

| 用途 | 推荐配置 |
|------|---------|
| **日常主力（推荐）** | **128K + FP8 KV + prefix cache + MTP**（T7b） |
| **偏重智力/少改动** | **128K + BF16 KV + prefix cache**（无 MTP，T7 已验证可启动） |
| **超长文探边** | 200K/256K + FP8 + prefix；256K 需 `max_num_seqs=1` |
| **不建议默认** | 64K 合同档（能跑，但低于官方对思考模型 ≥128K 的建议） |

**当前线上容器**：自动化结束时停在 **T9（256K）**，请切回主力脚本：

`bash ~/vllm-qwen36-tuning/configs/docker-run-main-128k-recommended.sh`

---

## 实测数据表

### 64K 性能（短请求 / 512 token 生成）

| 阶段 | prefix | MTP | FP8 | 短请求 | 512 token 耗时 | 约 tokens/s |
|------|--------|-----|-----|--------|----------------|-------------|
| T1-A 基线 | 否 | 否 | 否 | 0.58s | 10.72s | **~48** |
| T3 | 是 | 否 | 否 | 2.33s* | 10.80s | ~47 |
| T4 | 是 | 是 | 否 | — | 4.37s（128 tok） | ~29† |
| T7b（128K） | 是 | 是 | 是 | — | 12.70s | **~40** |

\* T3 后短请求变慢，可能含冷启动/调度因素，不宜单独解读。  
† T4 仅测 128 token，与 512 token 不可直接对比。

**A3 相同 system prompt（两次）**：基线 0.76s / 0.75s；T3 后 0.81s / 0.77s。日志 **Prefix cache hit rate 仍为 0%**（系统 prompt 过短，未体现收益）。**仍建议生产开启 prefix cache**（长 system/长文档场景才有意义）。

### 长上下文 prefill（6 万汉字档测试，+64 token 输出）

| 标签 | max_model_len | KV | 实测 prompt_tokens | 总耗时 | 启动 API |
|------|---------------|-----|------------------|--------|----------|
| 128K-baseline | 131072 | BF16 | 24016 | 7.9s | ~270s |
| 200K-baseline | 204800 | FP8 | 40016 | 12.8s | ~270s |
| 256K-baseline | 262144 | FP8 | 52016 | 16.8s | ~270s |
| 128K-prefixcache | 131072 | BF16 | 24016 | 7.9s | ~270s |
| 128K-main (T7b) | 131072 | FP8+MTP | 24016 | **7.4s** | ~300s |
| 200K-prefixcache | 204800 | FP8 | 40016 | 13.3s | ~270s |
| 256K-prefixcache | 262144 | FP8 | 52016 | 17.9s | ~270s |

- **128K 在 BF16 KV 下即可稳定启动**（T1-B1、T7 未触发 FP8 回退）。
- 上下文越长，prefill 近似线性变慢；**200K/256K 可用但不适合作为默认**。
- 显存占用：空闲约 **74643～75695 MiB/卡**（随 max_model_len 与配置略变）。

### 稳定性

- 标准路径 **12/12 步骤全部 DONE**，无容器启动失败、无 HTTP 非 200。
- 冷启动至 API 可用：**约 270～300s**（27B TP=2）。

---

## 与三大诉求的对照

| 诉求 | 结论 |
|------|------|
| **上下文** | **128K 为均衡点**：BF16 可跑、延迟可接受、满足 README ≥128K 思考能力；200K/256K 仅按需切换。 |
| **速度** | 64K 下 512 token 约 **48 tok/s**；128K+T7b 同测约 **40 tok/s**。MTP 在本轮短测中**未体现明显加速**，可保留（官方推荐）或改用无 MTP 的 128K BF16 配置做 A/B。 |
| **智力** | 权重保持 **BF16**；128K 优于 64K 默认合同；KV 优先 **BF16**，若开 MTP 可同时 **FP8 KV**（T7b 实测正常）。 |

---

## 推荐 docker 参数（日常）

见 `configs/docker-run-main-128k-recommended.sh`（T7b）。

可选简化版（128K BF16，无 MTP，略省心）：

```bash
--max-model-len 131072
--max-num-seqs 2
--gpu-memory-utilization 0.92
--enable-prefix-caching
# 不加 --kv-cache-dtype fp8
# 不加 --speculative-config
```

超长文临时切 256K 时：

```bash
-e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
--max-model-len 262144
--max-num-seqs 1
--gpu-memory-utilization 0.95
--kv-cache-dtype fp8
--enable-prefix-caching
```

---

## 未测 / 待手工

- **T2** Cherry 三套采样（客户端，不影响服务端最优参数）。
- **Prefix cache** 需在「长 system + 多轮相同前缀」真实业务下再观察 hit rate。
- **MTP** 建议在典型 2K～8K 输出任务上再测一轮 tok/s。
