# 双 A100 部署 Qwen3.6-27B：两轮 vLLM 调优，128K 吞吐提升 20% 的完整记录

> **系列**：AI实战落地 · 推理基座（第 1 篇）
> **硬件**：双 NVIDIA A100-SXM4-80GB（NVLink） · **软件**：vLLM v0.19.1 · Qwen3.6-27B BF16

---

要让 AI 真正在行业里跑起来，第一步不是选模型，而是把**推理基座**搭稳。这篇记录从裸机到 **128K 上下文可用**的完整调优过程——当前服务于公路运营 Agent 的推理后端，但部署与调优方法本身适用于任何需要长上下文、思考模式推理的场景。

**本篇你能带走什么：**

- 一套可复现的 vLLM 部署与两轮自动化调优流程
- 14 组变体 benchmark 数据与最终推荐配置（相对 T7b，B2 吞吐 **+20%**）
- 官方 README 采样参数对齐、MTP/prefix/KV 的实测结论

---

## 背景

拿到一台双 A100 服务器，打算跑 Qwen3.6-27B 作为内部主力大模型。它是阿里开源的 36B MoE 架构模型（27B 激活参数），支持推理链和非推理两种模式，官方推荐上下文至少 128K 才能充分发挥思考能力。

三个核心诉求：

- **上下文够长**：能处理长文档、多轮对话
- **生成快**：实际使用体感好，不干等
- **智力有保障**：权重保持 BF16，不降精度

整个过程分**两轮调优**（不是只有第二轮 R2）：

- **第一轮（standard_path）**：64K 合同验收后，自动化跑 12 个阶段，确定 **128K 主力档位** 与 T7b 组合（FP8 KV + prefix + MTP）。
- **第二轮（R2 矩阵）**：在 T7b 基准上做单因子扫描（14 变体），最终推荐 **R2-A1**（BF16 KV + prefix + MTP×2）。

---

## 第一步：部署与合同验收

### 环境准备

服务器（`172.16.167.200`）上提前同步好 52 GB 的 BF16 权重至 `/data/models/Qwen3.6-27B/`，载入 vLLM 镜像 `vllm/vllm-openai:v0.19.1`。

最初的合同配置是 64K 上下文：

```bash
docker run -d --name vllm-qwen36-27b \
  --runtime nvidia --gpus all \
  --shm-size 16g --ipc host \
  -p 50600:8000 \
  -v /data/models/Qwen3.6-27B:/models/Qwen3.6-27B:ro \
  vllm/vllm-openai:v0.19.1 \
  --model /models/Qwen3.6-27B \
  --tensor-parallel-size 2 \
  --max-model-len 65536 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.92 \
  --language-model-only \
  --reasoning-parser qwen3
```

三个必开参数的原因：

- `--tensor-parallel-size 2`：双卡 NVLink，权重切分
- `--language-model-only`：跳过视觉塔，把显存全给 KV cache
- `--reasoning-parser qwen3`：正确解析 `<think>` 思考块

冷启动至 API 可用约需 **4～5 分钟**。

### 合同验收（§4）

三项验收全部通过：

1. `GET /v1/models` 返回 200，模型 ID `/models/Qwen3.6-27B`，`max_model_len=65536`
2. 短对话（`你好`）正常响应
3. 64K token 输入请求正常完成

Win11 工作站（`172.16.70.50`）上 Cherry Studio 接入也验证通过，设置代理 `NO_PROXY` 包含 `172.16.0.0/16` 避免内网请求走代理。

---

## 第二步：对齐官方 README 采样建议

在正式调优前，先把权重目录里的 `README.md` 读了一遍，发现文档里有几处和官方推荐不一致的地方需要纠正。

### 官方 Best Practices 采样参数

Qwen3.6-27B 的思考模式和非思考模式有完全不同的采样建议：

| 场景 | temperature | top_p | top_k | presence_penalty |
|------|-------------|-------|-------|-----------------|
| 思考 / 一般任务 | **1.0** | 0.95 | 20 | 0.0 |
| 思考 / 编码 WebDev | **0.6** | 0.95 | 20 | 0.0 |
| 非思考（Instruct） | 0.7 | 0.80 | 20 | **1.5** |

旧文档中「思考模式用 temperature=0.7」是错的，官方推荐思考模式用 **1.0**，只有非思考才用 0.7。

### 服务端参数和官方对齐确认

| 官方要求 | 本机配置 | 符合 |
|---------|---------|------|
| vLLM ≥ 0.19.0 | 0.19.1 | ✓ |
| `--reasoning-parser qwen3` | 已开 | ✓ |
| `--language-model-only` | 已开 | ✓ |
| 上下文 ≥ 128K（思考能力） | 合同 64K，需上调 | 待优化 |
| MTP `num_speculative_tokens=2` | 待评估 | 待测 |

64K 低于官方建议的 128K 下限——这是第一轮调优的核心问题。

---

## 第一轮调优：确定档位（标准路径自动化）

### 扫描范围

用自动化脚本（`run_standard_path.sh`）跑完 12 个阶段，覆盖：

- 64K → 128K → 200K → 256K 多个上下文档位
- prefix caching 开/关
- MTP（Multi-Token Prediction）开/关
- KV cache FP8 开/关

### 关键实测数据

**64K 基线（短生成）：**

| 配置 | 短请求耗时 | 512 token 约 tok/s |
|------|-----------|-------------------|
| 纯基线 | 0.58s | ~48 |
| +prefix cache | ~0.8s | ~47 |
| +prefix+MTP | — | ~40（128K 档） |

**128K 长 prefill（6 万字文档，约 24K prompt tokens）：**

| 配置 | 总耗时 |
|------|-------|
| BF16 KV | 7.9s |
| BF16 KV + prefix cache | 7.9s |
| FP8 KV + MTP（T7b） | **7.4s** |

### 第一轮结论

| 诉求 | 结论 |
|------|------|
| 上下文 | **128K 是均衡点**，满足官方思考建议，200K/256K 按需切换 |
| 速度 | 128K + FP8 KV + prefix + MTP 组合（T7b）综合最优 |
| 智力 | 权重保持 BF16；128K 明显优于 64K；KV FP8 不影响权重精度 |

**第一轮主力配置（T7b）：**

```
--max-model-len 131072
--max-num-seqs 2
--gpu-memory-utilization 0.92
--enable-prefix-caching
--kv-cache-dtype fp8
--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

---

## 第二轮调优：128K 细粒度矩阵

T7b 确定之后，疑问还剩几个：FP8 KV 和 BF16 KV 差多少？MTP 真的有用吗？最优并发数是多少？

### 矩阵设计

14 个变体，单因子扫描，每次相对基准（R2-A0 = T7b）只改一个参数：

| 组 | 变体 | 变更内容 |
|----|------|---------|
| A | A0 | 基准（T7b） |
| A | A1 | 去掉 FP8 → BF16 KV |
| A | A2 | 关闭 MTP |
| A | A3 | BF16 KV + 关 MTP |
| A | A4 | MTP num=4 |
| A | A5 | 关闭 prefix cache |
| B | B1～B3 | util=0.88/0.94，seqs=1 |
| B | B4～B5 | seqs=3/4 |
| C | C1～C3 | chunked-prefill，batched-tokens |

### 基准测试套件

每个变体跑 6 项基准（B1～B6），其中 B6 是 2048 token 长输出，专门用于判断 MTP 效果（512 token 短测不足以体现 MTP 加速）：

| 测试 | 内容 | 主要指标 |
|------|------|---------|
| B1 | 短请求（`你好`，max=32） | 端到端耗时 |
| B2 | 512 token 生成 | tokens/s（官方思考采样） |
| B3 | 6 万字长文 prefill | 总耗时 |
| B4 | 长 system × 2 次请求 | prefix hit 降幅 |
| B5 | `nvidia-smi` | 显存余量 |
| B6 | 2048 token 生成 | tokens/s（MTP 指标） |

全部 14 个变体约 **80 分钟**跑完（tmux 挂机），`summarize_r2.py` 自动生成 `summary.csv`。

### 第二轮完整数据

| 变体 | B2 tok/s | B3 耗时(s) | B6 tok/s | prefix 降幅 | 显存余量(MiB) |
|------|----------|-----------|----------|------------|--------------|
| **A0 基准** | 70.95 | 7.41 | 73.08 | 39.6% | 5476 |
| **A1 BF16 KV** | **85.01** | **6.96** | — | **52.4%** | 6390 |
| A2 无 MTP | 47.49 | 7.45 | 46.35 | 49.7% | 5848 |
| A3 BF16+无MTP | 47.25 | 6.99 | — | 49.0% | 6242 |
| **A4 MTP×4** | 80.17 | 7.05 | **82.95** | 43.3% | 5426 |
| A5 无 prefix | 75.02 | 6.55 | — | **0.2%** | 5526 |
| B1 util=0.88 | 70.94 | 7.22 | — | 40.0% | **8706** |
| B2 util=0.94 | 72.35 | 7.41 | — | 39.7% | **3844** |
| B3 seqs=1 | 67.78 | 7.29 | — | 40.1% | 5488 |
| B4 seqs=3 | 73.60 | 7.24 | — | 40.2% | 5460 |
| C2 batched=8k | 76.32 | **6.68** | — | 39.2% | 4450 |
| C3 batched=16k | 72.26 | 6.63 | — | 39.3% | **2782** |

### 核心发现

**1. KV 精度：BF16 > FP8**

A1（BF16 KV）相对 A0（FP8 KV）：
- B2 吞吐 **+20%**（85 vs 71 tok/s）
- B3 长文 **-6%**（6.96 vs 7.41s）
- prefix hit 降幅提升至 **52%**
- 显存余量略多（BF16 KV 确实稍大，但 128K 下双卡各有足够空间）

128K 上下文下 BF16 KV 不但能跑，而且更快。原因可能是 FP8 的量化/反量化开销在 A100 上抵消了节省的带宽收益。

**2. MTP：一定要开，但关键是长输出场景**

A2 关 MTP 后 B2 吞吐 **-33%**（47 vs 71 tok/s），B6 也是 46 vs 73 tok/s——差距相当显著。

512 token 测试中 vLLM 日志显示 MTP 平均接受长度约 **2.4～2.7**，接受率 **70%～87%**，说明 MTP 在这个模型上效果相当好，不宜关闭。

A4（MTP num=4）的 B6 达到 **82.95 tok/s**，优于官方 num=2 的 73.08——对于输出超过 2K token 的任务（代码生成、长文写作）可以考虑 num=4。

**3. Prefix Cache：对长 system 很有价值**

A5 关闭 prefix cache 后，B4 降幅从 **~40%** 跌至 **0.2%**（符合预期）。对于有固定长 system prompt 的场景（如 Agent 角色设定、长文档 Q&A），prefix cache 能带来显著的重复请求加速。

**4. 显存利用率：0.92 是甜点**

- util=0.88（B1）：显存余量提升至 8706 MiB，适合不确定 OOM 的场景
- util=0.94（B2）：显存余量只剩 3844 MiB，过于紧张
- 单人使用默认 **0.92** 就够

**5. 并发数：2 是合理默认**

seqs=1（B3）相对基准略慢，seqs=3/4 在单用户场景意义不大。当前单人使用保持 `max_num_seqs=2` 即可。

---

## 最终推荐配置

```bash
docker run -d --name vllm-qwen36-27b \
  --restart unless-stopped \
  --runtime nvidia --gpus all \
  --shm-size 16g --ipc host \
  -p 50600:8000 \
  -v /data/models/Qwen3.6-27B:/models/Qwen3.6-27B:ro \
  vllm/vllm-openai:v0.19.1 \
  --model /models/Qwen3.6-27B \
  --tensor-parallel-size 2 \
  --max-model-len 131072 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.92 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

相对初始 T7b（FP8 KV）：生成吞吐 **+20%**，长文 prefill **-6%**。

### 客户端采样（按官方 README）

```python
# 思考 / 一般任务
{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0}

# 思考 / 编码 WebDev
{"temperature": 0.6, "top_p": 0.95, "top_k": 20}

# 非思考（快速响应）
{"temperature": 0.7, "top_p": 0.80, "top_k": 20, "presence_penalty": 1.5,
 "chat_template_kwargs": {"enable_thinking": false}}
```

---

## 工程经验小结

**自动化调优的价值**

第二轮 14 个变体约 80 分钟跑完，如果手工做至少需要大半天。关键是每个变体用同一套基准脚本，结果可以直接横向对比，不会因测试方法不同引入噪声。

**MTP 的判断不能只看短输出**

一开始 512 token 的短测中 MTP「未体现明显加速」，但 2048 token 的 B6 才是正确的判断场景。思考模型的输出往往比较长，MTP 在长输出下加速效果显著。

**BF16 KV 不只是「更保守」**

很多人以为 FP8 KV 是「攒资源」才用的，BF16 KV 是「豪华配置」。但在 128K + A100 这个场景下，BF16 KV 实测更快，因为 A100 的 HBM 带宽足够，FP8 量化反而带来额外开销。

**prefix cache 的 system prompt 要够长**

第一轮测试中短 system prompt 的 prefix hit rate 是 **0%**，没有任何收益。但把 system 扩到 8K+ 字符后，第二次请求降幅约 **40%～52%**。对于 Agent 场景或者固定角色设定，长 system prompt 是 prefix cache 的前提条件。

**显存余量要留够**

B2（util=0.94）虽然跑通了，但只剩 3844 MiB 显存余量（不到 5%），在长文档或者异常请求时有 OOM 风险。**util=0.92 是比较稳妥的值**，B1（util=0.88）适合对稳定性要求更高的场景。

---

## 结语

从「能跑 64K」到「128K BF16 KV + prefix + MTP×2 生产上线」，整个过程的核心收获：

1. **官方 README 要认真读**：采样参数、上下文建议都有明确指导，不能凭直觉猜
2. **自动化基准矩阵**比手工测试可靠，结果可复现
3. **实测优先**：FP8/BF16 哪个更快、MTP 有没有用，都要跑数据说话，不能靠理论推断
4. **分两轮**：先确定档位，再细粒度扫描参数，不要一上来就全参数搜索

最终服务地址：`http://172.16.167.200:50600/v1`，模型 `/models/Qwen3.6-27B`，OpenAI API 兼容。

---

## 系列与资源

**这是「AI实战落地」系列第 1 篇（推理基座）。**

| 篇目 | 主题 | 状态 |
|------|------|------|
| **第 1 篇** | 双 A100 部署与调优 Qwen3.6-27B（本文） | [掘金](https://juejin.cn/post/7641794682348601363) · [知乎](https://zhuanlan.zhihu.com/p/2040488166298228423) |
| [第 2 篇](02-win11-remote-eval-qwen36.md) | Win11 远程评测与本地/云端边界 | [掘金](https://juejin.cn/post/7642013733049958426) · [知乎](https://zhuanlan.zhihu.com/p/2041134684831211786) |
| 第 3 篇 | 基于 R2-A1 API 的运营 Agent MVP | 计划中 |

**可复现资源**（GitHub [ai-production-lab](https://github.com/ai-production-lab)）：

| 轮次 | 仓库路径 |
|------|---------|
| 生产上线 | `production/qwen36-27b-128k/docker-run.sh` |
| 第一轮 | `experiments/qwen36-27b-a100/round1-standard-path/` |
| 第二轮 | `experiments/qwen36-27b-a100/round2-matrix/`（含 `results/summary.csv`） |

**下一篇预告**：Win11 远程调 A100 上的 R2-A1——性能是否还接近实验室数据？C-Eval 子集实测 80% 与本地/云端边界怎么划？见 [第 2 篇](02-win11-remote-eval-qwen36.md)。

欢迎在评论区交流：你们在 A100 上 128K 场景下，**FP8 KV 和 BF16 KV** 谁更快？MTP 在长输出任务里收益如何？
