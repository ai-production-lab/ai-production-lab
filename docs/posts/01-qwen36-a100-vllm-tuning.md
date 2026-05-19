# 双 A100 部署 Qwen3.6-27B：两轮 vLLM 调优，128K 吞吐提升 20% 的完整记录

> **系列**：AI实战落地 · 推理基座（第 1 篇）
> **硬件**：双 NVIDIA A100-SXM4-80GB（NVLink） · **软件**：vLLM v0.19.1 · Qwen3.6-27B BF16

---

要让 AI 真正在行业里跑起来，第一步不是选模型，而是把**推理基座**搭稳。这篇记录从裸机到 **128K 上下文可用**的完整调优过程——当前服务于公路运营 Agent 的推理后端，但部署与调优方法本身适用于任何需要长上下文、思考模式推理的场景。

**本篇你能带走什么：**

- 一套可复现的 vLLM 部署与两轮自动化调优流程
- 14 组配置变体的 benchmark 数据与最终推荐配置（相对第一轮主力配置，**512 token 生成吞吐 +20%**）
- 官方 README 采样参数对齐、MTP/prefix/KV 的实测结论

---

## 背景

拿到一台双 A100 服务器，打算跑 Qwen3.6-27B 作为内部主力大模型。它是阿里开源的 36B MoE 架构模型（27B 激活参数），支持推理链和非推理两种模式，官方推荐上下文至少 128K 才能充分发挥思考能力。

三个核心诉求：

- **上下文够长**：能处理长文档、多轮对话
- **生成快**：实际使用体感好，不干等
- **智力有保障**：权重保持 BF16，不降精度

整个过程分**两轮调优**：

- **第一轮（standard_path）**：初始 64K 配置验收通过后，自动化跑 12 个阶段，确定 **128K 主力档位** 与第一轮推荐组合（128K + FP8 KV + prefix + MTP×2）。
- **第二轮（R2 矩阵）**：在第一轮主力配置基础上做单因子扫描（14 个配置变体），最终推荐 **R2-A1**（BF16 KV + prefix + MTP×2）。

---

## 第一步：部署与验收

### 环境准备

服务器（`<HOST>`，内网示例）上提前同步好 52 GB 的 BF16 权重至 `/data/models/Qwen3.6-27B/`，载入 vLLM 镜像 `vllm/vllm-openai:v0.19.1`。

最初上线时，我们把上下文窗口设为 **64K**（低于官方建议，但便于先跑通服务）：

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
- `--reasoning-parser qwen3`：正确分离 Qwen3 思考链（think 块）与最终回答

冷启动至 API 可用约需 **4～5 分钟**。

### 部署验收

三项验收全部通过：

1. `GET /v1/models` 返回 200，模型 ID `/models/Qwen3.6-27B`，`max_model_len=65536`
2. 短对话（`你好`）正常响应
3. 64K token 输入请求正常完成

Win11 工作站上 Cherry Studio 接入也验证通过；若本机有 HTTP 代理，请将内网段加入 `NO_PROXY`，避免内网请求走代理。

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
| 上下文 ≥ 128K（思考能力） | 初始 64K，需上调 | 待优化 |
| MTP `num_speculative_tokens=2` | 待评估 | 待测 |

64K 低于官方建议的 128K 下限——这是第一轮要把上下文档位往上调的核心原因。

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
| 128K + FP8 KV + prefix + MTP×2（第一轮推荐） | **7.4s** |

### 第一轮结论

| 诉求 | 结论 |
|------|------|
| 上下文 | **128K 是均衡点**，满足官方思考建议，200K/256K 按需切换 |
| 速度 | 128K + FP8 KV + prefix + MTP×2 综合最优 |
| 智力 | 权重保持 BF16；128K 明显优于 64K；KV FP8 不影响权重精度 |

**第一轮主力配置**（128K + FP8 KV + prefix cache + MTP×2）——自动化脚本收敛出的日常默认；第二轮基准 **R2-A0** 即在同一参数上复测对照：

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

第一轮主力配置确定之后，疑问还剩几个：FP8 KV 和 BF16 KV 差多少？MTP 真的有用吗？最优并发数是多少？

### 命名说明（读数据表前先看）

第二轮有两套独立编号，**不要混读**：

| 类型 | 编号示例 | 含义 |
|------|---------|------|
| **配置变体** | R2-A1、R2-B3 | 一组 docker 启动参数（每次相对基准只改一项） |
| **基准测试** | B1～B6 | 每个变体都跑的统一测试项（脚本 `benchmark_r2.py`） |

典型易混点：**变体 R2-B3** = `max-num-seqs=1`；**基准 B3** = 6 万字长文 prefill 测试——编号相同、含义不同。下文表格第一列写「配置变体」，测试指标列写自然语言；括号内保留脚本字段名供复现。

### 矩阵设计

14 个配置变体，单因子扫描，每次相对基准（**R2-A0** = 第一轮主力配置复测，参数不变）只改一个参数：

| 组 | 配置变体 | 变更内容 |
|----|---------|---------|
| A | R2-A0 | 基准（第一轮主力配置复测） |
| A | R2-A1 | 去掉 FP8 → BF16 KV |
| A | R2-A2 | 关闭 MTP |
| A | R2-A3 | BF16 KV + 关 MTP（组合探针） |
| A | R2-A4 | MTP num=4 |
| A | R2-A5 | 关闭 prefix cache |
| B | R2-B1 / R2-B2 / R2-B3 | util=0.88 / 0.94 / seqs=1 |
| B | R2-B4 / R2-B5 | seqs=3 / 4 |
| C | R2-C1～C3 | chunked-prefill，batched-tokens |

### 基准测试套件

每个配置变体至少跑 B1～B5；**B6（2048 token 长生成）** 是 MTP 专项测试——512 token 短测不足以体现 MTP 加速。

**为何 B6 只在 R2-A0 / R2-A2 / R2-A4 执行？**

B6 单次约 **30～45 秒**，全 14 变体都跑会明显拉长矩阵时间。我们只在 **MTP 配置本身发生变化** 的三组上跑 B6，用最小对照集回答「要不要开 MTP、num=2 还是 4」：

| 配置变体 | MTP 设置 | 跑 B6 的原因 |
|---------|---------|-------------|
| R2-A0 | 官方 MTP×2（基准） | 长输出下 MTP 效果的参照组 |
| R2-A2 | 关闭 MTP | 与 A0 对照，量化关 MTP 在长输出上的损失 |
| R2-A4 | MTP×4 | 与 A0 对照，看更长输出能否从 num=4 受益 |

其余变体**不改 MTP 开关或 token 数**（如 R2-A1 只改 KV 精度、B/C 组只改 util/并发/batch），或关 MTP 后短输出已与 A2 同量级（R2-A3），再跑 B6 重复验证意义不大。脚本 `run_round2_variant.sh` 对 A0/A2/A4 自动附加 `--b6`。

| 基准 | 测试内容 | 主要指标 |
|------|---------|---------|
| B1 | 短请求（`你好`，max=32） | 端到端耗时 |
| B2 | 512 token 生成 | tokens/s（官方思考采样） |
| B3 | 6 万字长文 prefill | 总耗时 |
| B4 | 长 system × 2 次请求 | prefix hit 降幅 |
| B5 | `nvidia-smi` | 显存余量 |
| B6 | 2048 token 生成 | tokens/s（MTP 专项） |

全部 14 个变体约 **80 分钟**跑完（tmux 挂机），`summarize_r2.py` 自动生成 `summary.csv`。

### 第二轮完整数据

| 配置变体 | 512token生成 tok/s (B2) | 长文prefill s (B3) | 2048token生成 tok/s (B6) | prefix降幅 (B4) | 显存余量 MiB (B5) |
|---------|------------------------|-------------------|-------------------------|----------------|------------------|
| **R2-A0** 第一轮主力复测 | 70.95 | 7.41 | 73.08 | 39.6% | 5476 |
| **R2-A1** BF16 KV ★ | **85.01** | **6.96** | — | **52.4%** | 6390 |
| R2-A2 无 MTP | 47.49 | 7.45 | 46.35 | 49.7% | 5848 |
| R2-A3 BF16+无MTP | 47.25 | 6.99 | — | 49.0% | 6242 |
| R2-A4 MTP×4 | 80.17 | 7.05 | **82.95** | 43.3% | 5426 |
| R2-A5 无 prefix | 75.02 | 6.55 | — | **0.2%** | 5526 |
| R2-B1 util=0.88 | 70.94 | 7.22 | — | 40.0% | **8706** |
| R2-B2 util=0.94 | 72.35 | 7.41 | — | 39.7% | **3844** |
| R2-B3 seqs=1 | 67.78 | 7.29 | — | 40.1% | 5488 |
| R2-B4 seqs=3 | 73.60 | 7.24 | — | 40.2% | 5460 |
| R2-B5 seqs=4 | 70.41 | 7.27 | — | 40.2% | 5454 |
| R2-C1 chunked-prefill | 69.97 | 7.53 | — | 39.9% | 5476 |
| R2-C2 batched=8k | 76.32 | **6.68** | — | 39.2% | 4450 |
| R2-C3 batched=16k | 72.26 | 6.63 | — | 39.3% | **2782** |

> B6 列仅 R2-A0 / R2-A2 / R2-A4 有数据（见上表说明）。★ 为推荐上线配置。

### 核心发现

**1. KV 精度：BF16 > FP8**

**R2-A1**（BF16 KV）相对 **R2-A0**（FP8 KV）：

- 512 token 生成（B2）**+20%**（85 vs 71 tok/s）
- 长文 prefill（B3）**-6%**（6.96 vs 7.41s），优于基准但非全矩阵最快（无 prefix 的 R2-A5、batched 更大的 R2-C2/C3 在长文上更快，分别以牺牲 prefix 或显存余量为代价）
- prefix 降幅（B4）提升至 **52%**
- 显存余量（B5）6390 MiB，128K 下双卡足够

128K 上下文下 BF16 KV 不但能跑，而且更快。原因可能是 FP8 的量化/反量化开销在 A100 上抵消了节省的带宽收益。

**2. MTP：一定要开，但关键是长输出场景**

**R2-A2** 关 MTP 后：512 token 生成（B2）**-33%**，2048 token 生成（B6）**-37%**（47 vs 71、46 vs 73 tok/s）——差距相当显著。

512 token 测试中 vLLM 日志显示 MTP 平均接受长度约 **2.4～2.7**，接受率 **70%～87%**，说明 MTP 在这个模型上效果相当好，不宜关闭。

**R2-A4**（MTP num=4）的 B6 达到 **82.95 tok/s**，优于官方 num=2 的 73.08——对于输出超过 2K token 的任务（代码生成、长文写作）可以考虑 num=4。

**3. Prefix Cache：对长 system 很有价值**

**R2-A5** 关闭 prefix cache 后，B4 降幅从 **~40%** 跌至 **0.2%**（符合预期）。对于有固定长 system prompt 的场景（如 Agent 角色设定、长文档 Q&A），prefix cache 能带来显著的重复请求加速。

**4. 显存利用率：0.92 是甜点**

- **R2-B1**（util=0.88）：显存余量 8706 MiB，适合不确定 OOM 的场景
- **R2-B2**（util=0.94）：显存余量只剩 3844 MiB，过于紧张
- 单人使用默认 **0.92** 就够

**5. 并发数：2 是合理默认**

**R2-B3**（`max-num-seqs=1`）相对基准略慢；**R2-B4/B5**（seqs=3/4）在单用户场景意义不大。当前单人使用保持 `max_num_seqs=2` 即可。

---

## 最终推荐配置

对应配置变体 **R2-A1**（BF16 KV + prefix + MTP×2）：

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

相对第一轮主力配置（R2-A0）：512 token 生成 **+20%**，长文 prefill **-6%**。

> **说明**：以上结论来自吞吐/延迟 benchmark；输出质量尚未做系统化 A/B 抽检，上线前建议按业务题单对 R2-A0 / A1 / A4 做人工对比。

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

一开始 512 token 的短测中 MTP「未体现明显加速」，但 **2048 token 长生成测试（B6）** 才是正确的判断场景。思考模型的输出往往比较长，MTP 在长输出下加速效果显著。

**BF16 KV 不只是「更保守」**

很多人以为 FP8 KV 是「攒资源」才用的，BF16 KV 是「豪华配置」。但在 128K + A100 这个场景下，BF16 KV 实测更快，因为 A100 的 HBM 带宽足够，FP8 量化反而带来额外开销。

**prefix cache 的 system prompt 要够长**

第一轮测试中短 system prompt 的 prefix hit rate 是 **0%**，没有任何收益。但把 system 扩到 8K+ 字符后，第二次请求降幅约 **40%～52%**。对于 Agent 场景或者固定角色设定，长 system prompt 是 prefix cache 的前提条件。

**显存余量要留够**

**R2-B2**（util=0.94）虽然跑通了，但只剩 3844 MiB 显存余量（不到 5%），在长文档或者异常请求时有 OOM 风险。**util=0.92 是比较稳妥的值**，**R2-B1**（util=0.88）适合对稳定性要求更高的场景。

---

## 结语

从「初始 64K 能跑通」到「128K BF16 KV + prefix + MTP×2 生产上线」，整个过程的核心收获：

1. **官方 README 要认真读**：采样参数、上下文建议都有明确指导，不能凭直觉猜
2. **自动化基准矩阵**比手工测试可靠，结果可复现
3. **实测优先**：FP8/BF16 哪个更快、MTP 有没有用，都要跑数据说话，不能靠理论推断
4. **分两轮**：先确定档位，再细粒度扫描参数，不要一上来就全参数搜索

最终服务地址：`http://<HOST>:50600/v1`，模型 `/models/Qwen3.6-27B`，OpenAI API 兼容。

---

## 系列与资源

**这是「AI实战落地」系列第 1 篇（推理基座）。**

| 篇目 | 主题 | 状态 |
|------|------|------|
| **第 1 篇** | 双 A100 部署与调优 Qwen3.6-27B（本文） | 待发布 |
| 第 2 篇 | 基于该推理基座，构建运营 Agent 的 MVP | 计划中 |
| 第 3 篇 | RAG / 长文档场景下的 prefix 与上下文策略 | 待定 |

**可复现资源**（GitHub [ai-production-lab](https://github.com/ai-production-lab/ai-production-lab)）：

| 轮次 | 仓库路径 |
|------|---------|
| 生产上线 | `production/qwen36-27b-128k/docker-run.sh` |
| 第一轮 | `experiments/qwen36-27b-a100/round1-standard-path/` |
| 第二轮 | `experiments/qwen36-27b-a100/round2-matrix/`（含 `results/summary.csv`） |
| 系列正文 | `docs/posts/01-qwen36-a100-vllm-tuning.md` |

**下一篇预告**：推理基座稳定之后，如何把 OpenAI 兼容 API 接到 Agent 框架里，完成公路运营场景的第一个可对话 MVP——包括 system prompt 设计、思考模式开关与人机协同边界。

欢迎在评论区交流：你们在 A100 上 128K 场景下，**FP8 KV 和 BF16 KV** 谁更快？MTP 在长输出任务里收益如何？
