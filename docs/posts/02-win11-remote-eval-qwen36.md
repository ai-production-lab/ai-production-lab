# Win11 远程调 Qwen3.6-27B：吞吐几乎不掉，C-Eval 从 2.7% 测到 80%

> **系列**：AI实战落地 · 推理基座（第 2 篇）
> **上篇**：[双 A100 部署与 vLLM 两轮调优](https://juejin.cn/post/7641794682348601363)
> **GitHub**：[ai-production-lab](https://github.com/ai-production-lab/ai-production-lab)
> **路径**：Win11 工作站 → 内网 API → 双 A100 上的 Qwen3.6-27B（R2-A1 生产配置）

---

[第 1 篇](https://juejin.cn/post/7641794682348601363)把 vLLM 在双 A100 上调到 **128K R2-A1** 并上线。本篇回答下一个工程问题：**真实用户路径**（Win11 IDE / Agent 客户端远程调 API）下，性能是否还接近实验室数据？中文能力能否支撑公路运营场景？什么时候该上云端？

**先说一个坑**：同一模型、同一 75 题，只调整 `max_tokens` 和 thinking 开关，C-Eval 子集分数从 **2.7% 升到 80%**——不是模型变聪明，是 **harness 配错了**。

**本篇你能带走什么：**

- 套系 A（性能）：Win11→A100 与 R2-A1 本机 benchmark 的对照表
- 套系 B（能力）：C-Eval 交通相关 5 科实测 **80%**，以及 r1→r3 配置踩坑记录
- 本地 vs 云端旗舰的使用边界决策矩阵
- 可复现脚本入口（[eval/](https://github.com/ai-production-lab/ai-production-lab/tree/main/eval)）

---

## 实验环境

| 项 | 配置 |
|----|------|
| 推理服务 | A100×2，R2-A1（128K + BF16 KV + prefix + MTP×2） |
| API | `http://172.16.167.200:50600/v1`，模型 `/models/Qwen3.6-27B` |
| 客户端 | Win11 工作站，Python + EvalScope |
| 网络 | 内网 HTTP（与 Cherry Studio / Cursor 同路径，非 SSH 隧道） |
| 能力定稿 | r3 方案 B |

上篇部署与调优见 [第 1 篇](https://juejin.cn/post/7641794682348601363)；生产启动脚本见 GitHub [`production/qwen36-27b-128k/docker-run.sh`](https://github.com/ai-production-lab/ai-production-lab/blob/main/production/qwen36-27b-128k/docker-run.sh)。

---

## 为什么要 Win11 远程测

实验室里在 A100 本机跑 benchmark，数字漂亮，但和业务不一致：

| 维度 | A100 本机测 | Win11 远程测（本篇） |
|------|------------|---------------------|
| 网络 | 回环 / 极短 | 内网 HTTP，与 Cherry/Cursor 一致 |
| 代理 | 无 | 需配置 `NO_PROXY`（见下） |
| 客户端 | shell 脚本 | Win11 Python + EvalScope |
| 意义 | 验证 GPU 配置上限 | 验证**交付路径**是否可用 |

Win11 上若系统或 IDE 走代理，内网请求可能被转发导致超时。当前会话可这样设置：

```powershell
# 系统环境变量或 PowerShell 当前会话
$env:NO_PROXY = "172.16.0.0/16,localhost,127.0.0.1"
```

结论先行：**性能几乎不掉队，能力可以测，但 harness 必须配对，否则分数会「假低」。**

---

## 评测设计：两套独立、不要混表

| 套系 | 回答什么 | 工具 | 能否在 Win11 跑 |
|------|---------|------|----------------|
| **A 性能** | 吞吐、prefill、prefix、长生成 | `benchmark_r2.py` + EvalScope perf | ✅（B5 显存除外） |
| **B 能力** | C-Eval 等准确率 | EvalScope eval | ✅（只调 API，推理在 A100） |

GitHub 通用脚本：[eval/performance/](https://github.com/ai-production-lab/ai-production-lab/tree/main/eval/performance)、[eval/capability/](https://github.com/ai-production-lab/ai-production-lab/tree/main/eval/capability)。

---

## 套系 A：性能 — 与 R2-A1 高度接近

| 指标 | Win11 远程 | R2-A1 参考 | 差异 |
|------|-----------|-----------|------|
| B1 短请求 | 0.499 s | ~0.49 s | ≈ |
| **B2 512 token 吞吐** | **82.36 tok/s** | **85.01 tok/s** | **约 −3%** |
| B3 24K prefill | 6.976 s | 6.957 s | ≈ |
| B4 prefix 降幅 | 51.3% | 52.4% | ≈ |
| B6 2048 token 吞吐 | 76.63 tok/s | — | 长生成可用 |
| EvalScope perf（512 tok） | ~82.4 tok/s，TTFT ~140 ms | — | 与 B2 一致 |

**解读：**

- 内网远程调用几乎不损伤 GPU 侧吞吐；Win11 路径可放心作为日常开发入口。
- **prefix cache 51% 降幅**在多轮 Agent / 长 system 场景仍是本地相对云端的核心优势之一。
- B5 显存需在 A100 上补测；Win11 本机 `nvidia-smi` 不代表推理卡。

---

## 套系 B：能力 — C-Eval 从 2.7% 到 80% 的教训

### 测什么

从 C-Eval 52 科中选取与**公路智能运营**相关的 5 科，每科 15 题，共 **75 题**：

| 科目 | 业务关联 |
|------|---------|
| 计算机网络 | 路侧设备、RSU、平台组网 |
| 大学编程 | 算法与系统集成 |
| 高中数学 | 流量统计与定量推理 |
| 法学 | 交通法规与合规 |
| 城乡规划 | 路网与设施布局 |

**说明**：这是**业务向子集**，不是官方全量 52 科 C-Eval，**不能与模型卡 91.4% 直接比数值**。

### 三轮配置的教训

| 轮次 | 配置 | 结果 | 问题 |
|------|------|------|------|
| r1 | thinking 关，`max_tokens=128` | **2.7%** | 72/75 题未输出到 `答案：X` 就被截断 |
| r2 | 分科 thinking，`max_tokens=2048` | **66.7%** | 推理科 thinking 占满 2048，高中数学仅 40% |
| **r3（方案 B，定稿）** | 知识科 thinking **关** + **1024**；推理科 thinking **开** + **4096** | **80.0%** | 可解释、可复现 |

r3 分科成绩：

| 科目 | thinking | max_tokens | 准确率 | 溢出 |
|------|----------|------------|--------|------|
| 计算机网络 | 关 | 1024 | **93.3%** | 1 |
| 大学编程 | 开 | 4096 | **93.3%** | 0 |
| 高中数学 | 开 | 4096 | **80.0%** | 3 |
| 法学 | 关 | 1024 | 73.3% | 2 |
| 城乡规划 | 关 | 1024 | 60.0% | 2 |

共 **8 题溢出、6 题无答案**。城乡规划曾遇 API 连接中断（只完成 6/15 题），补跑合并后凑满 15 题。

### 方案 B 调用示例

知识科与推理科需分开配参，这是 r3 相对 r2 回升的关键：

```python
# 知识科（计算机网络、法学、城乡规划）：thinking 关，max_tokens=1024
payload = {
    "model": "/models/Qwen3.6-27B",
    "messages": [...],
    "max_tokens": 1024,
    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
}

# 推理科（大学编程、高中数学）：thinking 开，max_tokens=4096
payload = {
    "model": "/models/Qwen3.6-27B",
    "messages": [...],
    "max_tokens": 4096,
    "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
}
```

### 80% 怎么理解

- **可用作**：Win11→A100 路径下、交通相关中文选择题的**实测能力锚点**。
- **不可用作**：官方全量 C-Eval 91.4% 的复现声明。
- **量级参考**：子集 80% 与公开榜单上 GPT-4 系 C-Eval 量级（约 80%+）接近，说明**中文知识问答内网可用**；与通义 Max / Claude 全量榜仍有 harness 与题量差异。

---

## 本地 vs 云端：使用边界

结合 r3 与公开 benchmark 参照（非同 harness 复测，仅看量级）：

| 场景 | 建议 | 原因 |
|------|------|------|
| IDE 补全、内网短问答 | **本地 A100** | ~82 tok/s、无排队、无 token 费 |
| 128K 长文 / 固定长 system Agent | **本地 A100** | prefix 降幅 ~51% |
| 中文领域知识题（交通/法规/网络） | **本地 A100** | 子集 C-Eval 80%，编程/网络 93% |
| 复杂 Agent（多工具、仓库级 SWE） | **云端 Claude Opus / Qwen Max** | 官方 SWE 仍高一档 |
| 高峰 burst、弹性扩容 | **云端** | 本地 2 卡并发固定 |
| 数据不出网 | **仅本地** | 合规硬约束 |

**性能上本地稳；能力上中文知识够用；Agent 上限与弹性仍可能需要云端补位。**

---

## 工程经验小结

**1. 能力评测先查 `max_tokens`，再信分数**

128 token 会把 Qwen3.6 测成「2.7% 弱智」——这是 harness 事故，不是模型事故。

**2. thinking 与输出长度要配对**

推理科开 thinking 时，`max_tokens=2048` 不够；提到 4096 后大学编程/高中数学明显回升。

**3. 性能与能力分开存档**

吞吐 82 tok/s 与 C-Eval 80% 互不代表；文档和 Dashboard 不要混一张表。

**4. 远程评测要当生产路径测**

API 502、连接 reset 会在能力长测里放大。脚本需分科跑、支持补跑合并，并对失败题目标记。

**5. 子集分数写进决策，不写进「对标官方」**

对外材料应写：**「交通相关 5 科×15 题，EvalScope 5-shot，80%」**，而非「C-Eval 91.4%」。

---

## 如何复现（摘要）

```powershell
# Win11，clone 后进入 eval 目录
git clone https://github.com/ai-production-lab/ai-production-lab.git
cd ai-production-lab/eval
cp config.env.example config.env   # 改 API 地址

# 套系 A：性能 benchmark
bash performance/run_deploy_benchmark.sh

# 套系 B：C-Eval 能力（需 EvalScope，见 eval/README.md）
bash capability/run_evalscope_capability.sh quick
```

完整说明见 [eval/README.md](https://github.com/ai-production-lab/ai-production-lab/blob/main/eval/README.md)。

---

## 结语

推理基座调优（[第 1 篇](https://juejin.cn/post/7641794682348601363)）解决「A100 上怎么跑快、跑稳」。本篇解决「**Win11 远程用起来是否还靠谱、中文能力是否够场景**」：

1. **性能**：远程路径与 R2-A1 本机 benchmark **差约 3%**，可作生产客户端路径。
2. **能力**：交通向 C-Eval 子集 **80%**（方案 B），网络/编程 **93%**，需正确 harness。
3. **边界**：日常中文知识 + 内网 Agent 优先本地；复杂 Agent 与 burst 考虑云端。

---

## 系列与资源

**「AI实战落地」系列第 2 篇（推理基座 · 评测与边界）。**

| 篇目 | 主题 | 状态 |
|------|------|------|
| [第 1 篇](https://juejin.cn/post/7641794682348601363) | 双 A100 部署与 vLLM 两轮调优 | [掘金](https://juejin.cn/post/7641794682348601363) · [知乎](https://zhuanlan.zhihu.com/p/2040488166298228423) |
| **第 2 篇** | Win11 远程评测与本地/云端边界（本文） | [掘金](https://juejin.cn/post/7642013733049958426) · [知乎](https://zhuanlan.zhihu.com/p/2041134684831211786) |
| 第 3 篇 | 基于 R2-A1 API 的运营 Agent MVP | 计划中 |

**可复现资源**（[ai-production-lab](https://github.com/ai-production-lab/ai-production-lab)）：

| 内容 | 路径 |
|------|------|
| 生产配置 R2-A1 | [production/qwen36-27b-128k/docker-run.sh](https://github.com/ai-production-lab/ai-production-lab/blob/main/production/qwen36-27b-128k/docker-run.sh) |
| 评测脚本（通用） | [eval/](https://github.com/ai-production-lab/ai-production-lab/tree/main/eval) |
| 第 1 篇实验 | [experiments/qwen36-27b-a100/](https://github.com/ai-production-lab/ai-production-lab/tree/main/experiments/qwen36-27b-a100) |

**下一篇预告**：把 OpenAI 兼容 API 接到 Agent 框架，完成公路运营场景第一个可对话 MVP——system prompt、thinking 开关与人机协同边界。

欢迎在评论区交流：你们远程调 vLLM 时，**能力榜和性能榜**是分开测还是混在一起的？
