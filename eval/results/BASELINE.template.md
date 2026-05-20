# 生产评测基线（复制本文件为 BASELINE.md 并填写）

| 项目 | 值 |
|------|-----|
| 快照日期 | |
| 配置 | R2-A1 / production/qwen36-27b-128k |
| vLLM 版本 | |
| API | |

## 套系 A：性能

| 指标 | 本次 | R2-A1 参考 | 备注 |
|------|------|-----------|------|
| B1 短请求 (s) | | ~0.49 | deploy benchmark |
| B2 512token (tok/s) | | ~85 | |
| B3 长文 prefill (s) | | ~6.96 | |
| B4 prefix 降幅 (%) | | ~52 | |
| B6 2048token (tok/s) | | A0 ~73 | 可选 |
| EvalScope TTFT (ms) | | — | perf 脚本 |
| EvalScope 输出吞吐 (tok/s) | | — | |

## 套系 B：能力

| 数据集 | 本次准确率 | limit | 备注 |
|--------|-----------|-------|------|
| GSM8K | | | |
| C-Eval | | | |
| CMMLU | | | standard 模式 |

## 人工抽检

- [ ] 5 题题单见 capability/manual_qa_template.md
