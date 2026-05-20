# 套系 B：能力评测（Win11 远程打 A100 vLLM API）
# 用法：.\run_evalscope_capability.ps1 quick
# 推理在 GPU 服务器上执行；本机只发 HTTP 请求、下载数据集、汇总分数

param(
    [ValidateSet("quick", "standard")]
    [string]$Mode = "quick"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$EnvFile = Join-Path $Root "eval\config.env"

if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match '^\s*([^#=]+)=(.*)$') {
            Set-Item -Path "env:$($Matches[1].Trim())" -Value $Matches[2].Trim()
        }
    }
}

$env:NO_PROXY = "172.16.0.0/16"
$BaseUrl = if ($env:VLLM_BASE_URL) { $env:VLLM_BASE_URL } else { "http://172.16.167.200:50600/v1" }
$Model = if ($env:VLLM_MODEL_ID) { $env:VLLM_MODEL_ID } else { "/models/Qwen3.6-27B" }

$Limit = switch ($Mode) {
    "quick" { if ($env:CAP_EVAL_LIMIT_QUICK) { $env:CAP_EVAL_LIMIT_QUICK } else { "20" } }
    "standard" {
        if ($env:CAP_EVAL_LIMIT_STANDARD) { $env:CAP_EVAL_LIMIT_STANDARD } else { "100" }
    }
}

$Out = Join-Path $Root "eval\results\capability\evalscope"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
Set-Location $Out

$Datasets = if ($Mode -eq "standard") { @("gsm8k", "ceval", "cmmlu") } else { @("gsm8k", "ceval") }

Write-Host "[evalscope-capability] mode=$Mode limit=$Limit api=$BaseUrl"
Write-Host "[evalscope-capability] client=Win11, inference=remote GPU server"

evalscope eval `
    --model $Model `
    --api-url $BaseUrl `
    --api-key EMPTY `
    --eval-type openai_api `
    --datasets @Datasets `
    --limit $Limit `
    --generation-config '{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_tokens": 2048}'

Write-Host "[evalscope-capability] done -> $Out"
