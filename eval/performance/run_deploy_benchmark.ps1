# 套系 A-1（Windows 远程）：对 A100 vLLM API 跑 deploy benchmark
# 用法：.\run_deploy_benchmark.ps1
# 注意：B5 显存需在 GPU 服务器上单独 nvidia-smi

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
$ChatUrl = if ($env:VLLM_CHAT_URL) { $env:VLLM_CHAT_URL } else { "http://172.16.167.200:50600/v1/chat/completions" }
$Model = if ($env:VLLM_MODEL_ID) { $env:VLLM_MODEL_ID } else { "/models/Qwen3.6-27B" }
$Label = if ($env:BENCH_LABEL) { "$($env:BENCH_LABEL)-$(Get-Date -Format yyyyMMdd)" } else { "prod-local-$(Get-Date -Format yyyyMMdd)" }
$Out = Join-Path $Root "eval\results\performance\deploy"
$Script = Join-Path $Root "experiments\qwen36-27b-a100\round2-matrix\scripts\benchmark_r2.py"

New-Item -ItemType Directory -Force -Path $Out | Out-Null

Write-Host "[deploy-benchmark] label=$Label url=$ChatUrl"
python $Script `
    --label $Label `
    --url $ChatUrl `
    --model $Model `
    --out-dir $Out `
    --run-b6

Write-Host "[deploy-benchmark] done -> $Out"
