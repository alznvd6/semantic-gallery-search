$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'run_cpu_model.ps1') -Model qwen
