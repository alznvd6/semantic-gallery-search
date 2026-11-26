param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('jina', 'embeddinggemma', 'qwen')]
    [string]$Model
)

$ErrorActionPreference = 'Stop'
$models = @{
    jina = @{
        ModelName = 'jinaai/jina-clip-v2'
        CacheId = 'jinaai--jina-clip-v2'
        MinimumFreeRamGB = 6
    }
    embeddinggemma = @{
        ModelName = 'google/embeddinggemma-2'
        CacheId = 'google--embeddinggemma-2'
        MinimumFreeRamGB = 10
    }
    qwen = @{
        ModelName = 'Qwen/Qwen3-VL-Embedding-2B'
        CacheId = 'Qwen--Qwen3-VL-Embedding-2B'
        MinimumFreeRamGB = 14
    }
}

$spec = $models[$Model]
function Assert-FreeRam {
    $os = Get-CimInstance Win32_OperatingSystem
    $freeRamGB = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
    if ($freeRamGB -lt $spec.MinimumFreeRamGB) {
        $message = (
            "Not starting {0}: {1:N1} GB RAM is free; close applications and retry " +
            "when at least {2} GB is free."
        ) -f $Model, $freeRamGB, $spec.MinimumFreeRamGB
        throw $message
    }
}

Assert-FreeRam

$hfHome = 'E:\semantic-photo-search-models'
$env:HF_HOME = $hfHome
$snapshotRoot = Join-Path $hfHome ("hub\models--{0}\snapshots" -f $spec.CacheId)
$weights = Get-ChildItem $snapshotRoot -Recurse -File -Filter 'model.safetensors' `
    -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $weights) {
    throw "Cached model weights were not found under $snapshotRoot."
}

$venvRoot = 'E:\semantic-photo-search-venv'
$python = Join-Path $venvRoot 'Scripts\python.exe'
if (-not (Test-Path $python)) {
    & (Join-Path $PSScriptRoot 'setup_cpu_env.ps1')
}

& $python -c 'import torch, transformers, sentence_transformers, faiss, PIL, google.protobuf, requests'
if ($LASTEXITCODE -ne 0) {
    & (Join-Path $PSScriptRoot 'setup_cpu_env.ps1')
}

& $python -c 'import torch, transformers, sentence_transformers, faiss, PIL, google.protobuf, requests'
if ($LASTEXITCODE -ne 0) {
    throw 'Required evaluation packages are still unavailable in the E-drive environment.'
}

Assert-FreeRam

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$outputRoot = 'E:\semantic-photo-search-results'
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$output = Join-Path $outputRoot ("{0}-{1}.json" -f $Model, $timestamp)

$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:PYTHONUNBUFFERED = '1'
$pythonScript = Join-Path $repoRoot 'evaluate.py'
& $python $pythonScript --device cpu --model-name $spec.ModelName `
    --batch-size 1 --output $output
if ($LASTEXITCODE -ne 0) {
    throw "Evaluation failed for $($spec.ModelName) with exit code $LASTEXITCODE."
}

Write-Output "Results saved to $output"
