param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8001
)

$ErrorActionPreference = 'Stop'
$repoRoot = $PSScriptRoot

if (-not $env:HF_HOME) {
    $env:HF_HOME = 'E:\semantic-photo-search-models'
}

$snapshotRoot = Join-Path $env:HF_HOME 'hub\models--jinaai--jina-clip-v2\snapshots'
$weights = Get-ChildItem $snapshotRoot -Recurse -File -Filter 'model.safetensors' `
    -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $weights) {
    throw "Cached Jina CLIP v2 weights were not found under $snapshotRoot."
}

$condaCommand = Get-Command conda -ErrorAction Stop
if ($env:CONDA_EXE -and (Test-Path $env:CONDA_EXE)) {
    $condaExe = $env:CONDA_EXE
} elseif ($condaCommand.CommandType -eq 'Application') {
    $condaExe = $condaCommand.Source
} else {
    throw 'Could not locate conda.exe.'
}

$gpuCheck = 'import torch, transformers, faiss, PIL; from transformers.models.clip.modeling_clip import clip_loss; assert torch.cuda.is_available(); print(transformers.__version__, torch.cuda.get_device_name(0))'
& $condaExe run --no-capture-output -n dl_env python -c $gpuCheck
if ($LASTEXITCODE -ne 0) {
    throw "The dl_env environment is missing app dependencies, compatible Transformers, or CUDA."
}

Push-Location $repoRoot
try {
    & $condaExe run --no-capture-output -n dl_env python -m uvicorn `
        app.backend.api:app --reload --host 127.0.0.1 --port $Port
    if ($LASTEXITCODE -ne 0) {
        throw "The Jina app backend exited with code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}
