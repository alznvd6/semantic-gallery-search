$ErrorActionPreference = 'Stop'

$nvidiaSmi = Get-Command nvidia-smi -ErrorAction Stop
$minimumFreeRamGB = 6
$minimumFreeVramMB = 3072
$hfHome = 'E:\semantic-photo-search-models'
$snapshotRoot = Join-Path $hfHome 'hub\models--jinaai--jina-clip-v2\snapshots'

function Get-GpuStatus {
    $gpuRows = & $nvidiaSmi.Source --query-gpu=name,memory.total,memory.free `
        --format=csv,noheader,nounits
    if ($LASTEXITCODE -ne 0 -or -not $gpuRows) {
        throw 'Could not read NVIDIA GPU status with nvidia-smi.'
    }

    $gpuFields = ($gpuRows | Select-Object -First 1) -split ',\s*'
    if ($gpuFields.Count -lt 3) {
        throw "Unexpected nvidia-smi output: $gpuRows"
    }

    [pscustomobject]@{
        Name = $gpuFields[0]
        TotalMemoryMB = [int]$gpuFields[1]
        FreeMemoryMB = [int]$gpuFields[2]
    }
}

function Assert-Resources {
    $os = Get-CimInstance Win32_OperatingSystem
    $freeRamGB = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
    if ($freeRamGB -lt $minimumFreeRamGB) {
        throw "Not starting Jina GPU evaluation: only $freeRamGB GB RAM is free; at least $minimumFreeRamGB GB is required."
    }

    $gpu = Get-GpuStatus
    if ($gpu.FreeMemoryMB -lt $minimumFreeVramMB) {
        throw "Not starting Jina GPU evaluation: only $($gpu.FreeMemoryMB) MB GPU memory is free; close GPU-heavy applications and retry when at least $minimumFreeVramMB MB is free."
    }

    Write-Output "GPU: $($gpu.Name), $($gpu.FreeMemoryMB) MB free of $($gpu.TotalMemoryMB) MB"
}

$runningJina = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and
    $_.CommandLine -match 'evaluate\.py' -and
    $_.CommandLine -match 'jinaai/jina-clip-v2'
}
if ($runningJina) {
    $processIds = ($runningJina | ForEach-Object { $_.ProcessId }) -join ', '
    throw "A Jina evaluation is already running (PID $processIds); wait for it to finish before starting another."
}

Assert-Resources
$env:HF_HOME = $hfHome
$weights = Get-ChildItem $snapshotRoot -Recurse -File -Filter 'model.safetensors' `
    -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $weights) {
    throw "Cached Jina model weights were not found under $snapshotRoot."
}

$condaCommand = Get-Command conda -ErrorAction Stop
if ($env:CONDA_EXE -and (Test-Path $env:CONDA_EXE)) {
    $condaExe = $env:CONDA_EXE
} elseif ($condaCommand.CommandType -eq 'Application') {
    $condaExe = $condaCommand.Source
} else {
    throw 'Could not locate conda.exe to activate the dl_env environment.'
}

$condaHook = & $condaExe shell.powershell hook
if ($LASTEXITCODE -ne 0 -or -not $condaHook) {
    throw 'Could not load the Conda PowerShell hook.'
}

Invoke-Expression ($condaHook -join [Environment]::NewLine)
conda activate dl_env
if ($env:CONDA_DEFAULT_ENV -ne 'dl_env') {
    throw "Could not activate the Conda environment 'dl_env'."
}

$python = (Get-Command python -ErrorAction Stop).Source
& $python -c 'import torch, transformers, faiss, PIL, google.protobuf, requests; version = tuple(int(part) for part in transformers.__version__.split(chr(46))[:2]); assert (4, 57) <= version < (5, 0), transformers.__version__; from transformers.models.clip.modeling_clip import clip_loss; assert torch.cuda.is_available(); print(transformers.__version__, torch.cuda.get_device_name(0))'
if ($LASTEXITCODE -ne 0) {
    throw "Required evaluation packages, Transformers 4.57 through 4.x, the Jina CLIP loss API, or CUDA are unavailable in the Conda environment 'dl_env'."
}

Assert-Resources

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$outputRoot = 'E:\semantic-photo-search-results'
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$output = Join-Path $outputRoot ("jina-gpu-{0}.json" -f $timestamp)

$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:PYTHONUNBUFFERED = '1'
$pythonScript = Join-Path $repoRoot 'evaluate.py'
& $python $pythonScript --device cuda --model-name 'jinaai/jina-clip-v2' `
    --batch-size 1 --output $output
if ($LASTEXITCODE -ne 0) {
    throw "GPU evaluation failed for jinaai/jina-clip-v2 with exit code $LASTEXITCODE."
}

Write-Output "Results saved to $output"
