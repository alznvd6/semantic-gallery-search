$ErrorActionPreference = 'Stop'

$venvRoot = 'E:\semantic-photo-search-venv'
$python = Join-Path $venvRoot 'Scripts\python.exe'
$requirements = Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'requirements.txt'
$drive = Get-PSDrive -Name E -ErrorAction Stop

if ($drive.Free -lt 12GB) {
    throw 'At least 12 GB free on E: is required for the CPU environment.'
}

if (-not (Test-Path $python)) {
    $launcher = Get-Command py -ErrorAction Stop
    & $launcher.Source -3.12 -m venv $venvRoot
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not create the Python 3.12 environment on E:.'
    }
}

& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw 'Could not upgrade pip in the E-drive environment.'
}

& $python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) {
    throw 'Could not install the CPU-only PyTorch packages.'
}

& $python -m pip install -r $requirements
if ($LASTEXITCODE -ne 0) {
    throw 'Could not install the project requirements.'
}

Write-Output "CPU environment ready: $python"
