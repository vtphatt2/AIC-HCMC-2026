param(
    [int]$BackendPort = 8002,
    [int]$FrontendPort = 3002,
    # onnx-cpu:  PECore ONNX text encoder, CPU only, no torch install needed (default; matches this machine).
    # torch-cpu: full OpenCLIP model on CPU.
    # torch-cuda: full OpenCLIP model on an NVIDIA GPU.
    # See docs/launch_scripts.md for which one to pick and what to install first.
    [ValidateSet("onnx-cpu", "torch-cpu", "torch-cuda")]
    [string]$Backend = "onnx-cpu"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$backendDir = Join-Path $root "local-client\local-backend"
$frontendDir = Join-Path $root "local-client\frontend"

function Test-Port([int]$Port) {
    return $null -ne (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

$backendEnvByChoice = @{
    "onnx-cpu"   = @{ PECORE_BACKEND = "onnx" }
    "torch-cpu"  = @{ PECORE_BACKEND = "torch"; PECORE_DEVICE = "cpu" }
    "torch-cuda" = @{ PECORE_BACKEND = "torch"; PECORE_DEVICE = "cuda" }
}
$backendEnvPrefix = (($backendEnvByChoice[$Backend]).GetEnumerator() |
    ForEach-Object { "set $($_.Key)=$($_.Value)&&" }) -join " "

# /k (not /c) keeps the console window open after the command exits, so you
# can see uvicorn/npm output live and read any crash output instead of the
# window vanishing.
if (-not (Test-Port $BackendPort)) {
    $origins = "http://localhost:$FrontendPort,http://127.0.0.1:$FrontendPort"
    $backendCommand = "title Backend ($Backend) && $backendEnvPrefix set CORS_ORIGINS=$origins&& `"$(Join-Path $backendDir '.venv\Scripts\python.exe')`" -m uvicorn main:app --host 127.0.0.1 --port $BackendPort"
    Start-Process -FilePath $env:ComSpec -ArgumentList "/d", "/k", $backendCommand -WorkingDirectory $backendDir | Out-Null
    $backendNote = "opened in its own terminal window"
} else {
    Write-Host "Backend already listening on $BackendPort"
    $backendNote = "already running"
}

$apiUrl = "http://127.0.0.1:$BackendPort"
if (-not (Test-Port $FrontendPort)) {
    $frontendCommand = "title Frontend && set NEXT_PUBLIC_API_URL=$apiUrl&& npm.cmd run dev -- -p $FrontendPort"
    Start-Process -FilePath $env:ComSpec -ArgumentList "/d", "/k", $frontendCommand -WorkingDirectory $frontendDir | Out-Null
    $frontendNote = "opened in its own terminal window"
} else {
    Write-Host "Frontend already listening on $FrontendPort"
    $frontendNote = "already running"
}

Write-Host "Backend:  $apiUrl (-Backend $Backend) - $backendNote"
Write-Host "Frontend: http://127.0.0.1:$FrontendPort - $frontendNote"
