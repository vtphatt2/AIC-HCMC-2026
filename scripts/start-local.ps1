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
$logDir = Join-Path $root "runtime-logs"
New-Item -ItemType Directory -Force $logDir | Out-Null

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

if (-not (Test-Port $BackendPort)) {
    $origins = "http://localhost:$FrontendPort,http://127.0.0.1:$FrontendPort"
    $backendCommand = "$backendEnvPrefix set CORS_ORIGINS=$origins&& `"$(Join-Path $backendDir '.venv\Scripts\python.exe')`" -m uvicorn main:app --host 127.0.0.1 --port $BackendPort"
    Start-Process -FilePath $env:ComSpec -ArgumentList "/d", "/c", $backendCommand -WorkingDirectory $backendDir -RedirectStandardOutput (Join-Path $logDir "backend.log") -RedirectStandardError (Join-Path $logDir "backend.err.log") -WindowStyle Hidden | Out-Null
} else { Write-Host "Backend already listening on $BackendPort" }

$apiUrl = "http://127.0.0.1:$BackendPort"
if (-not (Test-Port $FrontendPort)) {
    $frontendCommand = "set NEXT_PUBLIC_API_URL=$apiUrl&& npm.cmd run dev -- -p $FrontendPort"
    Start-Process -FilePath $env:ComSpec -ArgumentList "/d", "/c", $frontendCommand -WorkingDirectory $frontendDir -RedirectStandardOutput (Join-Path $logDir "frontend.log") -RedirectStandardError (Join-Path $logDir "frontend.err.log") -WindowStyle Hidden | Out-Null
} else { Write-Host "Frontend already listening on $FrontendPort" }

Write-Host "Backend:  $apiUrl (-Backend $Backend)"
Write-Host "Frontend: http://127.0.0.1:$FrontendPort"
