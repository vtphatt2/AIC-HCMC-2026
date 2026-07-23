param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 3000,
    [string]$LanAddress = "",
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
$backendPython = Join-Path $backendDir ".venv\Scripts\python.exe"

$bindHost = "127.0.0.1"
$publicHost = "127.0.0.1"
if ($LanAddress) {
    $parsedAddress = $null
    if (-not [System.Net.IPAddress]::TryParse($LanAddress, [ref]$parsedAddress) -or
        $parsedAddress.AddressFamily -ne [System.Net.Sockets.AddressFamily]::InterNetwork) {
        throw "-LanAddress must be an IPv4 address, for example 192.168.0.102"
    }
    $bindHost = "0.0.0.0"
    $publicHost = $LanAddress
}

if (-not (Test-Path -LiteralPath $backendPython)) {
    throw "Missing backend venv. Run: cd local-client\local-backend; python -m venv .venv; .venv\Scripts\pip install -r requirements.txt"
}
if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
    throw "npm.cmd was not found. Install Node.js first."
}

function Test-Port([int]$Port) {
    return $null -ne (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

if ($LanAddress -and ((Test-Port $BackendPort) -or (Test-Port $FrontendPort))) {
    throw "LAN mode needs fresh processes. Stop services on ports $BackendPort and $FrontendPort, then run this command again."
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
    $origins = "http://localhost:$FrontendPort,http://127.0.0.1:$FrontendPort,http://${publicHost}:$FrontendPort"
    $backendCommand = "title Backend ($Backend) && $backendEnvPrefix set CORS_ORIGINS=$origins&& `"$backendPython`" -m uvicorn main:app --host $bindHost --port $BackendPort"
    Start-Process -FilePath $env:ComSpec -ArgumentList "/d", "/k", $backendCommand -WorkingDirectory $backendDir | Out-Null
    $backendNote = "opened in its own terminal window"
} else {
    Write-Host "Backend already listening on $BackendPort"
    $backendNote = "already running"
}

$apiUrl = "http://${publicHost}:$BackendPort"
if (-not (Test-Port $FrontendPort)) {
    $frontendCommand = "title Frontend && set NEXT_PUBLIC_API_URL=$apiUrl&& npm.cmd run dev -- -H $bindHost -p $FrontendPort"
    Start-Process -FilePath $env:ComSpec -ArgumentList "/d", "/k", $frontendCommand -WorkingDirectory $frontendDir | Out-Null
    $frontendNote = "opened in its own terminal window"
} else {
    Write-Host "Frontend already listening on $FrontendPort"
    $frontendNote = "already running"
}

Write-Host "Backend:  $apiUrl (-Backend $Backend) - $backendNote"
Write-Host "Frontend: http://${publicHost}:$FrontendPort - $frontendNote"
