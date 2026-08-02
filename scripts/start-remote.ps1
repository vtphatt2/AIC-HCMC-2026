param(
    [int]$Port = 8000,
    [switch]$SkipDatabases,
    [switch]$Reload
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$remoteDir = Join-Path $root "remote-server"
$python = Join-Path $remoteDir ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing remote venv. Run: cd remote-server; python -m venv .venv; .venv\Scripts\pip install -r requirements.txt"
}
if (-not (Test-Path -LiteralPath (Join-Path $remoteDir ".env"))) {
    Copy-Item -LiteralPath (Join-Path $remoteDir ".env.example") -Destination (Join-Path $remoteDir ".env")
    Write-Host "Created remote-server\.env from .env.example; edit it for GPU/CORS settings."
}

function Test-Port([int]$TargetPort) {
    return $null -ne (Get-NetTCPConnection -LocalPort $TargetPort -State Listen -ErrorAction SilentlyContinue)
}

function Wait-Port([int]$TargetPort, [int]$TimeoutSeconds = 90) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-Port $TargetPort) { return }
        Start-Sleep -Seconds 1
    }
    throw "Port $TargetPort did not become ready within $TimeoutSeconds seconds."
}

if (-not $SkipDatabases) {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "docker was not found. Install Docker or use -SkipDatabases for external databases."
    }
    Push-Location $remoteDir
    try {
        & docker compose up -d
        if ($LASTEXITCODE -ne 0) { throw "docker compose up failed." }
    } finally {
        Pop-Location
    }
    Wait-Port 19530
    Wait-Port 15432
}

if (Test-Port $Port) {
    throw "Port $Port is already in use."
}

$uvicornArgs = @("-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", $Port)
if ($Reload) { $uvicornArgs += "--reload" }
Write-Host "Remote backend: http://127.0.0.1:$Port (Ctrl+C to stop; Docker databases stay running)"
Push-Location $remoteDir
try {
    & $python @uvicornArgs
} finally {
    Pop-Location
}
