$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$PidFile = Join-Path $Root ".archwise-pids.json"

if (-not (Test-Path -LiteralPath $PidFile)) {
    Write-Host "No ArchWise PID file found."
    exit 0
}

$services = Get-Content -LiteralPath $PidFile -Raw | ConvertFrom-Json

function Stop-ProcessTree {
    param([int]$ProcessId)
    $children = Get-CimInstance Win32_Process -Filter "ParentProcessId=$ProcessId" -ErrorAction SilentlyContinue
    foreach ($child in $children) {
        Stop-ProcessTree -ProcessId $child.ProcessId
    }
    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($process) {
        Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
    }
}

foreach ($service in $services) {
    $process = Get-Process -Id $service.pid -ErrorAction SilentlyContinue
    if ($process) {
        Stop-ProcessTree -ProcessId $service.pid
        Write-Host "Stopped $($service.name) on port $($service.port)."
    }
}

Remove-Item -LiteralPath $PidFile -Force
