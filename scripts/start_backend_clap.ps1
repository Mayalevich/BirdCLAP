# Start FastAPI with fine-tuned CLAP weights (run from repo root).
#
# Default: terminates processes LISTENing on the chosen TCP port (WinError 10048).
# Avoid that: -DontKillListeners
#
# Do NOT use -FreePort (removed). Kill is always on unless opted out.

param(
  [int]$Port = 8000,
  [string]$CheckpointPath = "checkpoints\best.pt",
  [string]$ListenHost = "127.0.0.1",
  [switch]$DontKillListeners
)

$ErrorActionPreference = "Stop"

$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

if (-not (Test-Path $CheckpointPath)) {
  Write-Error "Checkpoint not found: $CheckpointPath (cwd: $Root)"
}

function Get-ListenerProcessIds([int]$LocalPort) {
  $ids = New-Object System.Collections.Generic.HashSet[int]
  foreach ($conn in @(Get-NetTCPConnection -LocalPort $LocalPort -State Listen -ErrorAction SilentlyContinue)) {
    $oid = [int]$conn.OwningProcess
    if ($oid -gt 0) { [void]$ids.Add($oid) }
  }
  # Fallback: English netstat "TCP ... :PORT ... LISTENING pid"
  if ($ids.Count -eq 0) {
    foreach ($ln in @(netstat -ano)) {
      if ($ln -notmatch "LISTENING") { continue }
      if ($ln -notmatch ":${LocalPort}\s") { continue }
      $parts = ($ln -split "\s+", [System.StringSplitOptions]::RemoveEmptyEntries)
      $maybePid = [int]$parts[-1]
      if ($maybePid -gt 0) { [void]$ids.Add($maybePid) }
    }
  }
  return @($ids)
}

if (-not $DontKillListeners) {
  try {
    $killList = @(Get-ListenerProcessIds $Port | Sort-Object -Unique)
    foreach ($processId in $killList) {
      $proc = Get-Process -Id $processId -ErrorAction SilentlyContinue
      $nm = if ($proc) { $proc.ProcessName } else { "?" }
      Write-Host ("Port $Port in use - stopping PID $processId ($nm)")
      Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    }
    if ($killList.Count -gt 0) {
      Start-Sleep -Milliseconds 800
    }
  }
  catch {
    Write-Host ("Could not free port ${Port}: $($_.Exception.Message)") -ForegroundColor Yellow
  }
}

$env:MODEL_PROVIDER = "clap"
$env:CHECKPOINT_PATH = (Resolve-Path $CheckpointPath).Path
Remove-Item Env:CORS_ORIGIN -ErrorAction SilentlyContinue
Remove-Item Env:CORS_ORIGINS -ErrorAction SilentlyContinue

Write-Host "MODEL_PROVIDER=clap"
Write-Host "CHECKPOINT_PATH=$($env:CHECKPOINT_PATH)"
Write-Host ("Listen: http://${ListenHost}:$Port/  (set web/.env VITE_API_BASE_URL=http://localhost:$Port)")

if ($ListenHost -ne "127.0.0.1" -and $ListenHost -ne "localhost") {
  Write-Host "Note: CORS_ORIGIN is http://localhost:5173; change env if SPA uses another origin." -ForegroundColor Yellow
}

Write-Host ""

& uvicorn backend.app:app --host $ListenHost --port $Port
