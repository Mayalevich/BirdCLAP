# One-command demo launcher - starts FastAPI backend + Vite frontend.
# Run from repo root:
#   powershell -ExecutionPolicy Bypass -File start.ps1
#
# Optional flags:
#   -BackendPort 8000        TCP port for FastAPI (default 8000)
#   -CheckpointPath "..."    Path to .pt weights (default checkpoints\best.pt)
#   -RebuildGallery          Delete cached gallery so it rebuilds with current weights
#   -DontKillListeners       Do NOT stop processes listening on BackendPort before starting

param(
  [int]$BackendPort = 8000,
  [string]$CheckpointPath = "checkpoints\best.pt",   # matches config.py default
  [switch]$RebuildGallery,
  [switch]$DontKillListeners
)

$ErrorActionPreference = "Stop"
$Root = Split-Path $MyInvocation.MyCommand.Path -Parent
Set-Location $Root

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Bird Audio Demo Launcher" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# ── Preflight checks ────────────────────────────────────────────────────────

if (-not (Test-Path $CheckpointPath)) {
  Write-Host "[ERROR] Checkpoint not found: $CheckpointPath" -ForegroundColor Red
  Write-Host "        Put your .pt file at checkpoints\best.pt or pass -CheckpointPath" -ForegroundColor Red
  exit 1
}

$webDir = Join-Path $Root "web"
if (-not (Test-Path (Join-Path $webDir "package.json"))) {
  Write-Host "[ERROR] web/package.json not found. Is this the repo root?" -ForegroundColor Red
  exit 1
}

if (-not (Get-Command "uvicorn" -ErrorAction SilentlyContinue)) {
  Write-Host "[ERROR] uvicorn not found on PATH. Activate venv or install: pip install uvicorn" -ForegroundColor Red
  exit 1
}

if (-not (Get-Command "npm" -ErrorAction SilentlyContinue)) {
  Write-Host "[ERROR] npm not found. Install Node 18+." -ForegroundColor Red
  exit 1
}

# ── Optionally rebuild gallery ───────────────────────────────────────────────

if ($RebuildGallery) {
  $galleryPath = Join-Path $Root "data\gallery_embeddings.pt"
  if (Test-Path $galleryPath) {
    Remove-Item $galleryPath -Force
    Write-Host "[gallery] Removed cached gallery - will rebuild on first API call." -ForegroundColor Yellow
  } else {
    Write-Host "[gallery] No cached gallery found (already clean)."
  }
}

# ── Kill existing listeners on BackendPort ────────────────────────────────────

if (-not $DontKillListeners) {
  function Get-PortPids([int]$P) {
    $ids = [System.Collections.Generic.HashSet[int]]::new()
    foreach ($c in @(Get-NetTCPConnection -LocalPort $P -State Listen -ErrorAction SilentlyContinue)) {
      $oid = [int]$c.OwningProcess; if ($oid -gt 0) { $null = $ids.Add($oid) }
    }
    if ($ids.Count -eq 0) {
      foreach ($ln in @(netstat -ano)) {
        if ($ln -notmatch "LISTENING") { continue }
        if ($ln -notmatch ":${P}\s") { continue }
        $parts = $ln.Trim() -split "\s+"
        $n = [int]$parts[-1]; if ($n -gt 0) { $null = $ids.Add($n) }
      }
    }
    return @($ids)
  }
  $toKill = @(Get-PortPids $BackendPort | Sort-Object -Unique)
  foreach ($processId in $toKill) {
    $nm = (Get-Process -Id $processId -ErrorAction SilentlyContinue).ProcessName
    Write-Host "[port] Stopping PID $processId ($nm) on TCP $BackendPort"
    Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
  }
  if ($toKill.Count -gt 0) { Start-Sleep -Milliseconds 800 }
}

# ── Set env vars for backend ─────────────────────────────────────────────────

$env:MODEL_PROVIDER = "clap"
$env:CHECKPOINT_PATH = (Resolve-Path $CheckpointPath).Path
Remove-Item Env:CORS_ORIGIN -ErrorAction SilentlyContinue
Remove-Item Env:CORS_ORIGINS -ErrorAction SilentlyContinue

# ── Install frontend deps if needed ──────────────────────────────────────────

$nodeModules = Join-Path $webDir "node_modules"
if (-not (Test-Path $nodeModules)) {
  Write-Host "[npm] node_modules missing - running npm install..." -ForegroundColor Yellow
  Push-Location $webDir
  npm install --prefer-offline --silent
  Pop-Location
  Write-Host "[npm] Install done."
}

# ── Launch backend in a new window ───────────────────────────────────────────

$ckptFull = $env:CHECKPOINT_PATH
$backendCmd = "Write-Host 'BACKEND: FastAPI on port $BackendPort' -ForegroundColor Cyan; " +
              "`$env:MODEL_PROVIDER='clap'; " +
              "`$env:CHECKPOINT_PATH='$ckptFull'; " +
              "Set-Location '$Root'; " +
              "uvicorn backend.app:app --host 127.0.0.1 --port $BackendPort"

Write-Host "[backend] Starting FastAPI on http://127.0.0.1:$BackendPort ..."
Start-Process powershell -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $backendCmd

Start-Sleep -Milliseconds 800

# ── Launch frontend in a new window ──────────────────────────────────────────

$frontendCmd = "Write-Host 'FRONTEND: Vite dev server' -ForegroundColor Cyan; " +
               "Set-Location '$webDir'; " +
               "npm run dev"

Write-Host "[frontend] Starting Vite dev server..."
Start-Process powershell -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $frontendCmd

# ── Done ─────────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "Both services starting in separate windows." -ForegroundColor Green
Write-Host ""
Write-Host "  Backend:  http://localhost:$BackendPort/health" -ForegroundColor White
Write-Host "  Frontend: http://localhost:5173  (Vite prints the real URL)" -ForegroundColor White
Write-Host ""
Write-Host "CHECKPOINT: $ckptFull" -ForegroundColor DarkGray
Write-Host ""
Write-Host "NOTE: The CLAP model loads on the first search/classify call." -ForegroundColor DarkGray
Write-Host "      Run with -RebuildGallery to regenerate the embedding index after adding new audio." -ForegroundColor DarkGray
