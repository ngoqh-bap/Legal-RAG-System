Param(
  [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $VenvPath)) {
  python -m venv $VenvPath
}

& "$VenvPath\\Scripts\\python.exe" -m pip install -U pip
& "$VenvPath\\Scripts\\python.exe" -m pip install -e ".[dev]"

Write-Host "Done. Activate with: $VenvPath\\Scripts\\Activate.ps1"
Write-Host "Try: legal-rag --help"
