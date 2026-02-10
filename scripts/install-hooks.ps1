$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

git -C $repoRoot config core.hooksPath .githooks

Write-Host "Installed Git hooks path: .githooks"
Write-Host "pre-commit hook is now active for this repository."
