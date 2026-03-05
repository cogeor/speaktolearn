$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

git -C $repoRoot config core.hooksPath .githooks

Write-Host "Installed Git hooks path: .githooks"

# Verify the hook exists
$hookDir = Join-Path $repoRoot ".githooks"
$hookPath = Join-Path $hookDir "pre-commit"
if (-not (Test-Path $hookPath)) {
    Write-Warning "pre-commit hook file not found at $hookPath"
} else {
    Write-Host "pre-commit hook is now active for this repository."
}
