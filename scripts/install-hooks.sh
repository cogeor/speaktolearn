#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

chmod +x "$repo_root/.githooks/pre-commit"
git -C "$repo_root" config core.hooksPath .githooks

echo "Installed Git hooks path: .githooks"

hook_path="$repo_root/.githooks/pre-commit"
if [ ! -f "$hook_path" ]; then
    echo "WARNING: pre-commit hook file not found at $hook_path"
else
    echo "pre-commit hook is now active for this repository."
fi
