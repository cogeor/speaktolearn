#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

chmod +x "$repo_root/.githooks/pre-commit"
git -C "$repo_root" config core.hooksPath .githooks

echo "Installed Git hooks path: .githooks"
echo "pre-commit hook is now active for this repository."
