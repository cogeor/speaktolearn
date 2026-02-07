#!/bin/bash
# Generate learning content and export to Flutter assets

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
TEXT_GEN_DIR="$ROOT_DIR/tools/text_gen"
FLUTTER_ASSETS="$ROOT_DIR/apps/mobile_flutter/assets"

# Defaults
LANGUAGE="${1:-zh-CN}"
COUNT="${2:-50}"
TAGS="${3:-hsk1,daily}"
DIFFICULTY="${4:-1}"

cd "$TEXT_GEN_DIR"

# Activate venv
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

echo "==> Generating $COUNT $LANGUAGE sentences (difficulty $DIFFICULTY, tags: $TAGS)..."
python -m text_gen.cli generate --language "$LANGUAGE" --count "$COUNT" --tags "$TAGS" --difficulty "$DIFFICULTY"

echo ""
echo "==> Generating TTS audio..."
python -m text_gen.cli audio "output/sentences.${LANGUAGE%%-*}.json" --voices female,male

echo ""
echo "==> Exporting to Flutter assets..."
python -m text_gen.cli export --input output/ --flutter-assets "$FLUTTER_ASSETS/"

echo ""
echo "Done! Assets in $FLUTTER_ASSETS/"
