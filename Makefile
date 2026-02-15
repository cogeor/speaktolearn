.PHONY: setup install-hooks generate clean run test deploy-model help

# Default target
help:
	@echo "SpeakToLearn - Development Commands"
	@echo ""
	@echo "  make setup      - Set up Python virtual environment"
	@echo "  make install-hooks - Configure local git hooks"
	@echo "  make generate   - Generate text + audio and export to Flutter"
	@echo "  make deploy-model - Deploy V4 model to Flutter (auto-detects latest checkpoint)"
	@echo "  make clean      - Remove generated files"
	@echo "  make run        - Run Flutter app"
	@echo "  make test       - Run all tests"
	@echo ""

# Configuration
PYTHON := tools/text_gen/.venv/Scripts/python
TEXT_GEN := $(PYTHON) -m text_gen.cli
FLUTTER_ASSETS := apps/mobile_flutter/assets

# Set up Python environment
setup:
	cd tools/text_gen && uv venv .venv && uv pip install -e ".[dev]"
	@echo ""
	@echo "Setup complete. Add your OpenAI API key to tools/text_gen/.env"

# Configure local git hooks
install-hooks:
	powershell -ExecutionPolicy Bypass -File scripts/install-hooks.ps1

# Generate content and export to Flutter
generate:
	@echo "==> Generating text sequences..."
	cd tools/text_gen && $(TEXT_GEN) generate --language zh-CN --count 50 --tags hsk1,daily --difficulty 1
	@echo ""
	@echo "==> Generating TTS audio..."
	cd tools/text_gen && $(TEXT_GEN) audio output/sentences.zh.json --voices female,male
	@echo ""
	@echo "==> Exporting to Flutter assets..."
	cd tools/text_gen && $(TEXT_GEN) export --input output/ --flutter-assets ../../$(FLUTTER_ASSETS)/
	@echo ""
	@echo "Done! Assets exported to $(FLUTTER_ASSETS)/"

# Clean generated files
clean:
	rm -rf tools/text_gen/output/sentences.*.json
	rm -rf tools/text_gen/output/examples/
	rm -rf $(FLUTTER_ASSETS)/datasets/
	rm -rf $(FLUTTER_ASSETS)/examples/

# Run Flutter app
run:
	cd apps/mobile_flutter && flutter run

# Deploy V4 model to Flutter
deploy-model:
	@echo "==> Deploying V4 model to Flutter..."
	cd tools/mandarin_grader && uv run python scripts/deploy_to_flutter.py
	@echo ""
	@echo "==> Running flutter pub get..."
	cd apps/mobile_flutter && flutter pub get
	@echo ""
	@echo "Done! Model deployed to $(FLUTTER_ASSETS)/models/"

# Run tests
test:
	cd tools/text_gen && $(PYTHON) -m pytest -v
	cd apps/mobile_flutter && flutter test
