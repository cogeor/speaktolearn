# SpeakToLearn

A language learning app focused on pronunciation practice. Users listen to native speaker audio, record their own pronunciation, and receive feedback based on speech recognition accuracy.

## Quick Start

```bash
# 1. Set up Python tool
make setup

# 2. Add your OpenAI API key
echo "OPENAI__API_KEY=sk-your-key" > tools/text_gen/.env

# 3. Generate content (text + audio) and export to Flutter
make generate

# 4. Run the app
make run
```

## Project Structure

```
speaktolearn/
├── apps/
│   └── mobile_flutter/      # Flutter mobile app (iOS/Android)
├── tools/
│   └── text_gen/            # Python CLI for generating learning content
├── shared/
│   └── data_schema/         # JSON schemas for data contracts
└── .doc/                    # Architecture documentation
```

## Prerequisites

### Required Software

| Tool | Version | Purpose |
|------|---------|---------|
| Flutter SDK | 3.8.1+ | Mobile app development |
| Python | 3.11+ | Content generation tool |
| uv | latest | Python package manager |
| Android Studio | latest | Android emulator & SDK |
| Xcode | latest | iOS simulator (macOS only) |

### Installation Links

- **Flutter**: https://docs.flutter.dev/get-started/install
- **Python**: https://www.python.org/downloads/
- **uv**: https://docs.astral.sh/uv/getting-started/installation/
- **Android Studio**: https://developer.android.com/studio

---

## Full Installation Guide

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd speaktolearn
```

### Step 2: Set Up the Flutter App

```bash
# Navigate to the Flutter project
cd apps/mobile_flutter

# Get dependencies
flutter pub get

# Generate code (freezed, json_serializable)
dart run build_runner build --delete-conflicting-outputs

# Verify setup
flutter doctor
```

### Step 3: Set Up the Python Tool

```bash
# Navigate to the Python tool
cd tools/text_gen

# Create virtual environment with uv
uv venv .venv

# Activate the virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (Git Bash/MSYS2):
source .venv/Scripts/activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"
```

### Step 4: Configure API Keys

Create a `.env` file in `tools/text_gen/`:

```bash
cd tools/text_gen
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```env
OPENAI__API_KEY=sk-your-api-key-here
OPENAI__MODEL=gpt-4o
TTS__PROVIDER=openai
```

---

## Running the App

### Android Emulator

#### 1. Start Android Emulator

**Option A: From Android Studio**
1. Open Android Studio
2. Go to **Tools > Device Manager**
3. Click the play button next to your virtual device

**Option B: From Command Line**
```bash
# List available emulators
emulator -list-avds

# Start an emulator (replace with your AVD name)
emulator -avd Pixel_7_API_34
```

#### 2. Verify Device Connection

```bash
flutter devices
```

You should see your emulator listed:
```
sdk gphone64 x86 64 (mobile) • emulator-5554 • android-x64 • Android 14 (API 34)
```

#### 3. Run the App

```bash
cd apps/mobile_flutter
flutter run
```

### iOS Simulator (macOS only)

#### 1. Start iOS Simulator

```bash
# Open Simulator app
open -a Simulator

# Or start a specific device
xcrun simctl boot "iPhone 15 Pro"
```

#### 2. Run the App

```bash
cd apps/mobile_flutter
flutter run
```

### Hot Reload / Restart

While the app is running:
- Press `r` for hot reload (preserves state)
- Press `R` for hot restart (resets state)
- Press `q` to quit

---

## Generating Learning Content

### Generate Text Sequences

```bash
cd tools/text_gen

# Activate virtual environment
source .venv/Scripts/activate  # or .venv\Scripts\Activate.ps1 on Windows

# Generate 50 Chinese sentences (HSK1 level)
text-gen generate --language zh-CN --count 50 --tags hsk1,daily --difficulty 1

# Generate with different difficulty
text-gen generate --language zh-CN --count 20 --tags hsk2 --difficulty 2
```

### Generate Audio (TTS)

```bash
# Generate audio for existing dataset
text-gen audio output/sentences.zh.json --voices female,male
```

### Full Pipeline (Text + Audio)

```bash
text-gen full --language zh-CN --count 50 --tags hsk1
```

### Export to Flutter App

```bash
text-gen export --input output/ --flutter-assets ../apps/mobile_flutter/assets/
```

### Validate Dataset

```bash
text-gen validate output/sentences.zh.json
```

---

## Development Workflow

### Flutter Development

```bash
cd apps/mobile_flutter

# Run in debug mode
flutter run

# Run with specific device
flutter run -d emulator-5554

# Build APK
flutter build apk

# Build iOS
flutter build ios

# Run tests
flutter test

# Analyze code
flutter analyze
```

### Python Development

```bash
cd tools/text_gen

# Run tests
pytest -v

# Type checking
mypy text_gen/

# Linting
ruff check text_gen/

# Format code
ruff format text_gen/
```

---

## Troubleshooting

### Flutter Issues

**"No devices found"**
```bash
# Check Flutter setup
flutter doctor -v

# Ensure emulator is running
flutter devices
```

**Build errors after pulling changes**
```bash
cd apps/mobile_flutter
flutter clean
flutter pub get
dart run build_runner build --delete-conflicting-outputs
```

### Python Issues

**"ModuleNotFoundError"**
```bash
# Ensure virtual environment is activated
source .venv/Scripts/activate

# Reinstall package
uv pip install -e ".[dev]"
```

**"UnicodeDecodeError" on Windows**
- Already fixed in codebase - uses UTF-8 encoding for prompt files

**API Authentication Error**
- Verify `.env` file exists with correct `OPENAI__API_KEY`
- Check API key is valid at https://platform.openai.com/api-keys

---

## Project Features

### Mobile App
- Listen to native speaker audio examples
- Record your own pronunciation
- Get pronunciation feedback via speech recognition
- Track learning progress
- Browse and select practice sentences

### Text Generation Tool
- Generate sentences using GPT-4
- Multiple difficulty levels (HSK1-5)
- Tag-based content filtering
- Pinyin romanization included
- TTS audio generation (OpenAI voices)
- JSON Schema validation

---

## License

[Add your license here]
