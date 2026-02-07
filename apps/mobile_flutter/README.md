# SpeakToLearn

A Flutter mobile app for language learning through pronunciation practice. Users can practice Chinese phrases with native audio examples and receive pronunciation scores.

## Features

- Practice pronunciation with native speaker audio examples
- Record your own pronunciation and get scored
- Track progress on phrases you're learning
- Browse and search phrases by difficulty and topic
- Pure black dark theme optimized for OLED displays

## Requirements

- Flutter SDK ^3.8.1
- Dart SDK ^3.8.1

## Setup

1. Install dependencies:
   ```bash
   flutter pub get
   ```

2. Generate Freezed/JSON code:
   ```bash
   flutter pub run build_runner build --delete-conflicting-outputs
   ```

## Running the App

```bash
flutter run
```

## Running Tests

Run all tests:
```bash
flutter test
```

Run tests with coverage:
```bash
flutter test --coverage
```

## Project Structure

```
lib/
├── app/              # App shell (theme, router, DI)
├── core/             # Core utilities (Result, audio, storage)
└── features/         # Feature modules
    ├── example_audio/  # Native audio playback
    ├── practice/       # Practice screen and sheet
    ├── progress/       # Progress tracking
    ├── recording/      # Audio recording
    ├── scoring/        # Pronunciation scoring
    ├── selection/      # Sequence selection algorithm
    ├── settings/       # App settings
    └── text_sequences/ # Phrase data
```

## Architecture

The app follows a clean architecture pattern:

- **Domain Layer**: Entities and repository interfaces
- **Data Layer**: Repository implementations, DTOs, data sources
- **Presentation Layer**: StateNotifiers, widgets, screens

State management uses Riverpod with StateNotifier pattern.

## Permissions

### Android
- `RECORD_AUDIO` - For pronunciation recording
- `INTERNET` - For remote audio examples

### iOS
- `NSMicrophoneUsageDescription` - For pronunciation recording
- `NSSpeechRecognitionUsageDescription` - For speech recognition
