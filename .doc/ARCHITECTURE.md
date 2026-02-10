# Architecture Overview

## Design Philosophy

### Guiding Principles

1. **Feature-first organization** - Code grouped by feature, not layer
2. **Clean Architecture** - Dependency inversion, testable domain logic
3. **Simplicity** - No over-engineering, minimal abstractions
4. **Swappable implementations** - Interfaces allow changing providers

### Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│                    PRESENTATION                         │
│         (Screens, Widgets, Controllers/State)           │
│                    Flutter imports OK                   │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                      DOMAIN                             │
│        (Entities, Use Cases, Repository Interfaces)     │
│                  Pure Dart only                         │
└─────────────────────────────────────────────────────────┘
                           ▲
                           │
┌─────────────────────────────────────────────────────────┐
│                       DATA                              │
│    (Repository Impls, Datasources, External APIs)       │
│                Platform/plugin imports                  │
└─────────────────────────────────────────────────────────┘
```

**Key Rule**: Domain never imports from Data or Presentation.

---

## Simplified File Structure

The plans proposed many files. This documentation simplifies to the essential structure:

```
apps/mobile_flutter/lib/
├── main.dart
│
├── app/
│   ├── app.dart                    # MaterialApp setup
│   ├── router.dart                 # GoRouter configuration
│   ├── di.dart                     # Riverpod provider overrides
│   └── theme.dart                  # Dark theme definition
│
├── core/
│   ├── result.dart                 # Result<T, E> type
│   ├── audio/
│   │   └── audio_player.dart       # Playback abstraction
│   ├── storage/
│   │   └── hive_init.dart          # Hive initialization
│   └── utils/
│       └── string_utils.dart       # Text normalization
│
└── features/
    ├── text_sequences/
    │   ├── domain/
    │   │   ├── text_sequence.dart
    │   │   └── text_sequence_repository.dart
    │   ├── data/
    │   │   ├── text_sequence_dto.dart
    │   │   ├── dataset_source.dart
    │   │   └── text_sequence_repository_impl.dart
    │   └── presentation/
    │       ├── sequence_list_screen.dart
    │       └── sequence_list_controller.dart
    │
    ├── progress/
    │   ├── domain/
    │   │   ├── text_sequence_progress.dart
    │   │   ├── score_attempt.dart
    │   │   └── progress_repository.dart
    │   ├── data/
    │   │   └── progress_repository_impl.dart
    │   └── presentation/
    │       └── (none - consumed by other features)
    │
    ├── selection/
    │   └── domain/
    │       ├── sequence_ranker.dart
    │       └── get_next_tracked.dart
    │
    ├── practice/
    │   └── presentation/
    │       ├── home_screen.dart
    │       ├── home_controller.dart
    │       └── practice_sheet.dart
    │
    ├── recording/
    │   ├── domain/
    │   │   ├── recording.dart
    │   │   ├── audio_recorder.dart
    │   │   └── recording_repository.dart
    │   └── data/
    │       ├── record_plugin_recorder.dart
    │       └── recording_repository_impl.dart
    │
    ├── example_audio/
    │   ├── domain/
    │   │   └── example_audio_repository.dart
    │   └── data/
    │       └── example_audio_repository_impl.dart
    │
    ├── scoring/
    │   ├── domain/
    │   │   ├── grade.dart
    │   │   └── pronunciation_scorer.dart
    │   └── data/
    │       ├── speech_recognizer.dart
    │       ├── asr_similarity_scorer.dart
    │       └── cer_calculator.dart
    │
    └── settings/
        ├── domain/
        │   └── app_settings.dart
        └── data/
            └── settings_repository_impl.dart
```

**Simplifications from plans**:
- Removed separate `usecases/` folders - use cases are methods on repositories or simple functions
- Removed `mappers/` - inline mapping in repository implementations
- Removed `models/` folders - DTOs live alongside their datasources
- Combined related files where sensible

---

## Feature Dependencies

```
                    ┌─────────────┐
                    │   settings  │
                    └─────────────┘
                           │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│text_sequences│    │  progress   │    │  selection  │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           ▼
                    ┌─────────────┐
                    │  practice   │◄────┬────────────┐
                    └─────────────┘     │            │
                           │            │            │
              ┌────────────┼────────────┤            │
              ▼            ▼            ▼            │
       ┌───────────┐ ┌───────────┐ ┌─────────┐      │
       │ recording │ │example_aud│ │ scoring │──────┘
       └───────────┘ └───────────┘ └─────────┘
```

**Dependency Rules**:
- `practice` orchestrates all features for the home screen
- `selection` reads from `text_sequences` + `progress`
- `scoring` uses `recording` (needs the audio file)
- No circular dependencies

---

## State Management (Riverpod)

### Provider Structure

```dart
// Domain repositories as abstract providers
final textSequenceRepositoryProvider = Provider<TextSequenceRepository>((ref) {
  throw UnimplementedError('Override in main');
});

// Concrete implementations in di.dart
final textSequenceRepositoryImplProvider = Provider<TextSequenceRepository>((ref) {
  final source = ref.watch(datasetSourceProvider);
  return TextSequenceRepositoryImpl(source);
});

// Controllers as StateNotifier/AsyncNotifier
final homeControllerProvider = StateNotifierProvider<HomeController, HomeState>((ref) {
  return HomeController(
    textSequences: ref.watch(textSequenceRepositoryProvider),
    progress: ref.watch(progressRepositoryProvider),
    selection: ref.watch(sequenceRankerProvider),
  );
});
```

### State Objects (Freezed)

```dart
@freezed
class HomeState with _$HomeState {
  const factory HomeState({
    TextSequence? current,
    TextSequenceProgress? currentProgress,
    @Default(false) bool isLoading,
    @Default(false) bool isEmptyTracked,
  }) = _HomeState;
}
```

---

## Routing (GoRouter)

```dart
final router = GoRouter(
  initialLocation: '/',
  routes: [
    GoRoute(
      path: '/',
      builder: (_, __) => const HomeScreen(),
    ),
    GoRoute(
      path: '/list',
      builder: (_, __) => const SequenceListScreen(),
    ),
  ],
);
```

Only 2 routes. Keep it minimal.

---

## Error Handling Pattern

Use `Result<T, E>` type for operations that can fail:

```dart
sealed class Result<T, E> {
  const Result();
}

class Success<T, E> extends Result<T, E> {
  final T value;
  const Success(this.value);
}

class Failure<T, E> extends Result<T, E> {
  final E error;
  const Failure(this.error);
}
```

**Usage**:
```dart
abstract class SpeechRecognizer {
  Future<Result<String, RecognitionError>> recognize(String audioPath);
}
```

---

## Data Flow Examples

### Loading a Sequence

```
User taps item in list
       │
       ▼
SequenceListController.select(id)
       │
       ▼
HomeController.setCurrentSequence(id)
       │
       ├─► TextSequenceRepository.getById(id)
       │          │
       │          ▼
       │   DatasetSource.load() → parse JSON → find item
       │
       └─► ProgressRepository.getProgress(id)
                  │
                  ▼
           Hive box lookup
       │
       ▼
Update HomeState(current, currentProgress)
       │
       ▼
UI rebuilds with new sentence
```

### Recording and Scoring

```
User presses Record
       │
       ▼
PracticeSheet.startRecording()
       │
       ▼
AudioRecorder.start() → returns temp file path
       │
User presses Stop
       │
       ▼
PracticeSheet.stopAndScore()
       │
       ├─► AudioRecorder.stop()
       │
       ├─► RecordingRepository.saveLatest(textSequenceId, tempPath)
       │          │
       │          ▼
       │   Delete old file, move temp to stable location
       │
       └─► PronunciationScorer.score(textSequence, recording)
                  │
                  ├─► SpeechRecognizer.recognize(audioPath)
                  │          │
                  │          ▼
                  │   speech_to_text plugin → recognized string
                  │
                  └─► CerCalculator.calculate(expected, recognized)
                             │
                             ▼
                      Grade(overall, method, recognizedText)
       │
       ▼
ProgressRepository.saveAttempt(textSequenceId, grade)
       │
       ▼
Update UI with score
```

---

## Testing Strategy

### Test Pyramid

```
         /\
        /  \  E2E (few)
       /────\
      /      \  Integration (moderate)
     /────────\
    /          \  Unit (many)
   /────────────\
```

### What to Test

| Layer | Test Type | Focus |
|-------|-----------|-------|
| Domain entities | Unit | Validation, equality |
| Use cases | Unit | Business logic |
| Repositories | Integration | Data flow, persistence |
| Controllers | Widget | State transitions |
| Screens | Widget | User interactions |
| Full flows | Integration | End-to-end scenarios |

### Mocking Strategy

- Abstract all IO behind interfaces
- Use `mocktail` for mocks
- Use `riverpod` overrides for DI in tests

---

## Platform Considerations

### Android

- Minimum SDK: 21 (Lollipop)
- Microphone permission required
- ProGuard rules for Hive

### iOS

- Minimum iOS: 12.0
- `NSMicrophoneUsageDescription` in Info.plist
- `NSSpeechRecognitionUsageDescription` for ASR

### Audio Formats

| Use Case | Format | Codec |
|----------|--------|-------|
| User recordings | `.m4a` | AAC |
| Example audio | `.opus` in `.ogg` | Opus (12-20kbps) |

---

## Future Extension Points

The architecture supports these future enhancements without major refactoring:

1. **Cloud scoring (Option C)** - Implement new `PronunciationScorer`
2. **Downloaded datasets** - Implement new `DatasetSource`
3. **New languages** - Add new dataset files, configure `settings`
4. **Sync across devices** - Add remote `ProgressRepository`
5. **More voices** - Dataset already supports multiple voices
