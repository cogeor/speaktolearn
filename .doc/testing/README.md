# Testing Specifications

## Testing Philosophy

Focus on **integration tests** that verify complete user flows rather than unit tests for every function. Unit tests are reserved for complex business logic.

### Test Pyramid

```
         /\
        /  \  E2E (5-10 tests)
       /────\
      /      \  Integration (20-40 tests)
     /────────\
    /          \  Unit (where needed)
   /────────────\
```

---

## Test Categories

### 1. Unit Tests

For complex, isolated logic:
- CER calculation
- Priority ranking algorithm
- Text normalization
- Result type operations

### 2. Integration Tests

For component interactions:
- Repository + datasource
- Controller + repository
- Screen + controller

### 3. End-to-End Tests

For complete user flows:
- Track and practice sequence
- Complete recording and scoring
- Navigate between screens

---

## Flutter Test Structure

```
apps/mobile_flutter/test/
├── unit/
│   ├── core/
│   │   ├── result_test.dart
│   │   └── string_utils_test.dart
│   └── features/
│       ├── selection/
│       │   └── sequence_ranker_test.dart
│       └── scoring/
│           └── cer_calculator_test.dart
│
├── integration/
│   ├── features/
│   │   ├── text_sequences/
│   │   │   └── repository_test.dart
│   │   ├── progress/
│   │   │   └── repository_test.dart
│   │   ├── recording/
│   │   │   └── recorder_test.dart
│   │   └── scoring/
│   │       └── scorer_test.dart
│   └── flows/
│       ├── tracking_flow_test.dart
│       ├── practice_flow_test.dart
│       └── navigation_flow_test.dart
│
├── widget/
│   ├── screens/
│   │   ├── home_screen_test.dart
│   │   └── sequence_list_screen_test.dart
│   └── widgets/
│       └── practice_sheet_test.dart
│
├── e2e/
│   └── full_practice_session_test.dart
│
├── fixtures/
│   ├── test_dataset.json
│   └── test_audio.m4a
│
├── mocks/
│   ├── mock_repositories.dart
│   ├── mock_audio.dart
│   └── mock_providers.dart
│
└── helpers/
    ├── test_app.dart
    └── pump_helpers.dart
```

---

## Critical Integration Tests

### 1. Dataset Loading Flow

**File**: `integration/features/text_sequences/repository_test.dart`

**Scenario**: Load dataset from assets and query sequences

```dart
void main() {
  group('Dataset loading flow', () {
    late TextSequenceRepository repository;

    setUp(() {
      final source = AssetDatasetSource(assetPath: 'test/fixtures/test_dataset.json');
      repository = TextSequenceRepositoryImpl(source);
    });

    test('loads all sequences from dataset', () async {
      final sequences = await repository.getAll();

      expect(sequences, isNotEmpty);
      expect(sequences.first.text, isNotEmpty);
      expect(sequences.first.language, 'zh-CN');
    });

    test('finds sequence by ID', () async {
      final sequence = await repository.getById('ts_000001');

      expect(sequence, isNotNull);
      expect(sequence!.id, 'ts_000001');
    });

    test('filters by tag', () async {
      final sequences = await repository.getByTag('hsk1');

      expect(sequences, isNotEmpty);
      expect(sequences.every((s) => s.tags.contains('hsk1')), isTrue);
    });
  });
}
```

---

### 2. Progress Tracking Flow

**File**: `integration/features/progress/repository_test.dart`

**Scenario**: Track sequences and record attempts

```dart
void main() {
  group('Progress tracking flow', () {
    late Box<Map> progressBox;
    late Box<List> attemptsBox;
    late ProgressRepository repository;

    setUp(() async {
      progressBox = await Hive.openBox<Map>('test_progress_${DateTime.now().millisecondsSinceEpoch}');
      attemptsBox = await Hive.openBox<List>('test_attempts_${DateTime.now().millisecondsSinceEpoch}');
      repository = ProgressRepositoryImpl(progressBox, attemptsBox);
    });

    tearDown(() async {
      await progressBox.deleteFromDisk();
      await attemptsBox.deleteFromDisk();
    });

    test('track sequence creates progress entry', () async {
      await repository.setTracked('ts_001', true);

      final progress = await repository.getProgress('ts_001');

      expect(progress, isNotNull);
      expect(progress!.tracked, isTrue);
    });

    test('save attempt updates best score', () async {
      final attempt1 = ScoreAttempt(
        id: 'a1',
        textSequenceId: 'ts_001',
        gradedAt: DateTime.now(),
        score: 60.0,
        method: 'asr_cer_v1',
      );
      await repository.saveAttempt(attempt1);

      var progress = await repository.getProgress('ts_001');
      expect(progress!.bestScore, 60.0);

      final attempt2 = ScoreAttempt(
        id: 'a2',
        textSequenceId: 'ts_001',
        gradedAt: DateTime.now(),
        score: 85.0,
        method: 'asr_cer_v1',
      );
      await repository.saveAttempt(attempt2);

      progress = await repository.getProgress('ts_001');
      expect(progress!.bestScore, 85.0);
    });

    test('get tracked IDs returns only tracked sequences', () async {
      await repository.setTracked('ts_001', true);
      await repository.setTracked('ts_002', true);
      await repository.setTracked('ts_003', false);

      final tracked = await repository.getTrackedIds();

      expect(tracked, containsAll(['ts_001', 'ts_002']));
      expect(tracked, isNot(contains('ts_003')));
    });

    test('attempts are limited to 50', () async {
      for (var i = 0; i < 60; i++) {
        await repository.saveAttempt(ScoreAttempt(
          id: 'a$i',
          textSequenceId: 'ts_001',
          gradedAt: DateTime.now(),
          score: 70.0,
          method: 'asr_cer_v1',
        ));
      }

      final attempts = await repository.getAttempts('ts_001');
      expect(attempts.length, lessThanOrEqualTo(50));
    });
  });
}
```

---

### 3. Selection Priority Flow

**File**: `integration/features/selection/get_next_tracked_test.dart`

**Scenario**: Next button selects appropriate sequence

```dart
void main() {
  group('Selection priority flow', () {
    test('new sequences have highest priority', () async {
      final sequences = [
        TextSequence(id: '1', text: 'A', language: 'zh'),
        TextSequence(id: '2', text: 'B', language: 'zh'),
      ];
      final progressMap = {
        '1': TextSequenceProgress(
          textSequenceId: '1',
          tracked: true,
          bestScore: 90.0,
          attemptCount: 10,
          updatedAt: DateTime.now(),
        ),
        '2': TextSequenceProgress(
          textSequenceId: '2',
          tracked: true,
          attemptCount: 0, // Never attempted
          updatedAt: DateTime.now(),
        ),
      };

      final ranker = DefaultSequenceRanker(topK: 1);
      final selected = ranker.selectNext(sequences, progressMap);

      expect(selected!.id, '2'); // New sequence prioritized
    });

    test('low score sequences prioritized over high score', () async {
      final now = DateTime.now();
      final sequences = [
        TextSequence(id: '1', text: 'A', language: 'zh'),
        TextSequence(id: '2', text: 'B', language: 'zh'),
      ];
      final progressMap = {
        '1': TextSequenceProgress(
          textSequenceId: '1',
          tracked: true,
          bestScore: 95.0,
          lastAttemptAt: now.subtract(Duration(hours: 24)),
          attemptCount: 5,
          updatedAt: now,
        ),
        '2': TextSequenceProgress(
          textSequenceId: '2',
          tracked: true,
          bestScore: 40.0,
          lastAttemptAt: now.subtract(Duration(hours: 24)),
          attemptCount: 5,
          updatedAt: now,
        ),
      };

      final ranker = DefaultSequenceRanker(topK: 1);
      final selected = ranker.selectNext(sequences, progressMap);

      expect(selected!.id, '2'); // Low score prioritized
    });

    test('excludes current sequence from selection', () async {
      final sequences = [
        TextSequence(id: '1', text: 'A', language: 'zh'),
        TextSequence(id: '2', text: 'B', language: 'zh'),
      ];
      final progressMap = {
        '1': TextSequenceProgress(textSequenceId: '1', tracked: true, updatedAt: DateTime.now()),
        '2': TextSequenceProgress(textSequenceId: '2', tracked: true, updatedAt: DateTime.now()),
      };

      final ranker = DefaultSequenceRanker(topK: 1);
      final selected = ranker.selectNext(sequences, progressMap, excludeId: '1');

      expect(selected!.id, '2');
    });
  });
}
```

---

### 4. Recording Flow

**File**: `integration/features/recording/recorder_test.dart`

**Note**: Requires device/emulator with microphone permission

```dart
void main() {
  group('Recording flow', () {
    late AudioRecorder recorder;
    late RecordingRepository repository;

    setUp(() async {
      recorder = RecordPluginRecorder();
      repository = RecordingRepositoryImpl();
    });

    tearDown(() async {
      await recorder.dispose();
    });

    test('records and saves audio file', () async {
      // Start recording
      final startResult = await recorder.start();
      expect(startResult.isSuccess, isTrue);

      // Wait a bit
      await Future.delayed(Duration(seconds: 1));

      // Stop recording
      final stopResult = await recorder.stop();
      expect(stopResult.isSuccess, isTrue);

      final tempPath = stopResult.valueOrNull!;

      // Save to repository
      final recording = await repository.saveLatest('ts_001', tempPath);

      expect(recording.textSequenceId, 'ts_001');
      expect(await File(recording.filePath).exists(), isTrue);
    });

    test('replaces previous recording', () async {
      // First recording
      await recorder.start();
      await Future.delayed(Duration(milliseconds: 500));
      var result = await recorder.stop();
      final recording1 = await repository.saveLatest('ts_001', result.valueOrNull!);

      // Second recording
      await recorder.start();
      await Future.delayed(Duration(milliseconds: 500));
      result = await recorder.stop();
      final recording2 = await repository.saveLatest('ts_001', result.valueOrNull!);

      // Only latest should exist
      expect(recording2.id, isNot(recording1.id));
      expect(await File(recording1.filePath).exists(), isFalse);
      expect(await File(recording2.filePath).exists(), isTrue);
    });
  });
}
```

---

### 5. Scoring Flow

**File**: `integration/features/scoring/scorer_test.dart`

```dart
void main() {
  group('Scoring flow', () {
    test('perfect match scores 100', () {
      final calculator = CerCalculator();
      final result = calculator.calculate('我想喝水', '我想喝水');

      expect(result.score, 100.0);
    });

    test('one wrong character reduces score', () {
      final calculator = CerCalculator();
      // 喝 (hē) vs 和 (hé) - one substitution
      final result = calculator.calculate('我想喝水', '我想和水');

      expect(result.score, 75.0); // 1/4 = 25% error
    });

    test('normalizes punctuation before scoring', () {
      final calculator = CerCalculator();
      final result = calculator.calculate('我想喝水。', '我想喝水');

      expect(result.score, 100.0); // Punctuation ignored
    });

    test('full scorer pipeline works', () async {
      final mockRecognizer = MockSpeechRecognizer({
        '/path/recording.m4a': '我想喝水',
      });
      final scorer = AsrSimilarityScorer(mockRecognizer);

      final sequence = TextSequence(
        id: 'ts_001',
        text: '我想喝水。',
        language: 'zh-CN',
      );
      final recording = Recording(
        id: 'r1',
        textSequenceId: 'ts_001',
        createdAt: DateTime.now(),
        filePath: '/path/recording.m4a',
      );

      final grade = await scorer.score(sequence, recording);

      expect(grade.overall, 100.0);
      expect(grade.recognizedText, '我想喝水');
      expect(grade.method, 'asr_cer_v1');
    });
  });
}
```

---

### 6. Navigation Flow

**File**: `integration/flows/navigation_flow_test.dart`

```dart
void main() {
  group('Navigation flow', () {
    testWidgets('navigates from home to list and back', (tester) async {
      await tester.pumpWidget(
        ProviderScope(
          overrides: testOverrides,
          child: const SpeakToLearnApp(),
        ),
      );
      await tester.pumpAndSettle();

      // Verify on home
      expect(find.byType(HomeScreen), findsOneWidget);

      // Tap list button
      await tester.tap(find.byIcon(Icons.list));
      await tester.pumpAndSettle();

      // Verify on list
      expect(find.byType(SequenceListScreen), findsOneWidget);

      // Tap back
      await tester.tap(find.byIcon(Icons.arrow_back));
      await tester.pumpAndSettle();

      // Back on home
      expect(find.byType(HomeScreen), findsOneWidget);
    });

    testWidgets('selecting from list loads sequence on home', (tester) async {
      await tester.pumpWidget(
        ProviderScope(
          overrides: testOverrides,
          child: const SpeakToLearnApp(),
        ),
      );
      await tester.pumpAndSettle();

      // Navigate to list
      await tester.tap(find.byIcon(Icons.list));
      await tester.pumpAndSettle();

      // Tap a sequence
      await tester.tap(find.text('谢谢'));
      await tester.pumpAndSettle();

      // Verify back on home with selected sequence
      expect(find.byType(HomeScreen), findsOneWidget);
      expect(find.text('谢谢'), findsOneWidget);
    });
  });
}
```

---

### 7. Full Practice Session E2E

**File**: `e2e/full_practice_session_test.dart`

```dart
void main() {
  group('Full practice session', () {
    testWidgets('complete practice flow: track, practice, score', (tester) async {
      await tester.pumpWidget(
        ProviderScope(
          overrides: testOverridesWithMockedAudio,
          child: const SpeakToLearnApp(),
        ),
      );
      await tester.pumpAndSettle();

      // Initially empty (no tracked sequences)
      expect(find.text('No tracked sequences'), findsOneWidget);

      // Navigate to list
      await tester.tap(find.text('Open list'));
      await tester.pumpAndSettle();

      // Track first sequence
      await tester.tap(find.byIcon(Icons.star_border).first);
      await tester.pumpAndSettle();

      // Verify star changed
      expect(find.byIcon(Icons.star), findsOneWidget);

      // Select the sequence
      await tester.tap(find.byType(ListTile).first);
      await tester.pumpAndSettle();

      // Verify on home with sequence
      expect(find.byType(HomeScreen), findsOneWidget);
      expect(find.text('No tracked sequences'), findsNothing);

      // Tap sequence to open practice sheet
      await tester.tap(find.text('你好'));
      await tester.pumpAndSettle();

      // Verify practice sheet
      expect(find.byType(PracticeSheet), findsOneWidget);
      expect(find.text('Record'), findsOneWidget);

      // Tap record
      await tester.tap(find.text('Record'));
      await tester.pump(Duration(milliseconds: 100));

      // Verify recording state
      expect(find.text('Stop'), findsOneWidget);

      // Stop recording (mocked - will trigger scoring)
      await tester.tap(find.text('Stop'));
      await tester.pumpAndSettle();

      // Verify score displayed
      expect(find.textContaining('Latest'), findsOneWidget);

      // Close sheet
      await tester.tapAt(Offset(0, 100));
      await tester.pumpAndSettle();

      // Verify best score visible on home
      expect(find.textContaining('Best:'), findsOneWidget);

      // Tap Next
      await tester.tap(find.text('Next'));
      await tester.pumpAndSettle();

      // Flow complete!
    });
  });
}
```

---

## Test Fixtures

### `test_dataset.json`

```json
{
  "schema_version": "1.0.0",
  "dataset_id": "test_v1",
  "language": "zh-CN",
  "generated_at": "2026-01-01T00:00:00Z",
  "items": [
    {
      "id": "ts_000001",
      "text": "你好",
      "romanization": "nǐ hǎo",
      "gloss": {"en": "Hello"},
      "tags": ["hsk1", "greeting"],
      "difficulty": 1
    },
    {
      "id": "ts_000002",
      "text": "谢谢",
      "romanization": "xiè xie",
      "gloss": {"en": "Thank you"},
      "tags": ["hsk1"],
      "difficulty": 1
    },
    {
      "id": "ts_000003",
      "text": "我想喝水",
      "romanization": "wǒ xiǎng hē shuǐ",
      "gloss": {"en": "I want to drink water"},
      "tags": ["hsk1", "daily"],
      "difficulty": 1
    }
  ]
}
```

---

## Mocking Strategy

### `mocks/mock_repositories.dart`

```dart
import 'package:mocktail/mocktail.dart';

class MockTextSequenceRepository extends Mock implements TextSequenceRepository {}
class MockProgressRepository extends Mock implements ProgressRepository {}
class MockRecordingRepository extends Mock implements RecordingRepository {}
class MockExampleAudioRepository extends Mock implements ExampleAudioRepository {}

// Pre-configured mocks for common test scenarios
List<Override> get testOverrides => [
  textSequenceRepositoryProvider.overrideWithValue(
    FakeTextSequenceRepository(testDataset),
  ),
  progressRepositoryProvider.overrideWithValue(
    FakeProgressRepository(),
  ),
  // ...
];
```

### `mocks/mock_audio.dart`

```dart
class FakeAudioRecorder implements AudioRecorder {
  bool _isRecording = false;
  String? _path;

  @override
  bool get isRecording => _isRecording;

  @override
  Future<Result<String, RecordingError>> start() async {
    _isRecording = true;
    _path = '/fake/recording.m4a';
    return Success(_path!);
  }

  @override
  Future<Result<String, RecordingError>> stop() async {
    _isRecording = false;
    return Success(_path!);
  }

  @override
  Future<void> cancel() async {
    _isRecording = false;
  }

  @override
  Future<void> dispose() async {}
}

class FakeSpeechRecognizer implements SpeechRecognizer {
  final String recognizedText;

  FakeSpeechRecognizer({this.recognizedText = '你好'});

  @override
  Future<Result<String, RecognitionError>> recognize(
    String audioPath, {
    required String languageCode,
  }) async {
    return Success(recognizedText);
  }

  @override
  Future<bool> isAvailable(String languageCode) async => true;
}
```

---

## Running Tests

### Unit Tests

```bash
cd apps/mobile_flutter
flutter test test/unit/
```

### Integration Tests

```bash
flutter test test/integration/
```

### Widget Tests

```bash
flutter test test/widget/
```

### All Tests

```bash
flutter test
```

### With Coverage

```bash
flutter test --coverage
genhtml coverage/lcov.info -o coverage/html
open coverage/html/index.html
```

---

## Python Tests

```bash
cd tools/sentence_gen
pytest tests/ -v
```

---

## CI Pipeline Recommendations

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  flutter-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: subosito/flutter-action@v2
      - run: flutter pub get
        working-directory: apps/mobile_flutter
      - run: flutter test
        working-directory: apps/mobile_flutter

  python-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -e ".[dev]"
        working-directory: tools/sentence_gen
      - run: pytest tests/ -v
        working-directory: tools/sentence_gen
```
