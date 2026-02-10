# features/progress/ Module

## Purpose

Manages user-local state for each text sequence: tracking status, scores, and attempt history. This is the user's personal data, not the dataset content.

## Folder Structure

```
progress/
├── domain/
│   ├── text_sequence_progress.dart    # Entity
│   ├── score_attempt.dart             # Entity
│   └── progress_repository.dart       # Repository interface
└── data/
    └── progress_repository_impl.dart  # Hive implementation
```

---

## Domain Layer

### `text_sequence_progress.dart`

**Purpose**: Represents user's progress on a single text sequence.

**Implementation**:

```dart
import 'package:freezed_annotation/freezed_annotation.dart';
import 'score_attempt.dart';

part 'text_sequence_progress.freezed.dart';
part 'text_sequence_progress.g.dart';

/// User's progress for a single text sequence.
///
/// This is user-local data, persisted in Hive.
@freezed
class TextSequenceProgress with _$TextSequenceProgress {
  const TextSequenceProgress._();

  const factory TextSequenceProgress({
    /// The text sequence this progress belongs to.
    required String textSequenceId,

    /// Whether the user has marked this for practice.
    @Default(false) bool tracked,

    /// Best score achieved (0-100), or null if never attempted.
    double? bestScore,

    /// ID of the attempt that achieved best score.
    String? bestAttemptId,

    /// Timestamp of last attempt, or null if never attempted.
    DateTime? lastAttemptAt,

    /// Total number of attempts.
    @Default(0) int attemptCount,

    /// Last update timestamp.
    required DateTime updatedAt,
  }) = _TextSequenceProgress;

  factory TextSequenceProgress.fromJson(Map<String, dynamic> json) =>
      _$TextSequenceProgressFromJson(json);

  /// Creates a new progress entry for a text sequence.
  factory TextSequenceProgress.initial(String textSequenceId) =>
      TextSequenceProgress(
        textSequenceId: textSequenceId,
        updatedAt: DateTime.now(),
      );

  /// Whether this sequence has ever been attempted.
  bool get hasAttempts => attemptCount > 0;

  /// Hours since last attempt, or infinity if never attempted.
  double get hoursSinceLastAttempt {
    if (lastAttemptAt == null) return double.infinity;
    return DateTime.now().difference(lastAttemptAt!).inMinutes / 60.0;
  }
}
```

**Design Decisions**:
- Stores denormalized `bestScore`, `lastAttemptAt`, `attemptCount` for fast sorting
- Could derive these from attempts list, but direct storage is simpler
- `tracked` is the user's explicit intent to practice this sequence

---

### `score_attempt.dart`

**Purpose**: Records a single practice attempt.

**Implementation**:

```dart
import 'package:freezed_annotation/freezed_annotation.dart';

part 'score_attempt.freezed.dart';
part 'score_attempt.g.dart';

/// A single scoring attempt for a text sequence.
@freezed
class ScoreAttempt with _$ScoreAttempt {
  const factory ScoreAttempt({
    /// Unique identifier for this attempt.
    required String id,

    /// The text sequence that was attempted.
    required String textSequenceId,

    /// When the attempt was graded.
    required DateTime gradedAt,

    /// Overall score (0-100).
    required double score,

    /// Scoring method used (e.g., "asr_cer_v1").
    required String method,

    /// Text recognized by ASR (for debugging).
    String? recognizedText,

    /// Additional metrics (e.g., {"cer": 0.15}).
    Map<String, dynamic>? details,
  }) = _ScoreAttempt;

  factory ScoreAttempt.fromJson(Map<String, dynamic> json) =>
      _$ScoreAttemptFromJson(json);
}
```

**Design Decision**: Attempt stores only the grade, not the recording. Recording is stored separately (one per sequence, overwritten each time).

---

### `progress_repository.dart`

**Purpose**: Interface for progress data operations.

**Implementation**:

```dart
import 'text_sequence_progress.dart';
import 'score_attempt.dart';

/// Repository interface for user progress data.
abstract class ProgressRepository {
  /// Gets progress for a single text sequence.
  /// Returns null if no progress exists yet.
  Future<TextSequenceProgress?> getProgress(String textSequenceId);

  /// Gets progress for multiple sequences at once.
  /// Returns a map; missing entries mean no progress exists.
  Future<Map<String, TextSequenceProgress>> getProgressMap(
    List<String> textSequenceIds,
  );

  /// Gets all text sequence IDs that are tracked.
  Future<List<String>> getTrackedIds();

  /// Gets progress for all tracked sequences.
  Future<List<TextSequenceProgress>> getTrackedProgress();

  /// Toggles the tracked status for a sequence.
  /// Creates progress entry if it doesn't exist.
  Future<void> toggleTracked(String textSequenceId);

  /// Sets tracked status explicitly.
  Future<void> setTracked(String textSequenceId, bool tracked);

  /// Records a new score attempt.
  /// Updates best score and last attempt time automatically.
  Future<void> saveAttempt(ScoreAttempt attempt);

  /// Gets recent attempts for a sequence (most recent first).
  /// Limited to [limit] entries (default 50).
  Future<List<ScoreAttempt>> getAttempts(
    String textSequenceId, {
    int limit = 50,
  });
}
```

---

## Data Layer

### `progress_repository_impl.dart`

**Purpose**: Hive-based implementation of progress storage.

**Implementation**:

```dart
import 'dart:convert';
import 'package:hive/hive.dart';
import 'package:uuid/uuid.dart';
import '../domain/progress_repository.dart';
import '../domain/text_sequence_progress.dart';
import '../domain/score_attempt.dart';

class ProgressRepositoryImpl implements ProgressRepository {
  final Box<Map> _progressBox;
  final Box<List> _attemptsBox;
  final Uuid _uuid;

  static const _maxAttempts = 50;

  ProgressRepositoryImpl(this._progressBox, this._attemptsBox)
      : _uuid = const Uuid();

  @override
  Future<TextSequenceProgress?> getProgress(String textSequenceId) async {
    final data = _progressBox.get(textSequenceId);
    if (data == null) return null;
    return TextSequenceProgress.fromJson(Map<String, dynamic>.from(data));
  }

  @override
  Future<Map<String, TextSequenceProgress>> getProgressMap(
    List<String> textSequenceIds,
  ) async {
    final result = <String, TextSequenceProgress>{};
    for (final id in textSequenceIds) {
      final data = _progressBox.get(id);
      if (data != null) {
        result[id] = TextSequenceProgress.fromJson(
          Map<String, dynamic>.from(data),
        );
      }
    }
    return result;
  }

  @override
  Future<List<String>> getTrackedIds() async {
    final ids = <String>[];
    for (final key in _progressBox.keys) {
      final data = _progressBox.get(key);
      if (data != null) {
        final progress = TextSequenceProgress.fromJson(
          Map<String, dynamic>.from(data),
        );
        if (progress.tracked) {
          ids.add(progress.textSequenceId);
        }
      }
    }
    return ids;
  }

  @override
  Future<List<TextSequenceProgress>> getTrackedProgress() async {
    final result = <TextSequenceProgress>[];
    for (final key in _progressBox.keys) {
      final data = _progressBox.get(key);
      if (data != null) {
        final progress = TextSequenceProgress.fromJson(
          Map<String, dynamic>.from(data),
        );
        if (progress.tracked) {
          result.add(progress);
        }
      }
    }
    return result;
  }

  @override
  Future<void> toggleTracked(String textSequenceId) async {
    final current = await getProgress(textSequenceId);
    final updated = current == null
        ? TextSequenceProgress.initial(textSequenceId).copyWith(tracked: true)
        : current.copyWith(
            tracked: !current.tracked,
            updatedAt: DateTime.now(),
          );
    await _progressBox.put(textSequenceId, updated.toJson());
  }

  @override
  Future<void> setTracked(String textSequenceId, bool tracked) async {
    final current = await getProgress(textSequenceId);
    final updated = current == null
        ? TextSequenceProgress.initial(textSequenceId).copyWith(tracked: tracked)
        : current.copyWith(
            tracked: tracked,
            updatedAt: DateTime.now(),
          );
    await _progressBox.put(textSequenceId, updated.toJson());
  }

  @override
  Future<void> saveAttempt(ScoreAttempt attempt) async {
    // Save the attempt
    final attemptsKey = 'attempts_${attempt.textSequenceId}';
    final existingAttempts = _attemptsBox.get(attemptsKey) ?? [];
    final attemptsList = List<Map>.from(existingAttempts);

    // Add new attempt at the beginning
    attemptsList.insert(0, attempt.toJson());

    // Limit to max attempts
    if (attemptsList.length > _maxAttempts) {
      attemptsList.removeRange(_maxAttempts, attemptsList.length);
    }

    await _attemptsBox.put(attemptsKey, attemptsList);

    // Update progress
    final current = await getProgress(attempt.textSequenceId);
    final now = DateTime.now();

    if (current == null) {
      // Create new progress
      final newProgress = TextSequenceProgress(
        textSequenceId: attempt.textSequenceId,
        tracked: false,
        bestScore: attempt.score,
        bestAttemptId: attempt.id,
        lastAttemptAt: attempt.gradedAt,
        attemptCount: 1,
        updatedAt: now,
      );
      await _progressBox.put(attempt.textSequenceId, newProgress.toJson());
    } else {
      // Update existing progress
      final isBest = current.bestScore == null ||
          attempt.score > current.bestScore!;

      final updated = current.copyWith(
        bestScore: isBest ? attempt.score : current.bestScore,
        bestAttemptId: isBest ? attempt.id : current.bestAttemptId,
        lastAttemptAt: attempt.gradedAt,
        attemptCount: current.attemptCount + 1,
        updatedAt: now,
      );
      await _progressBox.put(attempt.textSequenceId, updated.toJson());
    }
  }

  @override
  Future<List<ScoreAttempt>> getAttempts(
    String textSequenceId, {
    int limit = 50,
  }) async {
    final attemptsKey = 'attempts_$textSequenceId';
    final data = _attemptsBox.get(attemptsKey);
    if (data == null) return [];

    return data
        .take(limit)
        .map((e) => ScoreAttempt.fromJson(Map<String, dynamic>.from(e)))
        .toList();
  }
}
```

**Design Decisions**:
- Two Hive boxes: one for progress (keyed by textSequenceId), one for attempts lists
- Attempts are stored as a list per sequence, limited to 50 most recent
- Best score is updated automatically on each attempt
- No presentation layer in this module (consumed by `practice/` and `text_sequences/`)

---

## Integration Tests

### Progress Repository Test

```dart
void main() {
  late Box<Map> progressBox;
  late Box<List> attemptsBox;
  late ProgressRepository repository;

  setUp(() async {
    await Hive.initFlutter();
    progressBox = await Hive.openBox<Map>('test_progress');
    attemptsBox = await Hive.openBox<List>('test_attempts');
    repository = ProgressRepositoryImpl(progressBox, attemptsBox);
  });

  tearDown(() async {
    await progressBox.clear();
    await attemptsBox.clear();
    await progressBox.close();
    await attemptsBox.close();
  });

  group('ProgressRepository', () {
    test('getProgress returns null for unknown sequence', () async {
      final progress = await repository.getProgress('unknown');
      expect(progress, isNull);
    });

    test('toggleTracked creates progress if not exists', () async {
      await repository.toggleTracked('ts_001');
      final progress = await repository.getProgress('ts_001');

      expect(progress, isNotNull);
      expect(progress!.tracked, isTrue);
    });

    test('toggleTracked toggles existing progress', () async {
      await repository.toggleTracked('ts_001'); // true
      await repository.toggleTracked('ts_001'); // false

      final progress = await repository.getProgress('ts_001');
      expect(progress!.tracked, isFalse);
    });

    test('getTrackedIds returns only tracked sequences', () async {
      await repository.setTracked('ts_001', true);
      await repository.setTracked('ts_002', false);
      await repository.setTracked('ts_003', true);

      final tracked = await repository.getTrackedIds();
      expect(tracked, containsAll(['ts_001', 'ts_003']));
      expect(tracked, isNot(contains('ts_002')));
    });

    test('saveAttempt updates best score if higher', () async {
      final attempt1 = ScoreAttempt(
        id: 'a1',
        textSequenceId: 'ts_001',
        gradedAt: DateTime.now(),
        score: 70.0,
        method: 'asr_cer_v1',
      );
      await repository.saveAttempt(attempt1);

      var progress = await repository.getProgress('ts_001');
      expect(progress!.bestScore, 70.0);

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
      expect(progress.bestAttemptId, 'a2');
    });

    test('saveAttempt does not update best score if lower', () async {
      final attempt1 = ScoreAttempt(
        id: 'a1',
        textSequenceId: 'ts_001',
        gradedAt: DateTime.now(),
        score: 85.0,
        method: 'asr_cer_v1',
      );
      await repository.saveAttempt(attempt1);

      final attempt2 = ScoreAttempt(
        id: 'a2',
        textSequenceId: 'ts_001',
        gradedAt: DateTime.now(),
        score: 60.0,
        method: 'asr_cer_v1',
      );
      await repository.saveAttempt(attempt2);

      final progress = await repository.getProgress('ts_001');
      expect(progress!.bestScore, 85.0);
      expect(progress.bestAttemptId, 'a1');
    });

    test('saveAttempt increments attempt count', () async {
      await repository.saveAttempt(ScoreAttempt(
        id: 'a1',
        textSequenceId: 'ts_001',
        gradedAt: DateTime.now(),
        score: 70.0,
        method: 'asr_cer_v1',
      ));
      await repository.saveAttempt(ScoreAttempt(
        id: 'a2',
        textSequenceId: 'ts_001',
        gradedAt: DateTime.now(),
        score: 75.0,
        method: 'asr_cer_v1',
      ));

      final progress = await repository.getProgress('ts_001');
      expect(progress!.attemptCount, 2);
    });

    test('getAttempts returns attempts in reverse chronological order', () async {
      for (var i = 0; i < 5; i++) {
        await repository.saveAttempt(ScoreAttempt(
          id: 'a$i',
          textSequenceId: 'ts_001',
          gradedAt: DateTime.now().add(Duration(minutes: i)),
          score: 70.0 + i,
          method: 'asr_cer_v1',
        ));
      }

      final attempts = await repository.getAttempts('ts_001');
      expect(attempts.length, 5);
      expect(attempts.first.id, 'a4'); // Most recent
      expect(attempts.last.id, 'a0'); // Oldest
    });

    test('getAttempts limits results', () async {
      for (var i = 0; i < 60; i++) {
        await repository.saveAttempt(ScoreAttempt(
          id: 'a$i',
          textSequenceId: 'ts_001',
          gradedAt: DateTime.now(),
          score: 70.0,
          method: 'asr_cer_v1',
        ));
      }

      final attempts = await repository.getAttempts('ts_001', limit: 10);
      expect(attempts.length, 10);
    });
  });
}
```

### Progress Update Flow Test

```dart
void main() {
  group('Progress update flow', () {
    testWidgets('scoring updates progress and UI reflects change', (tester) async {
      // Setup with mocked repositories
      await tester.pumpWidget(
        ProviderScope(
          overrides: testOverrides,
          child: const SpeakToLearnApp(),
        ),
      );
      await tester.pumpAndSettle();

      // Track a sequence
      await tester.tap(find.byIcon(Icons.list));
      await tester.pumpAndSettle();
      await tester.tap(find.byIcon(Icons.star_border).first);
      await tester.pumpAndSettle();

      // Go back and verify sequence appears
      await tester.tap(find.byIcon(Icons.arrow_back));
      await tester.pumpAndSettle();

      // Verify home shows the tracked sequence
      expect(find.text('我想喝水。'), findsOneWidget);

      // Simulate scoring (would need mock recorder/scorer)
      // ... mock flow here ...

      // Verify score appears in UI
      // expect(find.text('85'), findsOneWidget);
    });
  });
}
```

---

## Notes

### Storage Format

Progress is stored as JSON in Hive:

```json
{
  "textSequenceId": "ts_000001",
  "tracked": true,
  "bestScore": 85.0,
  "bestAttemptId": "a123",
  "lastAttemptAt": "2026-02-06T10:00:00Z",
  "attemptCount": 5,
  "updatedAt": "2026-02-06T10:30:00Z"
}
```

### Why Denormalize?

Storing `bestScore`, `lastAttemptAt`, and `attemptCount` directly instead of computing from attempts:

1. **Performance**: Sorting requires these values; computing from 50 attempts for each of 100+ sequences is slow
2. **Simplicity**: Single read to get all needed data
3. **Tradeoff**: Small storage overhead, must keep in sync on writes

### Attempt History Limits

- Maximum 50 attempts stored per sequence
- Oldest attempts discarded when limit reached
- Sufficient for learning curve analysis
- Prevents unbounded storage growth
