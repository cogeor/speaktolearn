# features/selection/ Module

## Purpose

Encapsulates the logic for selecting "which text sequence comes next". This is pure domain logic combining dataset and progress data to prioritize practice items.

## Folder Structure

```
selection/
└── domain/
    ├── sequence_ranker.dart       # Ranking interface + implementation
    └── get_next_tracked.dart      # Use case
```

**Note**: This module has no `data/` or `presentation/` layers. It's pure domain logic consumed by `practice/` module.

---

## Domain Layer

### `sequence_ranker.dart`

**Purpose**: Ranks text sequences by practice priority.

**Ranking Algorithm**:
```
priority = need + newBonus - recencyPenalty

where:
  need          = 1 - (bestScore / 100)           // Higher if score is low
  newBonus      = 0.35 if attemptCount == 0       // Boost untried items
  recencyPenalty = 0.25 * exp(-hoursSince / 24)   // Penalize recently practiced
```

**Implementation**:

```dart
import 'dart:math';
import '../progress/domain/text_sequence_progress.dart';
import '../text_sequences/domain/text_sequence.dart';

/// Computes priority scores for text sequences.
abstract class SequenceRanker {
  /// Ranks sequences by priority (highest first).
  List<RankedSequence> rank(
    List<TextSequence> sequences,
    Map<String, TextSequenceProgress> progressMap,
  );

  /// Selects the next sequence to practice.
  ///
  /// Returns null if [sequences] is empty.
  /// If [excludeId] is provided, that sequence is excluded from selection.
  TextSequence? selectNext(
    List<TextSequence> sequences,
    Map<String, TextSequenceProgress> progressMap, {
    String? excludeId,
  });
}

/// A sequence with its computed priority score.
class RankedSequence {
  final TextSequence sequence;
  final double priority;
  final TextSequenceProgress? progress;

  const RankedSequence({
    required this.sequence,
    required this.priority,
    this.progress,
  });
}

/// Default implementation of sequence ranking.
class DefaultSequenceRanker implements SequenceRanker {
  final Random _random;

  /// Number of top candidates to randomly select from.
  final int topK;

  DefaultSequenceRanker({
    Random? random,
    this.topK = 10,
  }) : _random = random ?? Random();

  @override
  List<RankedSequence> rank(
    List<TextSequence> sequences,
    Map<String, TextSequenceProgress> progressMap,
  ) {
    final ranked = sequences.map((seq) {
      final progress = progressMap[seq.id];
      final priority = _computePriority(progress);
      return RankedSequence(
        sequence: seq,
        priority: priority,
        progress: progress,
      );
    }).toList();

    // Sort by priority descending
    ranked.sort((a, b) => b.priority.compareTo(a.priority));

    return ranked;
  }

  @override
  TextSequence? selectNext(
    List<TextSequence> sequences,
    Map<String, TextSequenceProgress> progressMap, {
    String? excludeId,
  }) {
    // Filter to tracked only
    final tracked = sequences.where((s) {
      final progress = progressMap[s.id];
      return progress?.tracked == true && s.id != excludeId;
    }).toList();

    if (tracked.isEmpty) return null;

    // Rank them
    final ranked = rank(tracked, progressMap);

    // Pick randomly from top K
    final candidates = ranked.take(topK).toList();
    final index = _random.nextInt(candidates.length);

    return candidates[index].sequence;
  }

  double _computePriority(TextSequenceProgress? progress) {
    if (progress == null) {
      // Never seen, give it new bonus
      return 1.0 + 0.35; // need=1 (worst score) + newBonus
    }

    // Need: higher priority if score is low
    final bestScore = progress.bestScore ?? 0.0;
    final need = 1.0 - (bestScore / 100.0);

    // New bonus: boost sequences never attempted
    final newBonus = progress.attemptCount == 0 ? 0.35 : 0.0;

    // Recency penalty: reduce priority for recently practiced
    final hoursSince = progress.hoursSinceLastAttempt;
    final recencyPenalty = hoursSince.isFinite
        ? 0.25 * exp(-hoursSince / 24.0)
        : 0.0;

    return need + newBonus - recencyPenalty;
  }
}
```

**Design Decisions**:
- Pure function with no IO dependencies
- Injected `Random` allows deterministic testing
- `topK` randomization prevents feeling repetitive
- Returns `RankedSequence` for debugging/UI if needed

---

### `get_next_tracked.dart`

**Purpose**: Use case that orchestrates getting the next sequence.

**Implementation**:

```dart
import '../text_sequences/domain/text_sequence.dart';
import '../text_sequences/domain/text_sequence_repository.dart';
import '../progress/domain/progress_repository.dart';
import 'sequence_ranker.dart';

/// Use case: Get the next tracked text sequence to practice.
class GetNextTrackedSequence {
  final TextSequenceRepository _textSequences;
  final ProgressRepository _progress;
  final SequenceRanker _ranker;

  GetNextTrackedSequence({
    required TextSequenceRepository textSequences,
    required ProgressRepository progress,
    required SequenceRanker ranker,
  })  : _textSequences = textSequences,
        _progress = progress,
        _ranker = ranker;

  /// Returns the next sequence to practice, or null if none tracked.
  ///
  /// [currentId] is excluded from selection to avoid proposing
  /// the same sequence twice in a row.
  Future<TextSequence?> call({String? currentId}) async {
    // Get all tracked IDs
    final trackedIds = await _progress.getTrackedIds();
    if (trackedIds.isEmpty) return null;

    // Get sequences and progress
    final sequences = await _textSequences.getAll();
    final progressMap = await _progress.getProgressMap(trackedIds);

    // Filter to tracked sequences
    final trackedSequences = sequences
        .where((s) => trackedIds.contains(s.id))
        .toList();

    return _ranker.selectNext(
      trackedSequences,
      progressMap,
      excludeId: currentId,
    );
  }
}
```

---

## Riverpod Providers

```dart
import 'package:flutter_riverpod/flutter_riverpod.dart';

final sequenceRankerProvider = Provider<SequenceRanker>((ref) {
  return DefaultSequenceRanker();
});

final getNextTrackedSequenceProvider = Provider<GetNextTrackedSequence>((ref) {
  return GetNextTrackedSequence(
    textSequences: ref.watch(textSequenceRepositoryProvider),
    progress: ref.watch(progressRepositoryProvider),
    ranker: ref.watch(sequenceRankerProvider),
  );
});
```

---

## Integration Tests

### Ranker Unit Tests

```dart
void main() {
  group('DefaultSequenceRanker', () {
    late DefaultSequenceRanker ranker;

    setUp(() {
      // Use seeded random for deterministic tests
      ranker = DefaultSequenceRanker(random: Random(42), topK: 3);
    });

    test('ranks new sequences higher than practiced ones', () {
      final sequences = [
        TextSequence(id: '1', text: 'A', language: 'zh'),
        TextSequence(id: '2', text: 'B', language: 'zh'),
      ];
      final progressMap = {
        '1': TextSequenceProgress(
          textSequenceId: '1',
          tracked: true,
          bestScore: 80.0,
          attemptCount: 5,
          updatedAt: DateTime.now(),
        ),
        // '2' has no progress (new)
      };

      final ranked = ranker.rank(sequences, progressMap);

      expect(ranked[0].sequence.id, '2'); // New item ranked first
      expect(ranked[1].sequence.id, '1'); // Practiced item second
    });

    test('ranks low-score sequences higher than high-score', () {
      final sequences = [
        TextSequence(id: '1', text: 'A', language: 'zh'),
        TextSequence(id: '2', text: 'B', language: 'zh'),
      ];
      final now = DateTime.now();
      final progressMap = {
        '1': TextSequenceProgress(
          textSequenceId: '1',
          tracked: true,
          bestScore: 90.0,
          lastAttemptAt: now.subtract(Duration(hours: 48)),
          attemptCount: 5,
          updatedAt: now,
        ),
        '2': TextSequenceProgress(
          textSequenceId: '2',
          tracked: true,
          bestScore: 40.0,
          lastAttemptAt: now.subtract(Duration(hours: 48)),
          attemptCount: 5,
          updatedAt: now,
        ),
      };

      final ranked = ranker.rank(sequences, progressMap);

      expect(ranked[0].sequence.id, '2'); // Lower score = higher priority
    });

    test('applies recency penalty to recently practiced', () {
      final sequences = [
        TextSequence(id: '1', text: 'A', language: 'zh'),
        TextSequence(id: '2', text: 'B', language: 'zh'),
      ];
      final now = DateTime.now();
      final progressMap = {
        '1': TextSequenceProgress(
          textSequenceId: '1',
          tracked: true,
          bestScore: 50.0,
          lastAttemptAt: now.subtract(Duration(minutes: 30)), // Recent
          attemptCount: 5,
          updatedAt: now,
        ),
        '2': TextSequenceProgress(
          textSequenceId: '2',
          tracked: true,
          bestScore: 50.0,
          lastAttemptAt: now.subtract(Duration(hours: 48)), // Not recent
          attemptCount: 5,
          updatedAt: now,
        ),
      };

      final ranked = ranker.rank(sequences, progressMap);

      // Same score, but '1' was recent so '2' should rank higher
      expect(ranked[0].sequence.id, '2');
    });

    test('selectNext excludes specified ID', () {
      final sequences = [
        TextSequence(id: '1', text: 'A', language: 'zh'),
        TextSequence(id: '2', text: 'B', language: 'zh'),
      ];
      final progressMap = {
        '1': TextSequenceProgress(textSequenceId: '1', tracked: true, updatedAt: DateTime.now()),
        '2': TextSequenceProgress(textSequenceId: '2', tracked: true, updatedAt: DateTime.now()),
      };

      final selected = ranker.selectNext(
        sequences,
        progressMap,
        excludeId: '1',
      );

      expect(selected?.id, '2');
    });

    test('selectNext returns null for empty list', () {
      final selected = ranker.selectNext([], {});
      expect(selected, isNull);
    });

    test('selectNext only considers tracked sequences', () {
      final sequences = [
        TextSequence(id: '1', text: 'A', language: 'zh'),
        TextSequence(id: '2', text: 'B', language: 'zh'),
      ];
      final progressMap = {
        '1': TextSequenceProgress(textSequenceId: '1', tracked: false, updatedAt: DateTime.now()),
        '2': TextSequenceProgress(textSequenceId: '2', tracked: true, updatedAt: DateTime.now()),
      };

      final selected = ranker.selectNext(sequences, progressMap);

      expect(selected?.id, '2'); // Only tracked sequence
    });
  });
}
```

### Use Case Integration Test

```dart
void main() {
  group('GetNextTrackedSequence', () {
    test('returns null when no sequences are tracked', () async {
      final useCase = GetNextTrackedSequence(
        textSequences: MockTextSequenceRepository([
          TextSequence(id: '1', text: 'A', language: 'zh'),
        ]),
        progress: MockProgressRepository({}),
        ranker: DefaultSequenceRanker(),
      );

      final result = await useCase();
      expect(result, isNull);
    });

    test('returns tracked sequence', () async {
      final useCase = GetNextTrackedSequence(
        textSequences: MockTextSequenceRepository([
          TextSequence(id: '1', text: 'A', language: 'zh'),
        ]),
        progress: MockProgressRepository({
          '1': TextSequenceProgress(textSequenceId: '1', tracked: true, updatedAt: DateTime.now()),
        }),
        ranker: DefaultSequenceRanker(),
      );

      final result = await useCase();
      expect(result?.id, '1');
    });

    test('excludes current sequence from selection', () async {
      final useCase = GetNextTrackedSequence(
        textSequences: MockTextSequenceRepository([
          TextSequence(id: '1', text: 'A', language: 'zh'),
          TextSequence(id: '2', text: 'B', language: 'zh'),
        ]),
        progress: MockProgressRepository({
          '1': TextSequenceProgress(textSequenceId: '1', tracked: true, updatedAt: DateTime.now()),
          '2': TextSequenceProgress(textSequenceId: '2', tracked: true, updatedAt: DateTime.now()),
        }),
        ranker: DefaultSequenceRanker(topK: 1), // Deterministic: always pick top
      );

      // When current is '1', should get '2'
      final result = await useCase(currentId: '1');
      // Note: exact result depends on ranking, but '1' should be excluded
      expect(result?.id, isNot('1'));
    });
  });
}
```

---

## Notes

### Why a Separate Module?

Selection logic could live in `progress/` or `practice/`, but separating it:

1. **Single responsibility**: Progress stores data, Selection makes decisions
2. **Testability**: Pure functions with no IO
3. **Configurability**: Easy to swap ranking strategies
4. **Growth**: Selection logic tends to evolve (A/B testing, ML, etc.)

### Tuning the Algorithm

The constants in the priority formula can be tuned:

| Constant | Current | Effect |
|----------|---------|--------|
| `newBonus` | 0.35 | Boost for untried items |
| `recencyPenalty` multiplier | 0.25 | Max penalty for very recent |
| Decay rate | 24 hours | Half-life of recency penalty |
| `topK` | 10 | Randomization pool size |

For production, these could be moved to configuration.

### Alternative Strategies

The interface allows implementing alternative strategies:

- **Spaced repetition**: More sophisticated scheduling (like Anki)
- **Difficulty progression**: Prefer easier items first
- **Tag-based**: Prioritize certain categories
- **Time-of-day**: Different priorities at different times
