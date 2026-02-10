# core/ Module

## Purpose

Shared utilities and abstractions used across all features. Contains no business logic—only infrastructure code that multiple features depend on.

## Folder Structure

```
core/
├── result.dart              # Result<T, E> type for error handling
├── audio/
│   └── audio_player.dart    # Playback abstraction
├── storage/
│   └── hive_init.dart       # Hive database initialization
└── utils/
    └── string_utils.dart    # Text normalization utilities
```

---

## Files

### `result.dart`

**Purpose**: Type-safe error handling without exceptions.

**Responsibilities**:
- Provide `Result<T, E>` sealed class
- Enable exhaustive pattern matching
- Replace throwing exceptions in domain layer

**Implementation**:

```dart
/// A discriminated union representing either success or failure.
sealed class Result<T, E> {
  const Result();

  /// Returns true if this is a successful result.
  bool get isSuccess => this is Success<T, E>;

  /// Returns true if this is a failure result.
  bool get isFailure => this is Failure<T, E>;

  /// Returns the success value or null.
  T? get valueOrNull => switch (this) {
    Success(:final value) => value,
    Failure() => null,
  };

  /// Returns the error or null.
  E? get errorOrNull => switch (this) {
    Success() => null,
    Failure(:final error) => error,
  };

  /// Maps the success value.
  Result<U, E> map<U>(U Function(T value) transform) => switch (this) {
    Success(:final value) => Success(transform(value)),
    Failure(:final error) => Failure(error),
  };

  /// Maps the error value.
  Result<T, F> mapError<F>(F Function(E error) transform) => switch (this) {
    Success(:final value) => Success(value),
    Failure(:final error) => Failure(transform(error)),
  };

  /// Chains another Result-returning operation.
  Result<U, E> flatMap<U>(Result<U, E> Function(T value) transform) => switch (this) {
    Success(:final value) => transform(value),
    Failure(:final error) => Failure(error),
  };

  /// Executes the appropriate callback based on the result.
  R when<R>({
    required R Function(T value) success,
    required R Function(E error) failure,
  }) => switch (this) {
    Success(:final value) => success(value),
    Failure(:final error) => failure(error),
  };
}

/// Represents a successful result containing a value.
final class Success<T, E> extends Result<T, E> {
  final T value;
  const Success(this.value);

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is Success<T, E> && other.value == value;

  @override
  int get hashCode => value.hashCode;

  @override
  String toString() => 'Success($value)';
}

/// Represents a failed result containing an error.
final class Failure<T, E> extends Result<T, E> {
  final E error;
  const Failure(this.error);

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is Failure<T, E> && other.error == error;

  @override
  int get hashCode => error.hashCode;

  @override
  String toString() => 'Failure($error)';
}
```

**Usage**:

```dart
// Repository returns Result
Future<Result<String, RecognitionError>> recognize(String path);

// Consumer handles both cases
final result = await recognizer.recognize(audioPath);
result.when(
  success: (text) => print('Recognized: $text'),
  failure: (error) => print('Error: ${error.message}'),
);
```

**Design Decision**: Using a sealed class enables exhaustive switch statements. The Dart 3 pattern matching makes this ergonomic.

---

### `audio/audio_player.dart`

**Purpose**: Abstraction over audio playback allowing different sources.

**Responsibilities**:
- Define `AudioSource` sealed type
- Define `AudioPlayer` interface
- Enable playing from assets, files, or URLs

**Implementation**:

```dart
import 'dart:typed_data';

/// Represents different sources for audio playback.
sealed class AudioSource {
  const AudioSource();
}

/// Audio loaded from app assets.
class AssetAudioSource extends AudioSource {
  final String assetPath;
  const AssetAudioSource(this.assetPath);
}

/// Audio loaded from a local file.
class FileAudioSource extends AudioSource {
  final String filePath;
  const FileAudioSource(this.filePath);
}

/// Audio streamed from a URL.
class UrlAudioSource extends AudioSource {
  final String url;
  const UrlAudioSource(this.url);
}

/// Playback state.
enum PlaybackState {
  idle,
  loading,
  playing,
  paused,
  completed,
  error,
}

/// Abstract interface for audio playback.
abstract class AudioPlayer {
  /// Current playback state.
  Stream<PlaybackState> get stateStream;

  /// Current playback position.
  Stream<Duration> get positionStream;

  /// Total duration (available after loading).
  Duration? get duration;

  /// Loads an audio source without playing.
  Future<void> load(AudioSource source);

  /// Plays the loaded audio.
  Future<void> play();

  /// Pauses playback.
  Future<void> pause();

  /// Stops playback and resets position.
  Future<void> stop();

  /// Seeks to a position.
  Future<void> seek(Duration position);

  /// Disposes resources.
  Future<void> dispose();
}
```

**Implementation Note**: The concrete implementation uses `just_audio` package. See `features/example_audio/data/` for the implementation.

---

### `storage/hive_init.dart`

**Purpose**: Initialize Hive database and register adapters.

**Responsibilities**:
- Call `Hive.initFlutter()` once at app start
- Register type adapters for custom models
- Open required boxes

**Implementation**:

```dart
import 'package:hive_flutter/hive_flutter.dart';

/// Names of Hive boxes used in the app.
abstract class HiveBoxes {
  static const progress = 'progress';
  static const settings = 'settings';
}

/// Initializes Hive and opens required boxes.
/// Call this once in main() before runApp().
Future<HiveBoxManager> initHive() async {
  await Hive.initFlutter();

  // Register adapters for custom types
  // (Generated by hive_generator for each model with @HiveType)
  // Hive.registerAdapter(TextSequenceProgressAdapter());
  // Hive.registerAdapter(ScoreAttemptAdapter());

  final progressBox = await Hive.openBox<Map>(HiveBoxes.progress);
  final settingsBox = await Hive.openBox<dynamic>(HiveBoxes.settings);

  return HiveBoxManager(
    progress: progressBox,
    settings: settingsBox,
  );
}

/// Container for opened Hive boxes.
class HiveBoxManager {
  final Box<Map> progress;
  final Box<dynamic> settings;

  const HiveBoxManager({
    required this.progress,
    required this.settings,
  });

  Future<void> close() async {
    await progress.close();
    await settings.close();
  }
}
```

**Design Decision**: Using `Box<Map>` instead of typed boxes for flexibility. JSON serialization to/from maps allows easier schema evolution than Hive adapters. The tradeoff is slightly more verbose code in repositories.

---

### `utils/string_utils.dart`

**Purpose**: Text normalization utilities for scoring.

**Responsibilities**:
- Normalize Chinese text for comparison
- Remove punctuation and whitespace
- Unify full-width/half-width characters

**Implementation**:

```dart
/// Normalizes Chinese text for pronunciation comparison.
///
/// Operations:
/// 1. Removes all punctuation (Chinese and ASCII)
/// 2. Removes all whitespace
/// 3. Converts full-width ASCII to half-width
/// 4. Converts to lowercase (for any ASCII)
String normalizeZhText(String input) {
  // Chinese punctuation + ASCII punctuation
  final punctuation = RegExp(
    r'[。，！？、；：""''（）【】《》〈〉…—·\s\.\,\!\?\;\:\"\'\(\)\[\]\<\>]',
  );

  var result = input.replaceAll(punctuation, '');

  // Full-width ASCII to half-width (！→! ａ→a etc.)
  result = _fullWidthToHalfWidth(result);

  return result.toLowerCase();
}

/// Converts full-width ASCII characters to half-width.
String _fullWidthToHalfWidth(String input) {
  final buffer = StringBuffer();
  for (final codeUnit in input.codeUnits) {
    // Full-width range: 0xFF01 (！) to 0xFF5E (～)
    // Maps to ASCII 0x21 (!) to 0x7E (~)
    if (codeUnit >= 0xFF01 && codeUnit <= 0xFF5E) {
      buffer.writeCharCode(codeUnit - 0xFEE0);
    }
    // Full-width space (0x3000) to ASCII space (0x20)
    else if (codeUnit == 0x3000) {
      buffer.writeCharCode(0x20);
    } else {
      buffer.writeCharCode(codeUnit);
    }
  }
  return buffer.toString();
}

/// Calculates Character Error Rate (CER) between reference and hypothesis.
///
/// CER = edit_distance(ref, hyp) / length(ref)
///
/// Returns a value between 0.0 (perfect match) and potentially > 1.0
/// (hypothesis much longer than reference).
double calculateCer(String reference, String hypothesis) {
  if (reference.isEmpty) {
    return hypothesis.isEmpty ? 0.0 : 1.0;
  }

  final distance = _levenshteinDistance(reference, hypothesis);
  return distance / reference.length;
}

/// Levenshtein distance (edit distance) between two strings.
int _levenshteinDistance(String s1, String s2) {
  if (s1.isEmpty) return s2.length;
  if (s2.isEmpty) return s1.length;

  // Use characters (not code units) for proper Unicode handling
  final chars1 = s1.characters.toList();
  final chars2 = s2.characters.toList();

  final len1 = chars1.length;
  final len2 = chars2.length;

  // Two-row optimization (only need current and previous row)
  var prev = List<int>.generate(len2 + 1, (i) => i);
  var curr = List<int>.filled(len2 + 1, 0);

  for (var i = 1; i <= len1; i++) {
    curr[0] = i;
    for (var j = 1; j <= len2; j++) {
      final cost = chars1[i - 1] == chars2[j - 1] ? 0 : 1;
      curr[j] = [
        prev[j] + 1,      // deletion
        curr[j - 1] + 1,  // insertion
        prev[j - 1] + cost, // substitution
      ].reduce((a, b) => a < b ? a : b);
    }
    final temp = prev;
    prev = curr;
    curr = temp;
  }

  return prev[len2];
}
```

**Design Decision**: Using `String.characters` for proper Unicode grapheme cluster handling. Chinese characters are single graphemes, but this handles edge cases like combining characters correctly.

---

## Integration Tests

### `result_test.dart`

```dart
void main() {
  group('Result', () {
    test('Success.when calls success callback', () {
      const result = Success<int, String>(42);
      final output = result.when(
        success: (v) => 'value: $v',
        failure: (e) => 'error: $e',
      );
      expect(output, 'value: 42');
    });

    test('Failure.when calls failure callback', () {
      const result = Failure<int, String>('error');
      final output = result.when(
        success: (v) => 'value: $v',
        failure: (e) => 'error: $e',
      );
      expect(output, 'error: error');
    });

    test('map transforms success value', () {
      const result = Success<int, String>(10);
      final mapped = result.map((v) => v * 2);
      expect(mapped.valueOrNull, 20);
    });

    test('map preserves failure', () {
      const result = Failure<int, String>('error');
      final mapped = result.map((v) => v * 2);
      expect(mapped.errorOrNull, 'error');
    });

    test('flatMap chains operations', () {
      const result = Success<int, String>(10);
      final chained = result.flatMap((v) => Success(v.toString()));
      expect(chained.valueOrNull, '10');
    });
  });
}
```

### `string_utils_test.dart`

```dart
void main() {
  group('normalizeZhText', () {
    test('removes Chinese punctuation', () {
      expect(normalizeZhText('我想喝水。'), '我想喝水');
      expect(normalizeZhText('你好！'), '你好');
      expect(normalizeZhText('好的，谢谢'), '好的谢谢');
    });

    test('removes ASCII punctuation', () {
      expect(normalizeZhText('Hello, world!'), 'helloworld');
    });

    test('removes whitespace', () {
      expect(normalizeZhText('你 好'), '你好');
      expect(normalizeZhText('你　好'), '你好'); // Full-width space
    });

    test('handles full-width ASCII', () {
      expect(normalizeZhText('Ａ'), 'a');
      expect(normalizeZhText('１２３'), '123');
    });

    test('handles empty string', () {
      expect(normalizeZhText(''), '');
    });
  });

  group('calculateCer', () {
    test('returns 0 for identical strings', () {
      expect(calculateCer('我想喝水', '我想喝水'), 0.0);
    });

    test('returns 1 for completely different strings of same length', () {
      expect(calculateCer('你好', '我们'), 1.0);
    });

    test('calculates partial match correctly', () {
      // Reference: 我想喝水 (4 chars)
      // Hypothesis: 我想和水 (1 substitution)
      // CER = 1/4 = 0.25
      expect(calculateCer('我想喝水', '我想和水'), 0.25);
    });

    test('handles empty reference', () {
      expect(calculateCer('', ''), 0.0);
      expect(calculateCer('', 'abc'), 1.0);
    });

    test('handles empty hypothesis', () {
      // All deletions: CER = length(ref) / length(ref) = 1.0
      expect(calculateCer('abc', ''), 1.0);
    });
  });
}
```

### `audio_player_test.dart`

```dart
void main() {
  group('AudioSource', () {
    test('AssetAudioSource holds asset path', () {
      const source = AssetAudioSource('assets/audio/test.opus');
      expect(source.assetPath, 'assets/audio/test.opus');
    });

    test('FileAudioSource holds file path', () {
      const source = FileAudioSource('/data/recordings/test.m4a');
      expect(source.filePath, '/data/recordings/test.m4a');
    });

    test('UrlAudioSource holds URL', () {
      const source = UrlAudioSource('https://example.com/audio.opus');
      expect(source.url, 'https://example.com/audio.opus');
    });

    test('sources are distinct types', () {
      const asset = AssetAudioSource('path');
      const file = FileAudioSource('path');
      expect(asset, isNot(equals(file)));
    });
  });
}
```

---

## Design Notes

### Why Result instead of Exceptions?

1. **Explicit error handling** - Caller must handle both cases
2. **No hidden control flow** - Errors are values, not jumps
3. **Composable** - `map`, `flatMap` enable functional pipelines
4. **Type-safe** - Error types are part of the signature

### Why Separate AudioSource Types?

Different sources require different handling:
- Assets need `rootBundle` loading
- Files need file system access
- URLs need streaming/caching

Sealed class forces exhaustive handling in the player implementation.

### Why Manual Levenshtein?

- No external dependency
- Optimized for our use case (short strings)
- Proper Unicode handling with `characters` package
