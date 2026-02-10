# features/scoring/ Module

## Purpose

Scores user pronunciation by comparing ASR-recognized text to the expected text. Uses Character Error Rate (CER) to compute a 0-100 score.

## Folder Structure

```
scoring/
├── domain/
│   ├── grade.dart                    # Grade entity
│   └── pronunciation_scorer.dart     # Scoring interface
└── data/
    ├── speech_recognizer.dart        # ASR interface + impl
    ├── asr_similarity_scorer.dart    # Scorer implementation
    └── cer_calculator.dart           # CER algorithm
```

---

## Domain Layer

### `grade.dart`

**Purpose**: Represents a pronunciation score.

**Implementation**:

```dart
import 'package:freezed_annotation/freezed_annotation.dart';

part 'grade.freezed.dart';
part 'grade.g.dart';

/// Result of pronunciation scoring.
@freezed
class Grade with _$Grade {
  const factory Grade({
    /// Overall score (0-100).
    required double overall,

    /// Scoring method identifier (e.g., "asr_cer_v1").
    required String method,

    /// Text recognized by ASR.
    String? recognizedText,

    /// Additional metrics (e.g., {"cer": 0.15, "insertions": 2}).
    Map<String, dynamic>? details,
  }) = _Grade;

  factory Grade.fromJson(Map<String, dynamic> json) => _$GradeFromJson(json);
}
```

---

### `pronunciation_scorer.dart`

**Purpose**: Interface for pronunciation scoring.

**Implementation**:

```dart
import '../recording/domain/recording.dart';
import '../text_sequences/domain/text_sequence.dart';
import 'grade.dart';

/// Interface for pronunciation scoring.
///
/// Different implementations can use different scoring methods:
/// - ASR + text similarity (Option A - MVP)
/// - Phoneme alignment (Option B)
/// - Cloud pronunciation API (Option C)
abstract class PronunciationScorer {
  /// Scores the pronunciation in a recording against expected text.
  Future<Grade> score(TextSequence expected, Recording recording);
}
```

---

## Data Layer

### `speech_recognizer.dart`

**Purpose**: Interface for speech-to-text recognition.

**Implementation**:

```dart
import '../../../core/result.dart';

/// Error types for speech recognition.
enum RecognitionError {
  notAvailable,
  permissionDenied,
  noSpeechDetected,
  networkError,
  unknown,
}

/// Interface for speech-to-text recognition.
abstract class SpeechRecognizer {
  /// Recognizes speech from an audio file.
  ///
  /// [audioPath] is the path to the audio file.
  /// [languageCode] is the BCP-47 language code (e.g., "zh-CN").
  Future<Result<String, RecognitionError>> recognize(
    String audioPath, {
    required String languageCode,
  });

  /// Checks if recognition is available for a language.
  Future<bool> isAvailable(String languageCode);
}

/// Implementation using speech_to_text package.
class SpeechToTextRecognizer implements SpeechRecognizer {
  // Note: speech_to_text primarily works with streaming.
  // For file-based recognition, you may need to:
  // 1. Use a different package (e.g., google_speech)
  // 2. Implement streaming recognition
  // 3. Use a cloud API directly

  @override
  Future<Result<String, RecognitionError>> recognize(
    String audioPath, {
    required String languageCode,
  }) async {
    // Implementation depends on chosen approach
    // This is a placeholder for the MVP

    // Option 1: Use streaming recognition with audio playback
    // Option 2: Use cloud API (Google Speech, Azure, etc.)
    // Option 3: Use platform-specific APIs directly

    throw UnimplementedError(
      'File-based recognition requires cloud API integration',
    );
  }

  @override
  Future<bool> isAvailable(String languageCode) async {
    // Check if the device supports this language
    // speech_to_text provides a method for this
    return true;
  }
}

/// Mock recognizer for testing.
class MockSpeechRecognizer implements SpeechRecognizer {
  final Map<String, String> _responses;

  MockSpeechRecognizer(this._responses);

  @override
  Future<Result<String, RecognitionError>> recognize(
    String audioPath, {
    required String languageCode,
  }) async {
    final response = _responses[audioPath];
    if (response != null) {
      return Success(response);
    }
    return const Failure(RecognitionError.noSpeechDetected);
  }

  @override
  Future<bool> isAvailable(String languageCode) async => true;
}
```

**Implementation Note**: File-based speech recognition on mobile is complex. Consider:
1. **Cloud API**: Google Speech-to-Text, Azure Speech, etc.
2. **Streaming workaround**: Play audio to speech_to_text listener
3. **Platform channels**: Direct Android/iOS speech APIs

---

### `cer_calculator.dart`

**Purpose**: Character Error Rate calculation.

**Implementation**:

```dart
import '../../../core/utils/string_utils.dart';

/// Calculates Character Error Rate (CER) for Chinese text.
class CerCalculator {
  /// Computes CER between reference and hypothesis.
  ///
  /// Both strings are normalized before comparison.
  /// Returns a value between 0.0 (perfect) and potentially > 1.0.
  CerResult calculate(String reference, String hypothesis) {
    final normalizedRef = normalizeZhText(reference);
    final normalizedHyp = normalizeZhText(hypothesis);

    if (normalizedRef.isEmpty) {
      return CerResult(
        cer: normalizedHyp.isEmpty ? 0.0 : 1.0,
        referenceLength: 0,
        hypothesisLength: normalizedHyp.length,
        editDistance: normalizedHyp.length,
      );
    }

    final distance = _levenshteinDistance(normalizedRef, normalizedHyp);
    final cer = distance / normalizedRef.length;

    return CerResult(
      cer: cer,
      referenceLength: normalizedRef.length,
      hypothesisLength: normalizedHyp.length,
      editDistance: distance,
    );
  }

  int _levenshteinDistance(String s1, String s2) {
    // Same implementation as in string_utils.dart
    // Could be extracted to shared utility
    final chars1 = s1.characters.toList();
    final chars2 = s2.characters.toList();
    final len1 = chars1.length;
    final len2 = chars2.length;

    var prev = List<int>.generate(len2 + 1, (i) => i);
    var curr = List<int>.filled(len2 + 1, 0);

    for (var i = 1; i <= len1; i++) {
      curr[0] = i;
      for (var j = 1; j <= len2; j++) {
        final cost = chars1[i - 1] == chars2[j - 1] ? 0 : 1;
        curr[j] = [
          prev[j] + 1,
          curr[j - 1] + 1,
          prev[j - 1] + cost,
        ].reduce((a, b) => a < b ? a : b);
      }
      final temp = prev;
      prev = curr;
      curr = temp;
    }

    return prev[len2];
  }
}

/// Result of CER calculation.
class CerResult {
  final double cer;
  final int referenceLength;
  final int hypothesisLength;
  final int editDistance;

  const CerResult({
    required this.cer,
    required this.referenceLength,
    required this.hypothesisLength,
    required this.editDistance,
  });

  /// Converts CER to a 0-100 score.
  double get score => (100 * (1 - cer)).clamp(0.0, 100.0);
}
```

---

### `asr_similarity_scorer.dart`

**Purpose**: Pronunciation scorer using ASR + CER.

**Implementation**:

```dart
import '../domain/pronunciation_scorer.dart';
import '../domain/grade.dart';
import '../recording/domain/recording.dart';
import '../text_sequences/domain/text_sequence.dart';
import 'speech_recognizer.dart';
import 'cer_calculator.dart';

/// Pronunciation scorer using ASR and text similarity.
///
/// Algorithm (Option A):
/// 1. Run speech-to-text on the recording
/// 2. Normalize both expected and recognized text
/// 3. Calculate Character Error Rate (CER)
/// 4. Convert to 0-100 score
class AsrSimilarityScorer implements PronunciationScorer {
  final SpeechRecognizer _recognizer;
  final CerCalculator _cerCalculator;

  static const _methodVersion = 'asr_cer_v1';

  AsrSimilarityScorer(this._recognizer)
      : _cerCalculator = CerCalculator();

  @override
  Future<Grade> score(TextSequence expected, Recording recording) async {
    // Run ASR
    final recognitionResult = await _recognizer.recognize(
      recording.filePath,
      languageCode: expected.language,
    );

    // Handle recognition failure
    if (recognitionResult.isFailure) {
      return Grade(
        overall: 0.0,
        method: _methodVersion,
        details: {
          'error': recognitionResult.errorOrNull.toString(),
        },
      );
    }

    final recognizedText = recognitionResult.valueOrNull!;

    // Calculate CER
    final cerResult = _cerCalculator.calculate(expected.text, recognizedText);

    return Grade(
      overall: cerResult.score,
      method: _methodVersion,
      recognizedText: recognizedText,
      details: {
        'cer': cerResult.cer,
        'editDistance': cerResult.editDistance,
        'referenceLength': cerResult.referenceLength,
        'hypothesisLength': cerResult.hypothesisLength,
      },
    );
  }
}
```

---

## Riverpod Providers

```dart
import 'package:flutter_riverpod/flutter_riverpod.dart';

final speechRecognizerProvider = Provider<SpeechRecognizer>((ref) {
  return SpeechToTextRecognizer();
});

final pronunciationScorerProvider = Provider<PronunciationScorer>((ref) {
  final recognizer = ref.watch(speechRecognizerProvider);
  return AsrSimilarityScorer(recognizer);
});
```

---

## Integration Tests

### CER Calculator Tests

```dart
void main() {
  group('CerCalculator', () {
    late CerCalculator calculator;

    setUp(() {
      calculator = CerCalculator();
    });

    test('perfect match returns 0 CER', () {
      final result = calculator.calculate('我想喝水', '我想喝水');

      expect(result.cer, 0.0);
      expect(result.score, 100.0);
    });

    test('completely different returns 1 CER', () {
      final result = calculator.calculate('你好', '我们');

      expect(result.cer, 1.0);
      expect(result.score, 0.0);
    });

    test('one character difference', () {
      // Reference: 我想喝水 (4 chars)
      // Hypothesis: 我想和水 (1 substitution)
      // CER = 1/4 = 0.25
      final result = calculator.calculate('我想喝水', '我想和水');

      expect(result.cer, 0.25);
      expect(result.score, 75.0);
    });

    test('missing character', () {
      // Reference: 我想喝水 (4 chars)
      // Hypothesis: 我想喝 (1 deletion)
      // CER = 1/4 = 0.25
      final result = calculator.calculate('我想喝水', '我想喝');

      expect(result.cer, 0.25);
      expect(result.score, 75.0);
    });

    test('extra character', () {
      // Reference: 我想喝水 (4 chars)
      // Hypothesis: 我想喝水了 (1 insertion)
      // CER = 1/4 = 0.25
      final result = calculator.calculate('我想喝水', '我想喝水了');

      expect(result.cer, 0.25);
      expect(result.score, 75.0);
    });

    test('normalizes punctuation', () {
      // Both should normalize to the same string
      final result = calculator.calculate('我想喝水。', '我想喝水');

      expect(result.cer, 0.0);
      expect(result.score, 100.0);
    });

    test('handles empty reference', () {
      final result = calculator.calculate('', '你好');
      expect(result.cer, 1.0);
    });

    test('handles empty hypothesis', () {
      final result = calculator.calculate('你好', '');
      expect(result.cer, 1.0);
    });

    test('handles both empty', () {
      final result = calculator.calculate('', '');
      expect(result.cer, 0.0);
    });
  });
}
```

### Scorer Integration Test

```dart
void main() {
  group('AsrSimilarityScorer', () {
    test('returns grade with score and recognized text', () async {
      final mockRecognizer = MockSpeechRecognizer({
        '/path/to/recording.m4a': '我想喝水',
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
        filePath: '/path/to/recording.m4a',
      );

      final grade = await scorer.score(sequence, recording);

      expect(grade.overall, 100.0);
      expect(grade.recognizedText, '我想喝水');
      expect(grade.method, 'asr_cer_v1');
      expect(grade.details?['cer'], 0.0);
    });

    test('handles partial match', () async {
      final mockRecognizer = MockSpeechRecognizer({
        '/path/to/recording.m4a': '我想和水', // One wrong character
      });
      final scorer = AsrSimilarityScorer(mockRecognizer);

      final sequence = TextSequence(
        id: 'ts_001',
        text: '我想喝水',
        language: 'zh-CN',
      );
      final recording = Recording(
        id: 'r1',
        textSequenceId: 'ts_001',
        createdAt: DateTime.now(),
        filePath: '/path/to/recording.m4a',
      );

      final grade = await scorer.score(sequence, recording);

      expect(grade.overall, 75.0);
      expect(grade.details?['cer'], 0.25);
    });

    test('handles recognition failure', () async {
      final mockRecognizer = MockSpeechRecognizer({});
      final scorer = AsrSimilarityScorer(mockRecognizer);

      final sequence = TextSequence(
        id: 'ts_001',
        text: '我想喝水',
        language: 'zh-CN',
      );
      final recording = Recording(
        id: 'r1',
        textSequenceId: 'ts_001',
        createdAt: DateTime.now(),
        filePath: '/path/to/unknown.m4a',
      );

      final grade = await scorer.score(sequence, recording);

      expect(grade.overall, 0.0);
      expect(grade.details?['error'], isNotNull);
    });
  });
}
```

### Scoring Flow Integration Test

```dart
void main() {
  group('Scoring flow', () {
    testWidgets('records and displays score', (tester) async {
      // This requires full app setup with mocked audio services
      // See practice/README.md for full flow test
    });
  });
}
```

---

## Notes

### Scoring Method Limitations (Option A)

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| ASR errors ≠ pronunciation errors | May penalize correct pronunciation | Accept lower scores as "directional" |
| Tone detection is weak | Tones often ignored by ASR | Future: Option B/C with phoneme analysis |
| Language support varies | Some devices/APIs better than others | Test on target devices |
| Network dependency | Cloud ASR requires internet | Cache-first, show offline warning |

### Future Improvements

1. **Option B (Phoneme alignment)**:
   - Convert to pinyin with tones
   - Compare phoneme by phoneme
   - Highlight specific errors

2. **Option C (Cloud pronunciation API)**:
   - Azure Speech pronunciation assessment
   - Google Cloud Speech
   - Returns per-phoneme scores

### Score Interpretation

| Score Range | Interpretation |
|-------------|---------------|
| 90-100 | Excellent - very close match |
| 70-89 | Good - minor errors |
| 50-69 | Fair - noticeable errors |
| 0-49 | Needs practice |

### ASR Implementation Options

For MVP with file-based ASR:

1. **Google Cloud Speech-to-Text**:
   ```dart
   // Use google_speech package
   final speech = SpeechToText.viaApiKey(apiKey);
   final result = await speech.recognize(
     RecognitionConfig(languageCode: 'zh-CN'),
     audioContent,
   );
   ```

2. **Azure Speech**:
   ```dart
   // Direct REST API call
   final response = await http.post(
     Uri.parse('https://$region.stt.speech.microsoft.com/...'),
     headers: {'Ocp-Apim-Subscription-Key': apiKey},
     body: audioBytes,
   );
   ```

3. **On-device with workaround**:
   - Play audio through speaker
   - Use speech_to_text to recognize
   - (Not recommended - poor UX)
