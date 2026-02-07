// integration_test/mocks/mock_speech_recognizer.dart
import 'package:speak_to_learn/core/result.dart';
import 'package:speak_to_learn/features/scoring/data/speech_recognizer.dart';

/// Mock speech recognizer for integration tests.
///
/// Supports various response modes for testing different scoring scenarios.
class MockSpeechRecognizer implements SpeechRecognizer {
  MockSpeechRecognizer({
    this.mode = RecognizerMode.perfectMatch,
    this.customResponse,
  });

  /// The recognition mode to use.
  RecognizerMode mode;

  /// Custom response text (used when mode is [RecognizerMode.custom]).
  String? customResponse;

  /// Tracks the last recognized path for test assertions.
  String? lastRecognizedPath;

  /// Tracks recognition call count.
  int recognizeCallCount = 0;

  @override
  Future<Result<String, RecognitionError>> recognize(
    String audioPath,
    String languageCode,
  ) async {
    lastRecognizedPath = audioPath;
    recognizeCallCount++;

    switch (mode) {
      case RecognizerMode.perfectMatch:
        // Return the expected text (tests should set this via customResponse
        // or use the sequence text directly)
        return Success(customResponse ?? '你好');

      case RecognizerMode.partialMatch:
        // Return partial recognition (e.g., missing characters)
        return const Success('你');

      case RecognizerMode.noMatch:
        // Return completely different text
        return const Success('再见');

      case RecognizerMode.empty:
        // Return empty string (triggers 0 score)
        return const Success('');

      case RecognizerMode.failure:
        return const Failure(RecognitionError.recognitionFailed);

      case RecognizerMode.noSpeech:
        return const Failure(RecognitionError.noSpeechDetected);

      case RecognizerMode.custom:
        if (customResponse == null) {
          return const Failure(RecognitionError.noSpeechDetected);
        }
        return Success(customResponse!);
    }
  }

  @override
  Future<bool> isAvailable() async {
    return mode != RecognizerMode.failure;
  }

  /// Resets the mock state for a new test.
  void reset() {
    lastRecognizedPath = null;
    recognizeCallCount = 0;
  }

  /// Sets up the mock to return a perfect match for the given text.
  void setupPerfectMatch(String text) {
    mode = RecognizerMode.custom;
    customResponse = text;
  }

  /// Sets up the mock to return a partial match.
  void setupPartialMatch(String partialText) {
    mode = RecognizerMode.custom;
    customResponse = partialText;
  }

  /// Sets up the mock to fail recognition.
  void setupFailure() {
    mode = RecognizerMode.failure;
  }
}

/// Recognition behavior modes for testing.
enum RecognizerMode {
  /// Returns the exact expected text (100% match).
  perfectMatch,

  /// Returns partial text (partial score).
  partialMatch,

  /// Returns completely different text (0% match).
  noMatch,

  /// Returns empty string.
  empty,

  /// Simulates recognition failure.
  failure,

  /// Simulates no speech detected.
  noSpeech,

  /// Uses customResponse field.
  custom,
}
