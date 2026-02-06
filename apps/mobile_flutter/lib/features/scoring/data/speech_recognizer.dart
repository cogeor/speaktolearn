import '../../../core/result.dart';

/// Errors that can occur during speech recognition.
enum RecognitionError {
  /// Speech recognition is not available on this device.
  notAvailable,

  /// Microphone permission was denied.
  permissionDenied,

  /// No speech was detected in the audio.
  noSpeechDetected,

  /// The audio file could not be read.
  audioReadError,

  /// Recognition failed for an unknown reason.
  recognitionFailed,
}

/// Interface for speech recognition services.
abstract class SpeechRecognizer {
  /// Recognizes speech from an audio file.
  ///
  /// Returns the recognized text on success, or a [RecognitionError] on failure.
  Future<Result<String, RecognitionError>> recognize(
    String audioPath,
    String languageCode,
  );

  /// Whether speech recognition is available on this device.
  Future<bool> isAvailable();
}

/// Mock implementation of [SpeechRecognizer] for testing.
class MockSpeechRecognizer implements SpeechRecognizer {
  /// Creates a mock recognizer with predefined responses.
  ///
  /// [responses] maps audio file paths to recognized text.
  MockSpeechRecognizer({
    Map<String, String>? responses,
    this.defaultResponse,
    this.shouldFail = false,
    this.failureError = RecognitionError.recognitionFailed,
  }) : _responses = responses ?? {};

  final Map<String, String> _responses;

  /// Default response when path not found in responses map.
  final String? defaultResponse;

  /// Whether recognition should fail.
  final bool shouldFail;

  /// The error to return when failing.
  final RecognitionError failureError;

  @override
  Future<Result<String, RecognitionError>> recognize(
    String audioPath,
    String languageCode,
  ) async {
    if (shouldFail) {
      return Failure(failureError);
    }

    final text = _responses[audioPath] ?? defaultResponse;
    if (text == null) {
      return const Failure(RecognitionError.noSpeechDetected);
    }

    return Success(text);
  }

  @override
  Future<bool> isAvailable() async => !shouldFail;
}
