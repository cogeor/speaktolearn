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
