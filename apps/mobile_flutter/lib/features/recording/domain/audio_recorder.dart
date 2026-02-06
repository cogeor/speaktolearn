import '../../../core/result.dart';

/// Errors that can occur during audio recording.
enum RecordingError {
  /// Microphone permission was denied by the user.
  permissionDenied,

  /// No microphone is available on the device.
  noMicrophone,

  /// Recording is not supported on this platform.
  notSupported,

  /// An error occurred during recording.
  recordingFailed,

  /// Recording was already in progress.
  alreadyRecording,

  /// No recording in progress to stop.
  notRecording,
}

/// Abstract interface for audio recording functionality.
abstract class AudioRecorder {
  /// Whether a recording is currently in progress.
  bool get isRecording;

  /// Starts recording audio.
  ///
  /// Returns the file path where the recording will be saved on success,
  /// or a [RecordingError] on failure.
  Future<Result<String, RecordingError>> start();

  /// Stops the current recording.
  ///
  /// Returns the file path of the recorded audio on success,
  /// or a [RecordingError] on failure.
  Future<Result<String, RecordingError>> stop();

  /// Cancels the current recording without saving.
  Future<void> cancel();

  /// Releases resources used by the recorder.
  Future<void> dispose();
}
