import 'recording.dart';

/// Repository interface for managing user recordings.
abstract class RecordingRepository {
  /// Saves a recording, replacing any existing recording for the same sequence.
  Future<void> saveLatest(Recording recording);

  /// Gets the latest recording for a text sequence.
  /// Returns null if no recording exists.
  Future<Recording?> getLatest(String textSequenceId);

  /// Deletes the recording for a text sequence.
  Future<void> deleteLatest(String textSequenceId);

  /// Checks if a recording exists for a text sequence.
  Future<bool> hasRecording(String textSequenceId);
}
