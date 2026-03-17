import 'recording.dart';
import 'recording_metadata.dart';

/// Repository interface for managing user recordings.
abstract class RecordingRepository {
  /// Saves a recording, replacing any existing recording for the same sequence.
  Future<void> saveLatest(Recording recording);

  /// Writes a sidecar metadata JSON file alongside the WAV for [textSequenceId].
  ///
  /// The JSON file is named `{textSequenceId}.json` and lives in the same
  /// recordings directory as the corresponding WAV. It is silently overwritten
  /// if it already exists.
  Future<void> saveMetadata(RecordingMetadata metadata);

  /// Gets the latest recording for a text sequence.
  /// Returns null if no recording exists.
  Future<Recording?> getLatest(String textSequenceId);

  /// Deletes the recording for a text sequence.
  Future<void> deleteLatest(String textSequenceId);

  /// Checks if a recording exists for a text sequence.
  Future<bool> hasRecording(String textSequenceId);

  /// Returns all historical recordings for a text sequence, oldest first.
  Future<List<Recording>> listHistory(String textSequenceId);
}
