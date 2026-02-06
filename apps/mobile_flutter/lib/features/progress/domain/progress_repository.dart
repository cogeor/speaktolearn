import 'score_attempt.dart';
import 'text_sequence_progress.dart';

/// Repository interface for managing text sequence progress and score attempts.
abstract class ProgressRepository {
  // Progress methods

  /// Returns progress for a text sequence, or initial progress if not found.
  Future<TextSequenceProgress> getProgress(String textSequenceId);

  /// Returns a map of progress for multiple text sequences.
  Future<Map<String, TextSequenceProgress>> getProgressMap(List<String> ids);

  /// Returns the IDs of all tracked text sequences.
  Future<Set<String>> getTrackedIds();

  /// Returns progress for all tracked text sequences.
  Future<List<TextSequenceProgress>> getTrackedProgress();

  // Tracking methods

  /// Toggles the tracked status of a text sequence.
  Future<void> toggleTracked(String textSequenceId);

  /// Sets the tracked status of a text sequence explicitly.
  Future<void> setTracked(String textSequenceId, bool tracked);

  // Attempt methods

  /// Saves a score attempt and updates the associated progress.
  Future<void> saveAttempt(ScoreAttempt attempt);

  /// Returns attempts for a text sequence, optionally limited.
  Future<List<ScoreAttempt>> getAttempts(String textSequenceId, {int? limit});
}
