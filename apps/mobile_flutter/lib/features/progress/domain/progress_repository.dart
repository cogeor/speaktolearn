import 'rating_attempt.dart';
import 'text_sequence_progress.dart';

/// Repository interface for managing text sequence progress and rating attempts.
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

  /// Saves a rating attempt and updates the associated progress.
  Future<void> saveAttempt(RatingAttempt attempt);

  /// Returns rating attempts for a text sequence, optionally limited.
  Future<List<RatingAttempt>> getAttempts(String textSequenceId, {int? limit});

  /// Returns all rating attempts across all sequences.
  Future<List<RatingAttempt>> getAllAttempts();

  // Debug methods

  /// Generates fake practice data for testing the stats screen.
  /// Only available in debug builds.
  ///
  /// [sequenceIds] - List of sequence IDs to generate attempts for.
  /// [days] - Number of days of data to generate (default: 60).
  /// [attemptsPerDay] - Number of attempts per day (default: 10).
  Future<void> generateFakeStats({
    required List<String> sequenceIds,
    int days = 60,
    int attemptsPerDay = 10,
  });

  /// Clears all progress and attempt data.
  /// Only available in debug builds.
  Future<void> clearAllStats();
}
