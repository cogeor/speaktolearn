import 'dart:math';

import '../../progress/domain/sentence_rating.dart';
import '../../progress/domain/text_sequence_progress.dart';
import '../../text_sequences/domain/text_sequence.dart';
import 'ranked_sequence.dart';

/// Interface for ranking and selecting text sequences based on progress.
abstract class SequenceRanker {
  /// Ranks all sequences by priority based on user progress.
  ///
  /// Returns a list of [RankedSequence] sorted by priority (highest first).
  /// The [progressMap] maps sequence IDs to their corresponding progress.
  List<RankedSequence> rank(
    List<TextSequence> sequences,
    Map<String, TextSequenceProgress> progressMap,
  );

  /// Selects the next best sequence for the user to practice.
  ///
  /// Returns the highest priority sequence, or `null` if no sequences are
  /// available. Use [excludeId] to skip a specific sequence (e.g., the
  /// currently active one).
  TextSequence? selectNext(
    List<TextSequence> sequences,
    Map<String, TextSequenceProgress> progressMap, {
    String? excludeId,
  });
}

/// Default implementation of [SequenceRanker] using a priority algorithm.
///
/// Priority is computed based on:
/// - Need: How much practice is needed (inverse of best score)
/// - Recency: Recent practice reduces priority
/// - New sequences get a high priority bonus
class DefaultSequenceRanker implements SequenceRanker {
  DefaultSequenceRanker({Random? random}) : _random = random ?? Random();

  final Random _random;
  static const int _topK = 5;

  @override
  List<RankedSequence> rank(
    List<TextSequence> sequences,
    Map<String, TextSequenceProgress> progressMap,
  ) {
    final ranked = sequences.map((seq) {
      final progress = progressMap[seq.id];
      final priority = _computePriority(progress);
      return RankedSequence(
        sequence: seq,
        priority: priority,
        progress: progress,
      );
    }).toList();

    ranked.sort((a, b) => b.priority.compareTo(a.priority));
    return ranked;
  }

  @override
  TextSequence? selectNext(
    List<TextSequence> sequences,
    Map<String, TextSequenceProgress> progressMap, {
    String? excludeId,
  }) {
    final filtered = excludeId == null
        ? sequences
        : sequences.where((s) => s.id != excludeId).toList();

    if (filtered.isEmpty) return null;

    final ranked = rank(filtered, progressMap);
    final topCandidates = ranked.take(_topK).toList();

    return topCandidates[_random.nextInt(topCandidates.length)].sequence;
  }

  double _computePriority(TextSequenceProgress? progress) {
    if (progress == null) {
      // New sequences get high priority with a bonus
      return 100.0;
    }

    // Need: how much practice is needed (based on last rating)
    // hard=100, almost=75, good=50, easy=25, none=100
    final need = _ratingToNeed(progress.lastRating);

    // Recency penalty: recent practice reduces priority
    if (progress.lastAttemptAt == null) {
      // No attempts yet, treat as new
      return need;
    }

    final daysSincePractice = DateTime.now()
        .difference(progress.lastAttemptAt!)
        .inDays
        .toDouble();
    final recencyPenalty = daysSincePractice < 1
        ? 50.0
        : (50.0 / daysSincePractice);

    return need - recencyPenalty;
  }

  double _ratingToNeed(SentenceRating? rating) {
    if (rating == null) return 100.0;
    switch (rating) {
      case SentenceRating.hard:
        return 100.0;
      case SentenceRating.almost:
        return 75.0;
      case SentenceRating.good:
        return 50.0;
      case SentenceRating.easy:
        return 25.0;
    }
  }
}
