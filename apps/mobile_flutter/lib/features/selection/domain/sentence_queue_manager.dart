import 'dart:collection';

import '../../progress/domain/progress_repository.dart';
import '../../progress/domain/sentence_rating.dart';
import '../../progress/domain/text_sequence_progress.dart';
import '../../text_sequences/domain/text_sequence.dart';
import '../../text_sequences/domain/text_sequence_repository.dart';

/// Manages a queue of sentences for spaced repetition practice.
///
/// Maintains a queue of ~10 sentences and prioritizes them based on
/// self-reported ratings. Never-seen sentences have highest priority,
/// followed by hard > almost > good > easy.
class SentenceQueueManager {
  SentenceQueueManager({
    required TextSequenceRepository textSequenceRepository,
    required ProgressRepository progressRepository,
  }) : _textSequenceRepository = textSequenceRepository,
       _progressRepository = progressRepository;

  final TextSequenceRepository _textSequenceRepository;
  final ProgressRepository _progressRepository;
  final Queue<TextSequence> _queue = Queue();

  /// Default queue size to maintain.
  static const _queueSize = 10;

  /// Gets the next sentence to practice.
  ///
  /// Refills the queue if needed. Optionally excludes [excludeId] from
  /// selection (e.g., the currently displayed sentence).
  /// Returns null if no sequences exist for the level.
  Future<TextSequence?> getNext({int level = 1, String? excludeId}) async {
    if (_queue.isEmpty) {
      await _refillQueue(level);
    }

    if (_queue.isEmpty) return null;

    // Get next, excluding current if specified
    TextSequence? next;
    if (excludeId != null) {
      // Find first that's not the current
      for (final s in _queue) {
        if (s.id != excludeId) {
          next = s;
          _queue.remove(s);
          break;
        }
      }
      // If all items are the excluded one, just return the first
      next ??= _queue.removeFirst();
    } else {
      next = _queue.removeFirst();
    }

    return next;
  }

  /// Refills the queue with prioritized sentences from the given level.
  Future<void> _refillQueue(int level) async {
    final sequences = await _textSequenceRepository.getByLevel(level);
    if (sequences.isEmpty) return;

    final progressMap = await _progressRepository.getProgressMap(
      sequences.map((s) => s.id).toList(),
    );

    // Sort by priority: never-seen first, then by rating (hard > almost > good > easy)
    final sorted = List<TextSequence>.from(sequences);
    sorted.sort((a, b) {
      final progressA = progressMap[a.id];
      final progressB = progressMap[b.id];
      return _comparePriority(progressA, progressB);
    });

    // Take top N for queue
    _queue.addAll(sorted.take(_queueSize));
  }

  /// Compares two progress entries for priority sorting.
  ///
  /// Returns negative if a has higher priority, positive if b has higher priority.
  int _comparePriority(TextSequenceProgress? a, TextSequenceProgress? b) {
    final priorityA = _getPriority(a?.lastRating);
    final priorityB = _getPriority(b?.lastRating);
    return priorityA.compareTo(priorityB);
  }

  /// Gets the numeric priority for a rating.
  ///
  /// Lower numbers = higher priority:
  /// - 0: Never seen (highest priority)
  /// - 1: Hard
  /// - 2: Almost
  /// - 3: Good
  /// - 4: Easy (lowest priority)
  int _getPriority(SentenceRating? rating) {
    if (rating == null) return 0; // Never seen = highest priority
    return switch (rating) {
      SentenceRating.hard => 1,
      SentenceRating.almost => 2,
      SentenceRating.good => 3,
      SentenceRating.easy => 4,
    };
  }

  /// Clears the queue.
  ///
  /// Useful when the user changes levels or when you want to force a refill.
  void clear() => _queue.clear();

  /// Returns the current queue length.
  int get length => _queue.length;

  /// Returns whether the queue is empty.
  bool get isEmpty => _queue.isEmpty;
}
