import '../../progress/domain/text_sequence_progress.dart';
import '../../text_sequences/domain/text_sequence.dart';

/// A text sequence with its computed priority and progress.
class RankedSequence {
  const RankedSequence({
    required this.sequence,
    required this.priority,
    this.progress,
  });

  /// The text sequence.
  final TextSequence sequence;

  /// Computed priority score (higher = more likely to be selected).
  final double priority;

  /// The user's progress on this sequence, if any.
  final TextSequenceProgress? progress;
}
