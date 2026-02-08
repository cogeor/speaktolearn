import '../../progress/domain/progress_repository.dart';
import '../../text_sequences/domain/text_sequence.dart';
import '../../text_sequences/domain/text_sequence_repository.dart';
import 'sequence_ranker.dart';

/// Use case for selecting the next tracked sequence to practice.
class GetNextTrackedSequence {
  GetNextTrackedSequence({
    required TextSequenceRepository textSequenceRepository,
    required ProgressRepository progressRepository,
    required SequenceRanker ranker,
  }) : _textSequenceRepository = textSequenceRepository,
       _progressRepository = progressRepository,
       _ranker = ranker;

  final TextSequenceRepository _textSequenceRepository;
  final ProgressRepository _progressRepository;
  final SequenceRanker _ranker;

  /// Gets the next best tracked sequence to practice.
  ///
  /// Optionally excludes [currentId] from selection.
  /// Returns null if no tracked sequences exist.
  Future<TextSequence?> call({String? currentId}) async {
    // Get all tracked sequence IDs
    final trackedIds = await _progressRepository.getTrackedIds();
    if (trackedIds.isEmpty) return null;

    // Get the actual sequences
    final sequences = <TextSequence>[];
    for (final id in trackedIds) {
      final seq = await _textSequenceRepository.getById(id);
      if (seq != null) {
        sequences.add(seq);
      }
    }

    if (sequences.isEmpty) return null;

    // Get progress for ranking
    final progressMap = await _progressRepository.getProgressMap(
      trackedIds.toList(),
    );

    // Use ranker to select next
    return _ranker.selectNext(sequences, progressMap, excludeId: currentId);
  }
}
