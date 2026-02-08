import 'dart:math';

import '../../text_sequences/domain/text_sequence.dart';
import '../../text_sequences/domain/text_sequence_repository.dart';

/// Use case for selecting a random sequence from the current HSK level.
class GetNextByLevel {
  GetNextByLevel({
    required TextSequenceRepository textSequenceRepository,
    Random? random,
  }) : _textSequenceRepository = textSequenceRepository,
       _random = random ?? Random();

  final TextSequenceRepository _textSequenceRepository;
  final Random _random;

  /// Gets a random sequence from the specified [level].
  ///
  /// Optionally excludes [currentId] from selection.
  /// Returns null if no sequences exist for the level.
  Future<TextSequence?> call({required int level, String? currentId}) async {
    final sequences = await _textSequenceRepository.getByLevel(level);
    if (sequences.isEmpty) return null;

    // Filter out current sequence if provided
    final available = currentId != null
        ? sequences.where((s) => s.id != currentId).toList()
        : sequences;

    // If only one sequence and it's the current one, return it anyway
    if (available.isEmpty) {
      return sequences.first;
    }

    // Select random sequence from available
    final index = _random.nextInt(available.length);
    return available[index];
  }
}
