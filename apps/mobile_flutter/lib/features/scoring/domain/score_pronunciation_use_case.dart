import 'package:freezed_annotation/freezed_annotation.dart';

import '../../../core/domain/use_case.dart';
import '../../recording/domain/recording.dart';
import '../../text_sequences/domain/text_sequence.dart';
import 'grade.dart';
import 'pronunciation_scorer.dart';

part 'score_pronunciation_use_case.freezed.dart';

/// Parameters for scoring a pronunciation attempt.
@freezed
class ScorePronunciationParams with _$ScorePronunciationParams {
  const factory ScorePronunciationParams({
    required TextSequence textSequence,
    required Recording recording,
  }) = _ScorePronunciationParams;
}

/// Result of a pronunciation scoring operation.
sealed class ScorePronunciationResult {
  const ScorePronunciationResult();
}

/// Scoring completed successfully.
final class ScoringSuccess extends ScorePronunciationResult {
  final Grade grade;
  const ScoringSuccess(this.grade);
}

/// Scoring failed with an error.
final class ScoringFailure extends ScorePronunciationResult {
  final String message;
  const ScoringFailure(this.message);
}

/// Use case for scoring a user's pronunciation attempt.
///
/// Takes a text sequence and recording, returns a grade.
/// Handles errors from the scorer gracefully.
class ScorePronunciationUseCase
    extends FutureUseCase<ScorePronunciationParams, ScorePronunciationResult> {
  ScorePronunciationUseCase({required PronunciationScorer scorer})
      : _scorer = scorer;

  final PronunciationScorer _scorer;

  @override
  Future<ScorePronunciationResult> run(ScorePronunciationParams input) async {
    try {
      final grade = await _scorer.score(input.textSequence, input.recording);
      return ScoringSuccess(grade);
    } catch (e) {
      return ScoringFailure('Scoring failed: ${e.toString()}');
    }
  }
}
