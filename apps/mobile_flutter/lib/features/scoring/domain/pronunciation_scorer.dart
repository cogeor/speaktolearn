import '../../recording/domain/recording.dart';
import '../../text_sequences/domain/text_sequence.dart';
import 'grade.dart';

/// Interface for pronunciation scoring implementations.
abstract class PronunciationScorer {
  /// Scores a user's pronunciation attempt.
  ///
  /// Takes a [TextSequence] with the expected text and a [Recording] of the
  /// user's attempt. Returns a [Grade] with the scoring results.
  Future<Grade> score(TextSequence sequence, Recording recording);
}
