import '../../recording/domain/recording.dart';
import '../../scoring/domain/grade.dart';
import '../../scoring/domain/pronunciation_scorer.dart';
import '../../text_sequences/domain/text_sequence.dart';

/// ML-based pronunciation scorer interface.
///
/// Extends [PronunciationScorer] to provide character-level scoring
/// using machine learning models.
abstract class MlScorer implements PronunciationScorer {
  /// Score pronunciation with character-level granularity.
  ///
  /// Returns a [Grade] with `characterScores` populated as a list of
  /// scores (0.0-1.0) for each character in the sequence text.
  @override
  Future<Grade> score(TextSequence sequence, Recording recording);

  /// Whether the ML model is loaded and ready for inference.
  bool get isReady;

  /// Initialize/load the ML model.
  ///
  /// Called lazily on first [score] call, or eagerly if preloading is desired.
  Future<void> initialize();

  /// Release model resources.
  Future<void> dispose();
}
