import '../domain/ml_scorer.dart';
import 'onnx_ml_scorer_v4.dart';
import 'onnx_ml_scorer_v5.dart';
import 'onnx_ml_scorer_v6.dart';
import 'onnx_ml_scorer_v7.dart';

/// Model version enum for scorer selection.
enum ModelVersion {
  /// V4 architecture: 1s audio chunks, pinyin_ids for position
  v4,

  /// V5 architecture: full sentence audio, position embedding
  v5,

  /// V6 architecture: full sentence audio with sliding window attention
  v6,

  /// V7 architecture: CTC-based per-frame predictions, single pass inference
  v7,
}

/// Factory for creating ML scorers based on model version.
///
/// Usage:
/// ```dart
/// final scorer = MlScorerFactory.create(ModelVersion.v4);
/// await scorer.initialize();
/// final grade = await scorer.score(sequence, recording);
/// ```
class MlScorerFactory {
  MlScorerFactory._();

  /// Current default model version.
  ///
  /// Change this to switch the default model used by the app.
  static const defaultVersion = ModelVersion.v6;

  /// Create an ML scorer for the specified model version.
  static MlScorer create([ModelVersion version = defaultVersion]) {
    switch (version) {
      case ModelVersion.v4:
        return OnnxMlScorerV4();
      case ModelVersion.v5:
        return OnnxMlScorerV5();
      case ModelVersion.v6:
        return OnnxMlScorerV6();
      case ModelVersion.v7:
        return OnnxMlScorerV7();
    }
  }

  /// Create the default scorer.
  static MlScorer createDefault() => create(defaultVersion);
}
