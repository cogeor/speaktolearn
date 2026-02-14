import 'package:freezed_annotation/freezed_annotation.dart';

part 'grade.freezed.dart';
part 'grade.g.dart';

@freezed
class Grade with _$Grade {
  const factory Grade({
    /// Overall pronunciation score (0-100).
    required int overall,

    /// Scoring method identifier (e.g., 'asr_cer_v1').
    required String method,

    /// Accuracy score derived from CER (0-100).
    /// Represents how precisely the pronunciation matched expected phonemes.
    /// Higher is better. Null if not calculated.
    int? accuracy,

    /// Completeness score (0-100).
    /// Represents what percentage of the reference text was spoken.
    /// Higher is better. Null if not calculated.
    int? completeness,

    /// The text recognized by the speech recognizer.
    String? recognizedText,

    /// Additional details about the scoring (CER, edit distance, etc.).
    Map<String, dynamic>? details,

    /// Per-character scores from ML model (0.0-1.0 each).
    /// Length should match text.characters.length when present.
    List<double>? characterScores,
  }) = _Grade;

  factory Grade.fromJson(Map<String, dynamic> json) => _$GradeFromJson(json);
}
