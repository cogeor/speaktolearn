import 'package:freezed_annotation/freezed_annotation.dart';

part 'score_attempt.freezed.dart';
part 'score_attempt.g.dart';

@freezed
class ScoreAttempt with _$ScoreAttempt {
  const factory ScoreAttempt({
    required String id,
    required String textSequenceId,
    required DateTime gradedAt,
    required int score,
    required String method,
    String? recognizedText,
    Map<String, dynamic>? details,
  }) = _ScoreAttempt;

  factory ScoreAttempt.fromJson(Map<String, dynamic> json) =>
      _$ScoreAttemptFromJson(json);
}
