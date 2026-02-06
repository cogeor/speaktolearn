import 'package:freezed_annotation/freezed_annotation.dart';

part 'text_sequence_progress.freezed.dart';
part 'text_sequence_progress.g.dart';

@freezed
class TextSequenceProgress with _$TextSequenceProgress {
  const TextSequenceProgress._();

  const factory TextSequenceProgress({
    required bool tracked,
    int? bestScore,
    String? bestAttemptId,
    DateTime? lastAttemptAt,
    @Default(0) int attemptCount,
  }) = _TextSequenceProgress;

  factory TextSequenceProgress.fromJson(Map<String, dynamic> json) =>
      _$TextSequenceProgressFromJson(json);

  factory TextSequenceProgress.initial() =>
      const TextSequenceProgress(tracked: false);

  bool get hasAttempts => attemptCount > 0;

  double? get hoursSinceLastAttempt {
    if (lastAttemptAt == null) return null;
    return DateTime.now().difference(lastAttemptAt!).inMinutes / 60.0;
  }
}
