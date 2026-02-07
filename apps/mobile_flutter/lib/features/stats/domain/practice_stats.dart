import 'package:freezed_annotation/freezed_annotation.dart';

part 'practice_stats.freezed.dart';

/// Aggregate practice statistics for display.
@freezed
class PracticeStats with _$PracticeStats {
  const factory PracticeStats({
    @Default(0) int totalAttempts,
    @Default(0) int sequencesPracticed,
    double? averageScore,
    @Default(0) int currentStreak,
    @Default(0) int longestStreak,
    DateTime? lastPracticeDate,
    @Default({}) Map<DateTime, int> dailyAttempts,
  }) = _PracticeStats;
}
