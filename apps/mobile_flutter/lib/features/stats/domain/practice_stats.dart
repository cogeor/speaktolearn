import 'package:freezed_annotation/freezed_annotation.dart';

part 'practice_stats.freezed.dart';

/// Aggregate practice statistics for display.
@freezed
class PracticeStats with _$PracticeStats {
  const factory PracticeStats({
    @Default(0) int totalAttempts,
    @Default(0) int sequencesPracticed,
    @Default(0) int hardCount,
    @Default(0) int almostCount,
    @Default(0) int goodCount,
    @Default(0) int easyCount,
    @Default(0) int currentStreak,
    @Default(0) int longestStreak,
    DateTime? lastPracticeDate,
    @Default({}) Map<DateTime, int> dailyAttempts,
  }) = _PracticeStats;

  /// Creates an empty stats instance (default state).
  static const PracticeStats empty = PracticeStats();

  /// Creates mock/demo stats for testing and development.
  /// Generates 2 months of realistic practice data.
  static PracticeStats demo() {
    final now = DateTime.now();
    final today = DateTime(now.year, now.month, now.day);
    final dailyAttempts = <DateTime, int>{};

    // Generate ~60 days of practice data with realistic patterns
    // More practice on weekends, occasional gaps
    for (int i = 0; i < 60; i++) {
      final date = today.subtract(Duration(days: i));
      final dayOfWeek = date.weekday;

      // Skip some random days to simulate gaps (about 20% chance)
      if ((date.day * 7 + i) % 5 == 0) continue;

      // Weekend days tend to have more practice
      final isWeekend =
          dayOfWeek == DateTime.saturday || dayOfWeek == DateTime.sunday;
      final baseAttempts = isWeekend ? 8 : 4;

      // Add some variation based on day
      final variation = (date.day % 5) - 2;
      final attempts = (baseAttempts + variation).clamp(1, 15);

      dailyAttempts[date] = attempts;
    }

    // Calculate totals from the generated data
    final totalAttempts = dailyAttempts.values.fold<int>(0, (a, b) => a + b);

    // Generate realistic rating distribution
    // Most ratings are good/easy, fewer hard/almost (shows improvement)
    final hardCount = (totalAttempts * 0.1).round();
    final almostCount = (totalAttempts * 0.2).round();
    final goodCount = (totalAttempts * 0.4).round();
    final easyCount = totalAttempts - hardCount - almostCount - goodCount;

    return PracticeStats(
      totalAttempts: totalAttempts,
      sequencesPracticed: (totalAttempts * 0.7).round(),
      hardCount: hardCount,
      almostCount: almostCount,
      goodCount: goodCount,
      easyCount: easyCount,
      currentStreak: 5,
      longestStreak: 14,
      lastPracticeDate: today,
      dailyAttempts: dailyAttempts,
    );
  }
}
