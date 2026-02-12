import 'package:freezed_annotation/freezed_annotation.dart';

part 'practice_stats.freezed.dart';

/// A data point for cumulative progress at a specific date.
class CumulativeDataPoint {
  const CumulativeDataPoint({
    required this.date,
    required this.countsByLevel,
    required this.total,
  });

  final DateTime date;
  /// Counts per HSK level (1-6)
  final Map<int, int> countsByLevel;
  final int total;
}

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
    /// Cumulative mastered sentences over time by HSK level.
    /// Each data point contains counts per HSK level and total.
    @Default([]) List<CumulativeDataPoint> cumulativeProgress,
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

    // Generate cumulative progress data
    final cumulativeProgress = _generateDemoCumulativeProgress(today);

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
      cumulativeProgress: cumulativeProgress,
    );
  }

  /// Generates demo cumulative progress data showing growth over time.
  static List<CumulativeDataPoint> _generateDemoCumulativeProgress(
    DateTime today,
  ) {
    final dataPoints = <CumulativeDataPoint>[];

    // Start 60 days ago with some initial mastered sentences
    // HSK1: 8, HSK2: 5, HSK3: 3, HSK4: 1, HSK5: 0, HSK6: 0
    final baseCounts = {1: 8, 2: 5, 3: 3, 4: 1, 5: 0, 6: 0};

    // Growth rates per level (lower levels grow faster)
    final growthRates = {1: 0.3, 2: 0.25, 3: 0.2, 4: 0.15, 5: 0.1, 6: 0.05};

    for (int i = 60; i >= 0; i--) {
      final date = today.subtract(Duration(days: i));
      final daysSinceStart = 60 - i;

      final countsByLevel = <int, int>{};
      var total = 0;

      for (int level = 1; level <= 6; level++) {
        // Cumulative growth with some randomness
        final growth = (daysSinceStart * growthRates[level]!).round();
        final count = baseCounts[level]! + growth;
        countsByLevel[level] = count;
        total += count;
      }

      dataPoints.add(CumulativeDataPoint(
        date: date,
        countsByLevel: countsByLevel,
        total: total,
      ));
    }

    return dataPoints;
  }
}
