import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../app/di.dart';
import '../../progress/domain/sentence_rating.dart';
import '../domain/practice_stats.dart';

/// Provider to toggle demo mode for stats (debug builds only).
final statsDemoModeProvider = StateProvider<bool>((ref) => false);

/// Provider for the stats controller.
final statsControllerProvider =
    AsyncNotifierProvider<StatsController, PracticeStats>(StatsController.new);

/// Controller for computing and managing practice statistics.
class StatsController extends AsyncNotifier<PracticeStats> {
  @override
  Future<PracticeStats> build() async {
    // In debug mode, check if demo mode is enabled
    if (kDebugMode) {
      final demoMode = ref.watch(statsDemoModeProvider);
      if (demoMode) {
        return PracticeStats.demo();
      }
    }

    try {
      final progressRepo = ref.watch(progressRepositoryProvider);
      final textSequenceRepo = ref.watch(textSequenceRepositoryProvider);

      // Get all tracked sequences
      final trackedProgress = await progressRepo.getTrackedProgress();

      // Get all attempts for rating counts and daily tracking
      final allAttempts = await progressRepo.getAllAttempts();

      // Get all sequences to map IDs to HSK levels
      final allSequences = await textSequenceRepo.getAll();
      final sequenceHskMap = <String, int>{};
      for (final seq in allSequences) {
        if (seq.hskLevel != null) {
          sequenceHskMap[seq.id] = seq.hskLevel!;
        }
      }

      // Build daily attempts from actual attempt timestamps
      final dailyAttempts = <DateTime, int>{};
      for (final attempt in allAttempts) {
        final date = DateTime(
          attempt.gradedAt.year,
          attempt.gradedAt.month,
          attempt.gradedAt.day,
        );
        dailyAttempts[date] = (dailyAttempts[date] ?? 0) + 1;
      }

      // Compute totals
      final totalAttempts = allAttempts.length;

      // Count ratings from all attempts
      int hardCount = 0;
      int almostCount = 0;
      int goodCount = 0;
      int easyCount = 0;

      for (final attempt in allAttempts) {
        switch (attempt.rating) {
          case SentenceRating.hard:
            hardCount++;
          case SentenceRating.almost:
            almostCount++;
          case SentenceRating.good:
            goodCount++;
          case SentenceRating.easy:
            easyCount++;
        }
      }

      // Calculate streak (simplified - counts consecutive days with attempts)
      final streak = _calculateStreak(dailyAttempts.keys.toList());

      // Calculate cumulative progress by HSK level
      final cumulativeProgress = _calculateCumulativeProgress(
        allAttempts
            .map(
              (a) => (
                sequenceId: a.textSequenceId,
                date: a.gradedAt,
                rating: a.rating,
              ),
            )
            .toList(),
        sequenceHskMap,
      );

      return PracticeStats(
        totalAttempts: totalAttempts,
        sequencesPracticed: trackedProgress
            .where((p) => p.attemptCount > 0)
            .length,
        hardCount: hardCount,
        almostCount: almostCount,
        goodCount: goodCount,
        easyCount: easyCount,
        currentStreak: streak.current,
        longestStreak: streak.longest,
        lastPracticeDate: dailyAttempts.keys.isNotEmpty
            ? dailyAttempts.keys.reduce((a, b) => a.isAfter(b) ? a : b)
            : null,
        dailyAttempts: dailyAttempts,
        cumulativeProgress: cumulativeProgress,
      );
    } catch (e, stackTrace) {
      // Log error for debugging, return empty stats to prevent crash
      debugPrint('StatsController.build() error: $e\n$stackTrace');
      return PracticeStats.empty;
    }
  }

  ({int current, int longest}) _calculateStreak(List<DateTime> dates) {
    if (dates.isEmpty) return (current: 0, longest: 0);

    // Sort oldest to newest for easier processing
    final sortedDates = dates.toList()..sort();
    final today = DateTime.now();
    final todayDate = DateTime(today.year, today.month, today.day);

    int currentStreak = 0;
    int longestStreak = 0;
    int tempStreak = 1;

    // Calculate all streaks
    for (int i = 1; i < sortedDates.length; i++) {
      final diff = sortedDates[i].difference(sortedDates[i - 1]).inDays;
      if (diff == 1) {
        tempStreak++;
      } else if (diff > 1) {
        longestStreak = tempStreak > longestStreak ? tempStreak : longestStreak;
        tempStreak = 1;
      }
      // diff == 0 means same day, ignore
    }
    longestStreak = tempStreak > longestStreak ? tempStreak : longestStreak;

    // Calculate current streak (must include today or yesterday)
    final mostRecent = sortedDates.last;
    final daysSinceLast = todayDate.difference(mostRecent).inDays;

    if (daysSinceLast <= 1) {
      // Count backwards from most recent
      currentStreak = 1;
      for (int i = sortedDates.length - 2; i >= 0; i--) {
        final diff = sortedDates[i + 1].difference(sortedDates[i]).inDays;
        if (diff == 1) {
          currentStreak++;
        } else if (diff > 1) {
          break;
        }
        // diff == 0 means same day, continue checking
      }
    }

    return (current: currentStreak, longest: longestStreak);
  }

  /// Calculates cumulative mastered sentences per HSK level over time.
  /// A sentence is "mastered" when it receives a 'good' or 'easy' rating.
  List<CumulativeDataPoint> _calculateCumulativeProgress(
    List<({String sequenceId, DateTime date, SentenceRating rating})> attempts,
    Map<String, int> sequenceHskMap,
  ) {
    if (attempts.isEmpty) return [];

    // Sort attempts by date
    final sortedAttempts = attempts.toList()
      ..sort((a, b) => a.date.compareTo(b.date));

    // Track mastered sequences per HSK level
    final masteredByLevel = <int, Set<String>>{
      for (int i = 1; i <= 6; i++) i: {},
    };

    // Group attempts by date
    final attemptsByDate =
        <DateTime, List<({String sequenceId, SentenceRating rating})>>{};
    for (final attempt in sortedAttempts) {
      final date = DateTime(
        attempt.date.year,
        attempt.date.month,
        attempt.date.day,
      );
      attemptsByDate.putIfAbsent(date, () => []);
      attemptsByDate[date]!.add((
        sequenceId: attempt.sequenceId,
        rating: attempt.rating,
      ));
    }

    // Generate data points from first day to today
    final sortedDates = attemptsByDate.keys.toList()..sort();
    if (sortedDates.isEmpty) return [];

    final firstDate = sortedDates.first;
    final today = DateTime.now();
    final todayDate = DateTime(today.year, today.month, today.day);

    final dataPoints = <CumulativeDataPoint>[];
    var currentDate = firstDate;

    while (!currentDate.isAfter(todayDate)) {
      // Process attempts for this date
      final dayAttempts = attemptsByDate[currentDate] ?? [];
      for (final attempt in dayAttempts) {
        final hskLevel = sequenceHskMap[attempt.sequenceId];
        if (hskLevel == null) continue;

        // Mark as mastered if good or easy
        if (attempt.rating == SentenceRating.good ||
            attempt.rating == SentenceRating.easy) {
          masteredByLevel[hskLevel]!.add(attempt.sequenceId);
        }
      }

      // Create data point with current cumulative counts
      final countsByLevel = <int, int>{};
      var total = 0;
      for (int level = 1; level <= 6; level++) {
        final count = masteredByLevel[level]!.length;
        countsByLevel[level] = count;
        total += count;
      }

      dataPoints.add(
        CumulativeDataPoint(
          date: currentDate,
          countsByLevel: countsByLevel,
          total: total,
        ),
      );

      currentDate = currentDate.add(const Duration(days: 1));
    }

    return dataPoints;
  }
}
