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

      // Get all tracked sequences
      final trackedProgress = await progressRepo.getTrackedProgress();

      // Get all attempts for rating counts and daily tracking
      final allAttempts = await progressRepo.getAllAttempts();

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
}
