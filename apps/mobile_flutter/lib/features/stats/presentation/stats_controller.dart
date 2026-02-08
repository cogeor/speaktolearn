import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../app/di.dart';
import '../domain/practice_stats.dart';

/// Provider for the stats controller.
final statsControllerProvider =
    AsyncNotifierProvider<StatsController, PracticeStats>(StatsController.new);

/// Controller for computing and managing practice statistics.
class StatsController extends AsyncNotifier<PracticeStats> {
  @override
  Future<PracticeStats> build() async {
    final progressRepo = ref.watch(progressRepositoryProvider);

    // Get all tracked sequences
    final trackedProgress = await progressRepo.getTrackedProgress();

    // Get all attempts for accurate average calculation
    final allAttempts = await progressRepo.getAllAttempts();

    // Compute totals
    int totalAttempts = 0;
    final dailyAttempts = <DateTime, int>{};

    for (final progress in trackedProgress) {
      totalAttempts += progress.attemptCount;

      // Track daily attempts for heatmap
      if (progress.lastAttemptAt != null) {
        final date = DateTime(
          progress.lastAttemptAt!.year,
          progress.lastAttemptAt!.month,
          progress.lastAttemptAt!.day,
        );
        dailyAttempts[date] =
            (dailyAttempts[date] ?? 0) + progress.attemptCount;
      }
    }

    // Calculate average from all actual attempts (not just best scores)
    final double? averageScore;
    if (allAttempts.isNotEmpty) {
      final totalScore = allAttempts.fold<int>(0, (sum, a) => sum + a.score);
      averageScore = totalScore / allAttempts.length;
    } else {
      averageScore = null;
    }

    // Calculate streak (simplified - counts consecutive days with attempts)
    final streak = _calculateStreak(dailyAttempts.keys.toList());

    return PracticeStats(
      totalAttempts: totalAttempts,
      sequencesPracticed: trackedProgress
          .where((p) => p.attemptCount > 0)
          .length,
      averageScore: averageScore,
      currentStreak: streak.current,
      longestStreak: streak.longest,
      lastPracticeDate: dailyAttempts.keys.isNotEmpty
          ? dailyAttempts.keys.reduce((a, b) => a.isAfter(b) ? a : b)
          : null,
      dailyAttempts: dailyAttempts,
    );
  }

  ({int current, int longest}) _calculateStreak(List<DateTime> dates) {
    if (dates.isEmpty) return (current: 0, longest: 0);

    dates.sort((a, b) => b.compareTo(a)); // Most recent first
    final today = DateTime.now();
    final todayDate = DateTime(today.year, today.month, today.day);

    int currentStreak = 0;
    int longestStreak = 0;
    int tempStreak = 0;
    DateTime? lastDate;

    for (final date in dates) {
      if (lastDate == null) {
        // Check if streak is active (practiced today or yesterday)
        final diff = todayDate.difference(date).inDays;
        if (diff <= 1) {
          currentStreak = 1;
          tempStreak = 1;
        }
      } else {
        final diff = lastDate.difference(date).inDays;
        if (diff == 1) {
          tempStreak++;
          if (currentStreak > 0) currentStreak++;
        } else {
          longestStreak = tempStreak > longestStreak
              ? tempStreak
              : longestStreak;
          tempStreak = 1;
          currentStreak = 0;
        }
      }
      lastDate = date;
    }

    longestStreak = tempStreak > longestStreak ? tempStreak : longestStreak;

    return (current: currentStreak, longest: longestStreak);
  }
}
