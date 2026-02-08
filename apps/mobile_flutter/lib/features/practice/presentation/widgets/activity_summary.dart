import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../../app/di.dart';

/// Displays a summary of practice activity for the current week.
///
/// Shows the number of practice attempts made this week.
class ActivitySummary extends ConsumerWidget {
  const ActivitySummary({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return FutureBuilder<int>(
      future: _getWeeklyAttemptCount(ref),
      builder: (context, snapshot) {
        final count = snapshot.data ?? 0;
        final isLoading = snapshot.connectionState == ConnectionState.waiting;

        return Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(
                      Icons.trending_up,
                      size: 20,
                      color: Theme.of(context).colorScheme.primary,
                    ),
                    const SizedBox(width: 8),
                    Text(
                      'Activity this week',
                      style: Theme.of(context).textTheme.titleSmall?.copyWith(
                            color: Theme.of(context).colorScheme.onSurfaceVariant,
                          ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                if (isLoading)
                  const SizedBox(
                    height: 32,
                    child: Center(
                      child: SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      ),
                    ),
                  )
                else
                  Row(
                    crossAxisAlignment: CrossAxisAlignment.baseline,
                    textBaseline: TextBaseline.alphabetic,
                    children: [
                      Text(
                        '$count',
                        style: Theme.of(context).textTheme.headlineLarge?.copyWith(
                              fontWeight: FontWeight.bold,
                              color: Theme.of(context).colorScheme.primary,
                            ),
                      ),
                      const SizedBox(width: 8),
                      Text(
                        count == 1 ? 'practice session' : 'practice sessions',
                        style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                              color: Theme.of(context).colorScheme.onSurfaceVariant,
                            ),
                      ),
                    ],
                  ),
              ],
            ),
          ),
        );
      },
    );
  }

  /// Calculates the number of practice attempts in the current week.
  Future<int> _getWeeklyAttemptCount(WidgetRef ref) async {
    final progressRepo = ref.read(progressRepositoryProvider);
    final allAttempts = await progressRepo.getAllAttempts();

    final now = DateTime.now();
    final today = DateTime(now.year, now.month, now.day);
    // Start of week (Monday)
    final startOfWeek = today.subtract(Duration(days: today.weekday - 1));

    int count = 0;
    for (final attempt in allAttempts) {
      final attemptDate = DateTime(
        attempt.gradedAt.year,
        attempt.gradedAt.month,
        attempt.gradedAt.day,
      );
      if (!attemptDate.isBefore(startOfWeek)) {
        count++;
      }
    }

    return count;
  }
}
