import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../../app/theme.dart';
import '../domain/practice_stats.dart';
import 'stats_controller.dart';
import 'widgets/activity_heatmap.dart';

/// Screen displaying aggregate practice statistics.
class StatsScreen extends ConsumerWidget {
  const StatsScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final statsAsync = ref.watch(statsControllerProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Statistics'),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => context.go('/'),
        ),
      ),
      body: statsAsync.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (error, _) => Center(child: Text('Error: $error')),
        data: (stats) => _StatsContent(stats: stats),
      ),
    );
  }
}

class _StatsContent extends StatelessWidget {
  const _StatsContent({required this.stats});

  final PracticeStats stats;

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        // Stats grid
        GridView.count(
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          crossAxisCount: 2,
          mainAxisSpacing: 16,
          crossAxisSpacing: 16,
          childAspectRatio: 1.5,
          children: [
            _StatCard(
              icon: Icons.mic,
              label: 'Total Spoken',
              value: stats.totalAttempts.toString(),
            ),
            _StatCard(
              icon: Icons.library_books,
              label: 'Sentences Practiced',
              value: stats.sequencesPracticed.toString(),
            ),
            _StatCard(
              icon: Icons.local_fire_department,
              label: 'Current Streak',
              value: '${stats.currentStreak} days',
              valueColor: stats.currentStreak > 0 ? AppTheme.warning : null,
            ),
            _StatCard(
              icon: Icons.emoji_events,
              label: 'Longest Streak',
              value: '${stats.longestStreak} days',
              valueColor: stats.longestStreak > 0 ? AppTheme.warning : null,
            ),
          ],
        ),
        // Rating breakdown
        Padding(
          padding: const EdgeInsets.symmetric(vertical: 16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Rating Breakdown',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.w500),
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  _RatingCountCard(
                    color: AppTheme.ratingHard,
                    label: 'Hard',
                    count: stats.hardCount,
                  ),
                  const SizedBox(width: 8),
                  _RatingCountCard(
                    color: AppTheme.ratingAlmost,
                    label: 'Almost',
                    count: stats.almostCount,
                  ),
                  const SizedBox(width: 8),
                  _RatingCountCard(
                    color: AppTheme.ratingGood,
                    label: 'Good',
                    count: stats.goodCount,
                  ),
                  const SizedBox(width: 8),
                  _RatingCountCard(
                    color: AppTheme.ratingEasy,
                    label: 'Easy',
                    count: stats.easyCount,
                  ),
                ],
              ),
            ],
          ),
        ),
        // Activity heatmap
        const SizedBox(height: 24),
        const Text(
          'Activity',
          style: TextStyle(fontSize: 16, fontWeight: FontWeight.w500),
        ),
        const SizedBox(height: 12),
        ActivityHeatmap(dailyAttempts: stats.dailyAttempts),
      ],
    );
  }
}

class _StatCard extends StatelessWidget {
  const _StatCard({
    required this.icon,
    required this.label,
    required this.value,
    this.valueColor,
  });

  final IconData icon;
  final String label;
  final String value;
  final Color? valueColor;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: FittedBox(
          fit: BoxFit.scaleDown,
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(icon, size: 28),
              const SizedBox(height: 8),
              Text(
                value,
                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                  color: valueColor,
                  fontWeight: FontWeight.bold,
                ),
              ),
              Text(
                label,
                style: const TextStyle(color: AppTheme.subtle, fontSize: 12),
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _RatingCountCard extends StatelessWidget {
  const _RatingCountCard({
    required this.color,
    required this.label,
    required this.count,
  });

  final Color color;
  final String label;
  final int count;

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.2),
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: color, width: 2),
        ),
        child: Column(
          children: [
            Text(
              count.toString(),
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
                color: color,
              ),
            ),
            Text(label, style: const TextStyle(fontSize: 12)),
          ],
        ),
      ),
    );
  }
}
