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
    if (stats.totalAttempts == 0) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(32),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(
                Icons.bar_chart_outlined,
                size: 80,
                color: AppTheme.subtle,
              ),
              const SizedBox(height: 24),
              Text(
                'No practice data yet',
                style: Theme.of(context).textTheme.titleMedium,
              ),
              const SizedBox(height: 8),
              const Text(
                'Complete some practice sessions to see your statistics here.',
                textAlign: TextAlign.center,
                style: TextStyle(color: AppTheme.subtle),
              ),
              const SizedBox(height: 32),
              OutlinedButton.icon(
                onPressed: () => context.go('/'),
                icon: const Icon(Icons.play_arrow),
                label: const Text('Start Practicing'),
              ),
            ],
          ),
        ),
      );
    }

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
              label: 'Total Attempts',
              value: stats.totalAttempts.toString(),
            ),
            _StatCard(
              icon: Icons.library_books,
              label: 'Sequences Practiced',
              value: stats.sequencesPracticed.toString(),
            ),
            _StatCard(
              icon: Icons.grade,
              label: 'Average Score',
              value: stats.averageScore != null
                  ? '${stats.averageScore!.toStringAsFixed(0)}%'
                  : '-',
              valueColor: stats.averageScore?.round().scoreColor,
            ),
            _StatCard(
              icon: Icons.local_fire_department,
              label: 'Current Streak',
              value: '${stats.currentStreak} days',
              valueColor: stats.currentStreak > 0 ? AppTheme.warning : null,
            ),
          ],
        ),
        const SizedBox(height: 24),
        // Streak info
        if (stats.longestStreak > 0)
          Card(
            child: ListTile(
              leading: const Icon(Icons.emoji_events, color: AppTheme.warning),
              title: const Text('Longest Streak'),
              trailing: Text(
                '${stats.longestStreak} days',
                style: Theme.of(context).textTheme.titleMedium,
              ),
            ),
          ),
        // Activity heatmap
        const SizedBox(height: 24),
        const Text(
          'Activity',
          style: TextStyle(fontSize: 16, fontWeight: FontWeight.w500),
        ),
        const SizedBox(height: 12),
        ActivityHeatmap(dailyAttempts: stats.dailyAttempts, weeks: 13),
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
    );
  }
}
