import 'package:flutter/material.dart';

import '../../../../app/theme.dart';

/// A GitHub-style activity heatmap showing practice frequency.
class ActivityHeatmap extends StatelessWidget {
  const ActivityHeatmap({
    super.key,
    required this.dailyAttempts,
    this.weeks = 13,
  });

  /// Map of date to attempt count.
  final Map<DateTime, int> dailyAttempts;

  /// Number of weeks to display (default 13 = ~3 months).
  final int weeks;

  @override
  Widget build(BuildContext context) {
    final today = DateTime.now();
    final startDate = today.subtract(Duration(days: weeks * 7 - 1));

    // Normalize start to beginning of week (Monday)
    final mondayOffset = startDate.weekday - 1;
    final adjustedStart = startDate.subtract(Duration(days: mondayOffset));

    // Generate all dates
    final totalDays = weeks * 7;
    final dates = List.generate(
      totalDays,
      (i) => adjustedStart.add(Duration(days: i)),
    );

    // Find max attempts for color scaling
    final maxAttempts = dailyAttempts.values.fold(0, (a, b) => a > b ? a : b);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Month labels
        _buildMonthLabels(adjustedStart, weeks),
        const SizedBox(height: 4),
        Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Day of week labels
            _buildDayLabels(),
            const SizedBox(width: 4),
            // Heatmap grid
            Expanded(
              child: _buildGrid(dates, maxAttempts, today),
            ),
          ],
        ),
        const SizedBox(height: 8),
        // Legend
        _buildLegend(),
      ],
    );
  }

  Widget _buildMonthLabels(DateTime start, int weeks) {
    final months = <String>[];
    final positions = <int>[];
    String? lastMonth;

    for (int week = 0; week < weeks; week++) {
      final date = start.add(Duration(days: week * 7));
      final monthName = _monthName(date.month);
      if (monthName != lastMonth) {
        months.add(monthName);
        positions.add(week);
        lastMonth = monthName;
      }
    }

    return SizedBox(
      height: 16,
      child: Row(
        children: [
          const SizedBox(width: 24), // Space for day labels
          Expanded(
            child: Stack(
              children: [
                for (int i = 0; i < months.length; i++)
                  Positioned(
                    left: positions[i] * 12.0,
                    child: Text(
                      months[i],
                      style: const TextStyle(
                        fontSize: 10,
                        color: AppTheme.subtle,
                      ),
                    ),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDayLabels() {
    const days = ['M', '', 'W', '', 'F', '', 'S'];
    return Column(
      children: days.map((day) {
        return SizedBox(
          height: 11,
          width: 20,
          child: Text(
            day,
            style: const TextStyle(fontSize: 9, color: AppTheme.subtle),
          ),
        );
      }).toList(),
    );
  }

  Widget _buildGrid(List<DateTime> dates, int maxAttempts, DateTime today) {
    // Group dates by week
    final weekGroups = <List<DateTime>>[];
    for (int i = 0; i < dates.length; i += 7) {
      weekGroups.add(dates.sublist(i, i + 7));
    }

    return Row(
      mainAxisAlignment: MainAxisAlignment.start,
      children: weekGroups.map((week) {
        return Column(
          children: week.map((date) {
            final normalizedDate = DateTime(date.year, date.month, date.day);
            final attempts = dailyAttempts[normalizedDate] ?? 0;
            final isFuture = date.isAfter(today);

            return Tooltip(
              message: isFuture
                  ? ''
                  : '${_formatDate(date)}: $attempts attempts',
              child: Container(
                width: 10,
                height: 10,
                margin: const EdgeInsets.all(0.5),
                decoration: BoxDecoration(
                  color: isFuture
                      ? Colors.transparent
                      : _getColor(attempts, maxAttempts),
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
            );
          }).toList(),
        );
      }).toList(),
    );
  }

  Widget _buildLegend() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.end,
      children: [
        const Text(
          'Less',
          style: TextStyle(fontSize: 10, color: AppTheme.subtle),
        ),
        const SizedBox(width: 4),
        for (int i = 0; i <= 4; i++)
          Container(
            width: 10,
            height: 10,
            margin: const EdgeInsets.symmetric(horizontal: 1),
            decoration: BoxDecoration(
              color: _getLegendColor(i),
              borderRadius: BorderRadius.circular(2),
            ),
          ),
        const SizedBox(width: 4),
        const Text(
          'More',
          style: TextStyle(fontSize: 10, color: AppTheme.subtle),
        ),
      ],
    );
  }

  Color _getColor(int attempts, int maxAttempts) {
    if (attempts == 0) {
      return const Color(0xFF1a1a1a); // Dark gray for no activity
    }
    if (maxAttempts == 0) return const Color(0xFF1a1a1a);

    // Scale from light green to dark green based on activity level
    final ratio = attempts / maxAttempts;
    if (ratio >= 0.75) return const Color(0xFF39d353); // Brightest
    if (ratio >= 0.50) return const Color(0xFF26a641);
    if (ratio >= 0.25) return const Color(0xFF006d32);
    return const Color(0xFF0e4429); // Dimmest active
  }

  Color _getLegendColor(int level) {
    switch (level) {
      case 0:
        return const Color(0xFF1a1a1a);
      case 1:
        return const Color(0xFF0e4429);
      case 2:
        return const Color(0xFF006d32);
      case 3:
        return const Color(0xFF26a641);
      case 4:
        return const Color(0xFF39d353);
      default:
        return const Color(0xFF1a1a1a);
    }
  }

  String _monthName(int month) {
    const names = [
      '',
      'Jan',
      'Feb',
      'Mar',
      'Apr',
      'May',
      'Jun',
      'Jul',
      'Aug',
      'Sep',
      'Oct',
      'Nov',
      'Dec'
    ];
    return names[month];
  }

  String _formatDate(DateTime date) {
    return '${date.month}/${date.day}/${date.year}';
  }
}
