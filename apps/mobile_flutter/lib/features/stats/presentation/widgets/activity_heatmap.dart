import 'package:flutter/material.dart';

import '../../../../app/theme.dart';

/// A GitHub-style activity heatmap showing practice frequency over the past year.
class ActivityHeatmap extends StatelessWidget {
  const ActivityHeatmap({super.key, required this.dailyAttempts});

  /// Map of date to attempt count.
  final Map<DateTime, int> dailyAttempts;

  @override
  Widget build(BuildContext context) {
    final today = DateTime.now();

    // Start from 52 weeks ago, aligned to Monday
    final startDate = today.subtract(const Duration(days: 52 * 7));
    final mondayOffset = (startDate.weekday - 1) % 7;
    final adjustedStart = startDate.subtract(Duration(days: mondayOffset));

    // Generate 52 weeks of dates
    final weeks = <List<DateTime>>[];
    for (int w = 0; w < 52; w++) {
      final week = <DateTime>[];
      for (int d = 0; d < 7; d++) {
        week.add(adjustedStart.add(Duration(days: w * 7 + d)));
      }
      weeks.add(week);
    }

    return LayoutBuilder(
      builder: (context, constraints) {
        // Calculate cell size to fit 52 weeks in available width
        const dayLabelWidth = 16.0;
        final availableWidth = constraints.maxWidth - dayLabelWidth - 4;
        final cellSize = (availableWidth / 52).floorToDouble() - 1;
        final actualCellSize = cellSize.clamp(3.0, 10.0);

        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            // Month labels
            _MonthLabels(
              weeks: weeks,
              cellSize: actualCellSize,
              dayLabelWidth: dayLabelWidth,
            ),
            const SizedBox(height: 2),
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Day labels
                _DayLabels(cellSize: actualCellSize),
                const SizedBox(width: 4),
                // Grid
                _Grid(
                  weeks: weeks,
                  cellSize: actualCellSize,
                  today: today,
                  dailyAttempts: dailyAttempts,
                ),
              ],
            ),
          ],
        );
      },
    );
  }
}

class _MonthLabels extends StatelessWidget {
  const _MonthLabels({
    required this.weeks,
    required this.cellSize,
    required this.dayLabelWidth,
  });

  final List<List<DateTime>> weeks;
  final double cellSize;
  final double dayLabelWidth;

  @override
  Widget build(BuildContext context) {
    final monthLabels = <_MonthLabel>[];
    String? lastMonth;

    for (int i = 0; i < weeks.length; i++) {
      final firstDayOfWeek = weeks[i].first;
      final monthName = _monthName(firstDayOfWeek.month);

      // Show label when month changes
      if (monthName != lastMonth) {
        monthLabels.add(_MonthLabel(monthName, i));
        lastMonth = monthName;
      }
    }

    final cellWidth = cellSize + 1; // cell + gap

    return SizedBox(
      height: 14,
      child: Row(
        children: [
          SizedBox(width: dayLabelWidth + 4),
          Expanded(
            child: Stack(
              clipBehavior: Clip.none,
              children: monthLabels.map((label) {
                return Positioned(
                  left: label.weekIndex * cellWidth,
                  child: Text(
                    label.name,
                    style: const TextStyle(fontSize: 9, color: AppTheme.subtle),
                  ),
                );
              }).toList(),
            ),
          ),
        ],
      ),
    );
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
      'Dec',
    ];
    return names[month];
  }
}

class _MonthLabel {
  final String name;
  final int weekIndex;
  _MonthLabel(this.name, this.weekIndex);
}

class _DayLabels extends StatelessWidget {
  const _DayLabels({required this.cellSize});

  final double cellSize;

  @override
  Widget build(BuildContext context) {
    const labels = ['', 'M', '', 'W', '', 'F', ''];
    final cellHeight = cellSize + 1;

    return Column(
      children: labels.map((label) {
        return SizedBox(
          height: cellHeight,
          width: 16,
          child: Align(
            alignment: Alignment.centerRight,
            child: Text(
              label,
              style: const TextStyle(fontSize: 8, color: AppTheme.subtle),
            ),
          ),
        );
      }).toList(),
    );
  }
}

class _Grid extends StatelessWidget {
  const _Grid({
    required this.weeks,
    required this.cellSize,
    required this.today,
    required this.dailyAttempts,
  });

  final List<List<DateTime>> weeks;
  final double cellSize;
  final DateTime today;
  final Map<DateTime, int> dailyAttempts;

  @override
  Widget build(BuildContext context) {
    // Find max attempts for relative color scaling
    final maxAttempts = dailyAttempts.values.fold(0, (a, b) => a > b ? a : b);

    return Row(
      mainAxisSize: MainAxisSize.min,
      children: weeks.map((week) {
        return Column(
          mainAxisSize: MainAxisSize.min,
          children: week.map((date) {
            final normalized = DateTime(date.year, date.month, date.day);
            final attempts = dailyAttempts[normalized] ?? 0;
            final isFuture = date.isAfter(today);

            return Tooltip(
              message: isFuture ? '' : '${_formatDate(date)}: $attempts',
              child: Container(
                width: cellSize,
                height: cellSize,
                margin: const EdgeInsets.all(0.5),
                decoration: BoxDecoration(
                  color: isFuture
                      ? Colors.transparent
                      : _getColor(attempts, maxAttempts),
                  borderRadius: BorderRadius.circular(1),
                ),
              ),
            );
          }).toList(),
        );
      }).toList(),
    );
  }

  Color _getColor(int attempts, int maxAttempts) {
    if (attempts == 0) return const Color(0xFF161b22); // Empty
    if (maxAttempts == 0) return const Color(0xFF161b22);

    // Scale color based on ratio to max
    final ratio = attempts / maxAttempts;
    if (ratio > 0.75) return const Color(0xFF39d353); // Brightest
    if (ratio > 0.50) return const Color(0xFF26a641);
    if (ratio > 0.25) return const Color(0xFF006d32);
    return const Color(0xFF0e4429); // Dimmest green
  }

  String _formatDate(DateTime date) {
    return '${date.month}/${date.day}/${date.year}';
  }
}
