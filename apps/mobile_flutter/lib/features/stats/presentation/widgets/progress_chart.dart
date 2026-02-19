import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';

import '../../domain/practice_stats.dart';

/// Colors for each HSK level line in the chart.
const _hskColors = <int, Color>{
  1: Color(0xFF4CAF50), // Green
  2: Color(0xFF2196F3), // Blue
  3: Color(0xFFFF9800), // Orange
  4: Color(0xFF9C27B0), // Purple
  5: Color(0xFFE91E63), // Pink
  6: Color(0xFF00BCD4), // Cyan
};

const _totalColor = Color(0xFF424242); // Dark gray for total

/// A line chart showing cumulative mastered sentences over time.
class ProgressChart extends StatelessWidget {
  const ProgressChart({super.key, required this.dataPoints});

  final List<CumulativeDataPoint> dataPoints;

  @override
  Widget build(BuildContext context) {
    if (dataPoints.isEmpty) {
      return const SizedBox(
        height: 200,
        child: Center(
          child: Text(
            'No progress data yet',
            style: TextStyle(color: Colors.grey),
          ),
        ),
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Mastered Sentences',
          style: TextStyle(fontSize: 16, fontWeight: FontWeight.w500),
        ),
        const SizedBox(height: 8),
        SizedBox(height: 200, child: LineChart(_buildChartData(context))),
        const SizedBox(height: 8),
        _buildLegend(),
      ],
    );
  }

  LineChartData _buildChartData(BuildContext context) {
    // Convert data points to line spots
    final lines = <int, List<FlSpot>>{
      for (int level = 1; level <= 6; level++) level: [],
    };
    final totalLine = <FlSpot>[];

    for (int i = 0; i < dataPoints.length; i++) {
      final point = dataPoints[i];
      final x = i.toDouble();

      for (int level = 1; level <= 6; level++) {
        final count = point.countsByLevel[level] ?? 0;
        lines[level]!.add(FlSpot(x, count.toDouble()));
      }
      totalLine.add(FlSpot(x, point.total.toDouble()));
    }

    // Find max Y value for scaling
    final maxY = dataPoints.isEmpty
        ? 10.0
        : dataPoints
              .map((p) => p.total)
              .reduce((a, b) => a > b ? a : b)
              .toDouble();
    final yInterval = _calculateYInterval(maxY);

    return LineChartData(
      gridData: FlGridData(
        show: true,
        drawVerticalLine: false,
        horizontalInterval: yInterval,
        getDrawingHorizontalLine: (value) =>
            FlLine(color: Colors.grey.withValues(alpha: 0.2), strokeWidth: 1),
      ),
      titlesData: FlTitlesData(
        leftTitles: AxisTitles(
          sideTitles: SideTitles(
            showTitles: true,
            reservedSize: 40,
            interval: yInterval,
            getTitlesWidget: (value, meta) {
              return Text(
                value.toInt().toString(),
                style: const TextStyle(fontSize: 10, color: Colors.grey),
              );
            },
          ),
        ),
        bottomTitles: AxisTitles(
          sideTitles: SideTitles(
            showTitles: true,
            reservedSize: 22,
            interval: _calculateXInterval(),
            getTitlesWidget: (value, meta) {
              final index = value.toInt();
              if (index < 0 || index >= dataPoints.length) {
                return const SizedBox.shrink();
              }
              final date = dataPoints[index].date;
              return Text(
                '${date.month}/${date.day}',
                style: const TextStyle(fontSize: 9, color: Colors.grey),
              );
            },
          ),
        ),
        topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
        rightTitles: const AxisTitles(
          sideTitles: SideTitles(showTitles: false),
        ),
      ),
      borderData: FlBorderData(show: false),
      minX: 0,
      maxX: (dataPoints.length - 1).toDouble(),
      minY: 0,
      maxY: maxY * 1.1, // Add 10% padding
      lineBarsData: [
        // Total line (thicker, dashed)
        LineChartBarData(
          spots: totalLine,
          isCurved: true,
          curveSmoothness: 0.2,
          color: _totalColor,
          barWidth: 3,
          dotData: const FlDotData(show: false),
          dashArray: [5, 3],
        ),
        // HSK level lines
        for (int level = 1; level <= 6; level++)
          LineChartBarData(
            spots: lines[level]!,
            isCurved: true,
            curveSmoothness: 0.2,
            color: _hskColors[level]!,
            barWidth: 2,
            dotData: const FlDotData(show: false),
          ),
      ],
      lineTouchData: LineTouchData(
        touchTooltipData: LineTouchTooltipData(
          getTooltipColor: (spot) => Colors.black87,
          getTooltipItems: (touchedSpots) {
            return touchedSpots.map((spot) {
              final levelIndex = spot.barIndex;
              String label;
              Color color;

              if (levelIndex == 0) {
                label = 'Total';
                color = _totalColor;
              } else {
                label = 'HSK$levelIndex';
                color = _hskColors[levelIndex]!;
              }

              return LineTooltipItem(
                '$label: ${spot.y.toInt()}',
                TextStyle(color: color, fontSize: 12),
              );
            }).toList();
          },
        ),
      ),
    );
  }

  double _calculateXInterval() {
    if (dataPoints.length <= 7) return 1;
    if (dataPoints.length <= 14) return 2;
    if (dataPoints.length <= 30) return 5;
    return 10;
  }

  double _calculateYInterval(double maxY) {
    if (maxY <= 10) return 2;
    if (maxY <= 50) return 10;
    if (maxY <= 100) return 20;
    return (maxY / 5).ceilToDouble();
  }

  Widget _buildLegend() {
    return Wrap(
      spacing: 12,
      runSpacing: 4,
      children: [
        _LegendItem(color: _totalColor, label: 'Total', isDashed: true),
        for (int level = 1; level <= 6; level++)
          _LegendItem(color: _hskColors[level]!, label: 'HSK$level'),
      ],
    );
  }
}

class _LegendItem extends StatelessWidget {
  const _LegendItem({
    required this.color,
    required this.label,
    this.isDashed = false,
  });

  final Color color;
  final String label;
  final bool isDashed;

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 16,
          height: 3,
          decoration: BoxDecoration(
            color: isDashed ? null : color,
            border: isDashed
                ? Border(
                    bottom: BorderSide(
                      color: color,
                      width: 2,
                      style: BorderStyle.solid,
                    ),
                  )
                : null,
          ),
          child: isDashed
              ? CustomPaint(painter: _DashedLinePainter(color: color))
              : null,
        ),
        const SizedBox(width: 4),
        Text(label, style: const TextStyle(fontSize: 10, color: Colors.grey)),
      ],
    );
  }
}

class _DashedLinePainter extends CustomPainter {
  _DashedLinePainter({required this.color});

  final Color color;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;

    const dashWidth = 3.0;
    const dashSpace = 2.0;
    var startX = 0.0;

    while (startX < size.width) {
      canvas.drawLine(
        Offset(startX, size.height / 2),
        Offset(startX + dashWidth, size.height / 2),
        paint,
      );
      startX += dashWidth + dashSpace;
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
