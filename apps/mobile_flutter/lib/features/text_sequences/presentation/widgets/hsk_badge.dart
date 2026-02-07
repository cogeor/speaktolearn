import 'package:flutter/material.dart';

/// A badge displaying the HSK level of a text sequence.
///
/// HSK (Hanyu Shuiping Kaoshi) is the standardized Chinese proficiency test.
/// Levels range from 1 (beginner) to 6 (advanced).
class HskBadge extends StatelessWidget {
  /// Creates an HskBadge.
  const HskBadge({
    super.key,
    required this.level,
    this.compact = false,
  });

  /// The HSK level (1-6).
  final int level;

  /// If true, shows just the number without "HSK" prefix.
  final bool compact;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: EdgeInsets.symmetric(
        horizontal: compact ? 6 : 8,
        vertical: 2,
      ),
      decoration: BoxDecoration(
        color: _levelColor.withValues(alpha: 0.2),
        borderRadius: BorderRadius.circular(4),
        border: Border.all(
          color: _levelColor.withValues(alpha: 0.5),
          width: 1,
        ),
      ),
      child: Text(
        compact ? '$level' : 'HSK $level',
        style: TextStyle(
          color: _levelColor,
          fontSize: compact ? 10 : 11,
          fontWeight: FontWeight.w600,
        ),
      ),
    );
  }

  /// Returns the color associated with the HSK level.
  /// Gradient from green (easy) to red (hard).
  Color get _levelColor {
    switch (level) {
      case 1:
        return const Color(0xFF4CAF50); // Green
      case 2:
        return const Color(0xFF8BC34A); // Light green
      case 3:
        return const Color(0xFFCDDC39); // Lime
      case 4:
        return const Color(0xFFFFEB3B); // Yellow
      case 5:
        return const Color(0xFFFF9800); // Orange
      case 6:
        return const Color(0xFFF44336); // Red
      default:
        return const Color(0xFF888888); // Gray for unknown
    }
  }
}
