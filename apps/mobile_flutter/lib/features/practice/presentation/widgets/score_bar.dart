import 'package:flutter/material.dart';

import '../../../../app/theme.dart';

/// A horizontal progress bar that displays a pronunciation score with color coding.
///
/// Colors:
/// - 0-49: error (red)
/// - 50-79: warning (yellow)
/// - 80-100: success (green)
/// - null: subtle gray (empty state)
class ScoreBar extends StatelessWidget {
  const ScoreBar({
    super.key,
    required this.score,
    this.height = 8.0,
    this.showLabel = false,
    this.borderRadius = 4.0,
  });

  /// The score value (0-100), or null for empty state.
  final int? score;

  /// Height of the progress bar in logical pixels.
  final double height;

  /// Whether to show the percentage label above the bar.
  final bool showLabel;

  /// Border radius for rounded corners.
  final double borderRadius;

  Color get _barColor {
    if (score == null) return AppTheme.subtle;
    return score!.scoreColor;
  }

  double get _progress {
    if (score == null) return 0.0;
    return (score!.clamp(0, 100)) / 100.0;
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        if (showLabel) ...[
          Text(
            score != null ? '$score%' : '-',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
              color: _barColor,
              fontWeight: FontWeight.w500,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 4),
        ],
        Container(
          height: height,
          decoration: BoxDecoration(
            color: AppTheme.subtle.withValues(alpha: 0.3),
            borderRadius: BorderRadius.circular(borderRadius),
          ),
          child: FractionallySizedBox(
            alignment: Alignment.centerLeft,
            widthFactor: _progress,
            child: Container(
              decoration: BoxDecoration(
                color: _barColor,
                borderRadius: BorderRadius.circular(borderRadius),
              ),
            ),
          ),
        ),
      ],
    );
  }
}
