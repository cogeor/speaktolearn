import 'package:flutter/material.dart';

import '../../../../app/theme.dart';

/// Widget that displays text with per-character color-coding based on scores.
///
/// Uses existing theme colors:
/// - [AppTheme.ratingHard] (red) for scores < 0.2 (bad)
/// - [AppTheme.ratingAlmost] (yellow) for scores 0.2-0.4 (almost)
/// - [AppTheme.ratingGood] (green) for scores 0.4-0.6 (good)
/// - [AppTheme.ratingEasy] (blue) for scores >= 0.6 (easy)
/// - [AppTheme.foreground] (white) as default when no scores provided
///
/// Handles Unicode text properly using grapheme clusters.
class ColoredText extends StatelessWidget {
  const ColoredText({
    super.key,
    required this.text,
    this.scores,
    this.style,
    this.textAlign = TextAlign.center,
  });

  /// The text to display.
  final String text;

  /// Per-character scores (0.0-1.0). Length should match text.characters.length.
  /// When null or empty, displays plain white text.
  final List<double>? scores;

  /// Base text style. Color will be overridden per character when scores are provided.
  final TextStyle? style;

  /// Text alignment.
  final TextAlign textAlign;

  @override
  Widget build(BuildContext context) {
    final baseStyle = style ?? Theme.of(context).textTheme.displayLarge;

    // No scores provided - show plain white text
    if (scores == null || scores!.isEmpty) {
      return Text(
        text,
        style: baseStyle?.copyWith(color: AppTheme.foreground),
        textAlign: textAlign,
      );
    }

    // Split text into grapheme clusters for proper Unicode handling
    final characters = text.characters.toList();
    final spans = <TextSpan>[];

    for (int i = 0; i < characters.length; i++) {
      final score = i < scores!.length ? scores![i] : null;
      final color = score != null ? _scoreToColor(score) : AppTheme.foreground;

      spans.add(
        TextSpan(
          text: characters[i],
          style: baseStyle?.copyWith(color: color),
        ),
      );
    }

    return RichText(
      text: TextSpan(children: spans),
      textAlign: textAlign,
    );
  }

  /// Map score (0.0-1.0) to theme color using grade thresholds.
  ///
  /// Thresholds match Python model's probability_to_grade():
  /// - bad: < 0.2 -> ratingHard (red)
  /// - almost: 0.2-0.4 -> ratingAlmost (yellow)
  /// - good: 0.4-0.6 -> ratingGood (green)
  /// - easy: >= 0.6 -> ratingEasy (blue)
  Color _scoreToColor(double score) {
    if (score >= 0.6) return AppTheme.ratingEasy;
    if (score >= 0.4) return AppTheme.ratingGood;
    if (score >= 0.2) return AppTheme.ratingAlmost;
    return AppTheme.ratingHard;
  }
}
