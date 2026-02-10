import 'package:flutter/material.dart';

import '../../../progress/domain/sentence_rating.dart';

/// A small colored indicator showing the last rating for a sentence.
///
/// Displays as a rounded square, with color based on the [SentenceRating].
/// If no rating is provided (null), shows a dark gray indicator.
class RatingIndicator extends StatelessWidget {
  /// Creates a RatingIndicator.
  const RatingIndicator({super.key, this.rating, this.size = 16});

  /// The sentence rating to display. If null, shows dark gray (no rating yet).
  final SentenceRating? rating;

  /// The size of the indicator (width and height). Defaults to 16.
  final double size;

  /// Color shown when no rating has been recorded.
  static const Color _noRatingColor = Color(0xFF333333);

  @override
  Widget build(BuildContext context) {
    final color = rating?.color ?? _noRatingColor;

    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        color: color,
        borderRadius: BorderRadius.circular(size * 0.25),
      ),
    );
  }
}
