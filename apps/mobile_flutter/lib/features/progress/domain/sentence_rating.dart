import 'package:flutter/material.dart';

import '../../../app/theme.dart';

/// Self-reported difficulty rating for a sentence.
enum SentenceRating {
  /// Sentence was difficult/hard.
  hard,

  /// Sentence was almost correct.
  almost,

  /// Sentence was good/correct.
  good,

  /// Sentence was easy.
  easy,
}

/// Extension for getting the color associated with each rating.
extension SentenceRatingColor on SentenceRating {
  /// Gets the theme color for this rating.
  Color get color {
    switch (this) {
      case SentenceRating.hard:
        return AppTheme.ratingHard;
      case SentenceRating.almost:
        return AppTheme.ratingAlmost;
      case SentenceRating.good:
        return AppTheme.ratingGood;
      case SentenceRating.easy:
        return AppTheme.ratingEasy;
    }
  }
}

/// Extension for getting the display label for each rating.
extension SentenceRatingLabel on SentenceRating {
  /// Gets the human-readable label for this rating.
  String get label {
    switch (this) {
      case SentenceRating.hard:
        return 'Hard';
      case SentenceRating.almost:
        return 'Almost';
      case SentenceRating.good:
        return 'Good';
      case SentenceRating.easy:
        return 'Easy';
    }
  }
}
