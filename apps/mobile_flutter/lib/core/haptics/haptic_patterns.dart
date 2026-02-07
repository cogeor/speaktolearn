import 'package:flutter/services.dart';

/// Semantic haptic feedback patterns for consistent UX.
///
/// Use these methods instead of calling HapticFeedback directly
/// to ensure consistent feedback across the app.
class HapticPatterns {
  HapticPatterns._(); // Prevent instantiation

  /// Action started successfully (e.g., recording started).
  static void actionStarted() {
    HapticFeedback.mediumImpact();
  }

  /// Action completed successfully (e.g., recording stopped).
  static void actionCompleted() {
    HapticFeedback.lightImpact();
  }

  /// Error occurred - alerts user to problem.
  static void error() {
    HapticFeedback.heavyImpact();
  }

  /// Positive result (e.g., good score).
  static void success() {
    HapticFeedback.mediumImpact();
  }

  /// Selection changed (e.g., list item selected).
  static void selection() {
    HapticFeedback.selectionClick();
  }
}
