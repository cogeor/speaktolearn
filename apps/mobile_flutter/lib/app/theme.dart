import 'package:flutter/material.dart';

/// App theme configuration with pure black dark theme.
class AppTheme {
  AppTheme._();

  /// Pure black background color.
  static const Color background = Color(0xFF000000);

  /// White color for text and accents.
  static const Color foreground = Color(0xFFFFFFFF);

  /// Subtle gray for secondary elements.
  static const Color subtle = Color(0xFF888888);

  /// Success/correct color.
  static const Color success = Color(0xFF4CAF50);

  /// Warning color.
  static const Color warning = Color(0xFFFFC107);

  /// Error/incorrect color.
  static const Color error = Color(0xFFF44336);

  // Rating colors for self-reported difficulty
  /// Rating color for "hard" - Red.
  static const Color ratingHard = Color.fromARGB(255, 255, 40, 40);

  /// Rating color for "almost" - Yellow.
  static const Color ratingAlmost = Color.fromARGB(255, 255, 202, 56);

  /// Rating color for "good" - Green.
  static const Color ratingGood = Color.fromARGB(255, 74, 210, 0);

  /// Rating color for "easy" - Blue.
  static const Color ratingEasy = Color.fromARGB(255, 83, 100, 255);

  /// The dark theme data.
  static ThemeData get darkTheme => ThemeData(
    useMaterial3: true,
    brightness: Brightness.dark,
    scaffoldBackgroundColor: background,
    colorScheme: const ColorScheme.dark(
      surface: background,
      primary: foreground,
      onPrimary: background,
      secondary: subtle,
      onSecondary: foreground,
      error: error,
      onError: foreground,
    ),
    appBarTheme: const AppBarTheme(
      backgroundColor: background,
      foregroundColor: foreground,
      elevation: 0,
    ),
    cardTheme: const CardThemeData(color: Color(0xFF111111), elevation: 0),
    elevatedButtonTheme: ElevatedButtonThemeData(
      style: ElevatedButton.styleFrom(
        backgroundColor: foreground,
        foregroundColor: background,
      ),
    ),
    textButtonTheme: TextButtonThemeData(
      style: TextButton.styleFrom(foregroundColor: foreground),
    ),
    iconTheme: const IconThemeData(color: foreground),
    textTheme: const TextTheme(
      displayLarge: TextStyle(
        fontSize: 36,
        fontWeight: FontWeight.w400,
        color: foreground,
      ),
      titleMedium: TextStyle(
        fontSize: 18,
        fontWeight: FontWeight.w500,
        color: foreground,
      ),
      labelLarge: TextStyle(
        fontSize: 16,
        fontWeight: FontWeight.w500,
        color: foreground,
      ),
      bodySmall: TextStyle(
        fontSize: 12,
        fontWeight: FontWeight.w400,
        color: foreground,
      ),
    ),
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: OutlinedButton.styleFrom(
        foregroundColor: foreground,
        side: const BorderSide(color: foreground),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
      ),
    ),
    iconButtonTheme: IconButtonThemeData(
      style: IconButton.styleFrom(foregroundColor: foreground),
    ),
    bottomSheetTheme: const BottomSheetThemeData(
      backgroundColor: background,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
      ),
    ),
  );
}

/// Extension for score-based color coding.
extension ScoreColors on int {
  /// Gets the color for this score value.
  /// 0-49: error (red), 50-79: warning (yellow), 80-100: success (green)
  Color get scoreColor {
    if (this >= 80) return AppTheme.success;
    if (this >= 50) return AppTheme.warning;
    return AppTheme.error;
  }
}
