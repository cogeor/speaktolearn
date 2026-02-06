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
        cardTheme: const CardThemeData(
          color: Color(0xFF111111),
          elevation: 0,
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: foreground,
            foregroundColor: background,
          ),
        ),
        textButtonTheme: TextButtonThemeData(
          style: TextButton.styleFrom(
            foregroundColor: foreground,
          ),
        ),
        iconTheme: const IconThemeData(
          color: foreground,
        ),
      );
}
