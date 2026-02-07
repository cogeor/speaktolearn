/// Calculates the expected recording duration for a text.
///
/// Uses language-specific speaking rates:
/// - Chinese: ~3 characters per second (syllabic nature)
/// - Other languages: ~5 characters per second (approximate)
///
/// Adds a 2-second buffer for preparation and trailing silence.
/// Minimum duration is 3 seconds.
Duration calculateRecordingDuration(String text, String language) {
  final charCount = text.length;

  // Calculate base seconds based on language
  final int baseSeconds;
  if (language == 'zh' || language == 'cmn' || language.startsWith('zh-')) {
    // Chinese: approximately 3 characters per second
    // Each character is typically one syllable
    baseSeconds = (charCount / 3).ceil();
  } else {
    // Other languages: approximately 5 characters per second
    baseSeconds = (charCount / 5).ceil();
  }

  // Add 2-second buffer for preparation and trailing
  final totalSeconds = baseSeconds + 2;

  // Enforce minimum of 3 seconds
  final finalSeconds = totalSeconds < 3 ? 3 : totalSeconds;

  return Duration(seconds: finalSeconds);
}

/// Extension to provide a formatted countdown string.
extension DurationFormatting on Duration {
  /// Returns duration as "M:SS" format for display.
  String toCountdownString() {
    final minutes = inMinutes;
    final seconds = inSeconds % 60;
    return '$minutes:${seconds.toString().padLeft(2, '0')}';
  }
}
