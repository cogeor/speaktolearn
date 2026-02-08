import 'package:freezed_annotation/freezed_annotation.dart';

part 'app_settings.freezed.dart';
part 'app_settings.g.dart';

/// Voice preference for audio playback.
///
/// Controls which voice is selected when multiple voices are available
/// for an example sentence:
/// - [noPreference]: Picks the first available voice
/// - [male]: Prefers male voices if available
/// - [female]: Prefers female voices if available
enum VoicePreference {
  /// Use the first available voice (no gender preference).
  @JsonValue('systemDefault') // Backward compatibility with stored settings
  noPreference,

  /// Prefer male voices if available.
  male,

  /// Prefer female voices if available.
  female,
}

@freezed
class AppSettings with _$AppSettings {
  const factory AppSettings({
    @Default('en') String uiLanguageCode,
    @Default('zh-CN') String targetLanguageCode,
    @Default(true) bool showRomanization,
    @Default(true) bool showGloss,
    @Default(1.0) double playbackSpeed,
    @Default(false) bool autoPlayExample,
    String? preferredVoiceId,
    @Default(VoicePreference.noPreference) VoicePreference voicePreference,
    /// Current HSK level (1-6). Defaults to 1.
    @Default(1) int currentLevel,
  }) = _AppSettings;

  factory AppSettings.fromJson(Map<String, dynamic> json) =>
      _$AppSettingsFromJson(json);

  static const defaults = AppSettings();
}
