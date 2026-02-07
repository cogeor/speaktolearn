import 'package:freezed_annotation/freezed_annotation.dart';

part 'app_settings.freezed.dart';
part 'app_settings.g.dart';

/// Voice preference for TTS playback.
enum VoicePreference {
  systemDefault,
  male,
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
    @Default(VoicePreference.systemDefault) VoicePreference voicePreference,
  }) = _AppSettings;

  factory AppSettings.fromJson(Map<String, dynamic> json) =>
      _$AppSettingsFromJson(json);

  static const defaults = AppSettings();
}
