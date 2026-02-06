import 'package:freezed_annotation/freezed_annotation.dart';

part 'app_settings.freezed.dart';
part 'app_settings.g.dart';

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
  }) = _AppSettings;

  factory AppSettings.fromJson(Map<String, dynamic> json) =>
      _$AppSettingsFromJson(json);

  static const defaults = AppSettings();
}
