// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'app_settings.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

_$AppSettingsImpl _$$AppSettingsImplFromJson(Map<String, dynamic> json) =>
    _$AppSettingsImpl(
      uiLanguageCode: json['uiLanguageCode'] as String? ?? 'en',
      targetLanguageCode: json['targetLanguageCode'] as String? ?? 'zh-CN',
      showRomanization: json['showRomanization'] as bool? ?? true,
      showGloss: json['showGloss'] as bool? ?? true,
      playbackSpeed: (json['playbackSpeed'] as num?)?.toDouble() ?? 1.0,
      autoPlayExample: json['autoPlayExample'] as bool? ?? false,
      preferredVoiceId: json['preferredVoiceId'] as String?,
      voicePreference:
          $enumDecodeNullable(
            _$VoicePreferenceEnumMap,
            json['voicePreference'],
          ) ??
          VoicePreference.systemDefault,
    );

Map<String, dynamic> _$$AppSettingsImplToJson(_$AppSettingsImpl instance) =>
    <String, dynamic>{
      'uiLanguageCode': instance.uiLanguageCode,
      'targetLanguageCode': instance.targetLanguageCode,
      'showRomanization': instance.showRomanization,
      'showGloss': instance.showGloss,
      'playbackSpeed': instance.playbackSpeed,
      'autoPlayExample': instance.autoPlayExample,
      'preferredVoiceId': instance.preferredVoiceId,
      'voicePreference': _$VoicePreferenceEnumMap[instance.voicePreference]!,
    };

const _$VoicePreferenceEnumMap = {
  VoicePreference.systemDefault: 'systemDefault',
  VoicePreference.male: 'male',
  VoicePreference.female: 'female',
};
