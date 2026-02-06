// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'example_audio_dto.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

VoiceDto _$VoiceDtoFromJson(Map<String, dynamic> json) => VoiceDto(
  id: json['id'] as String,
  label: Map<String, String>.from(json['label'] as Map),
  uri: json['uri'] as String,
  durationMs: (json['duration_ms'] as num).toInt(),
);

Map<String, dynamic> _$VoiceDtoToJson(VoiceDto instance) => <String, dynamic>{
  'id': instance.id,
  'label': instance.label,
  'uri': instance.uri,
  'duration_ms': instance.durationMs,
};

ExampleAudioDto _$ExampleAudioDtoFromJson(Map<String, dynamic> json) =>
    ExampleAudioDto(
      voices: (json['voices'] as List<dynamic>)
          .map((e) => VoiceDto.fromJson(e as Map<String, dynamic>))
          .toList(),
    );

Map<String, dynamic> _$ExampleAudioDtoToJson(ExampleAudioDto instance) =>
    <String, dynamic>{'voices': instance.voices};
