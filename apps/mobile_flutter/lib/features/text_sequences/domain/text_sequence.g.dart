// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'text_sequence.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

_$TextSequenceImpl _$$TextSequenceImplFromJson(Map<String, dynamic> json) =>
    _$TextSequenceImpl(
      id: json['id'] as String,
      text: json['text'] as String,
      language: json['language'] as String,
      romanization: json['romanization'] as String?,
      gloss: (json['gloss'] as Map<String, dynamic>?)?.map(
        (k, e) => MapEntry(k, e as String),
      ),
      tokens: (json['tokens'] as List<dynamic>?)
          ?.map((e) => e as String)
          .toList(),
      tags: (json['tags'] as List<dynamic>?)?.map((e) => e as String).toList(),
      difficulty: (json['difficulty'] as num?)?.toInt(),
      voices: (json['voices'] as List<dynamic>?)
          ?.map((e) => ExampleVoice.fromJson(e as Map<String, dynamic>))
          .toList(),
    );

Map<String, dynamic> _$$TextSequenceImplToJson(_$TextSequenceImpl instance) =>
    <String, dynamic>{
      'id': instance.id,
      'text': instance.text,
      'language': instance.language,
      'romanization': instance.romanization,
      'gloss': instance.gloss,
      'tokens': instance.tokens,
      'tags': instance.tags,
      'difficulty': instance.difficulty,
      'voices': instance.voices,
    };

_$ExampleVoiceImpl _$$ExampleVoiceImplFromJson(Map<String, dynamic> json) =>
    _$ExampleVoiceImpl(
      id: json['id'] as String,
      label: (json['label'] as Map<String, dynamic>?)?.map(
        (k, e) => MapEntry(k, e as String),
      ),
      uri: json['uri'] as String,
      durationMs: (json['durationMs'] as num?)?.toInt(),
    );

Map<String, dynamic> _$$ExampleVoiceImplToJson(_$ExampleVoiceImpl instance) =>
    <String, dynamic>{
      'id': instance.id,
      'label': instance.label,
      'uri': instance.uri,
      'durationMs': instance.durationMs,
    };
