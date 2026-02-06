// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'text_sequence_dto.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

TextSequenceDto _$TextSequenceDtoFromJson(
  Map<String, dynamic> json,
) => TextSequenceDto(
  id: json['id'] as String,
  text: json['text'] as String,
  romanization: json['romanization'] as String?,
  gloss: (json['gloss'] as Map<String, dynamic>?)?.map(
    (k, e) => MapEntry(k, e as String),
  ),
  tokens: (json['tokens'] as List<dynamic>?)?.map((e) => e as String).toList(),
  tags: (json['tags'] as List<dynamic>?)?.map((e) => e as String).toList(),
  difficulty: (json['difficulty'] as num?)?.toInt(),
  exampleAudio: json['example_audio'] == null
      ? null
      : ExampleAudioDto.fromJson(json['example_audio'] as Map<String, dynamic>),
);

Map<String, dynamic> _$TextSequenceDtoToJson(TextSequenceDto instance) =>
    <String, dynamic>{
      'id': instance.id,
      'text': instance.text,
      'romanization': instance.romanization,
      'gloss': instance.gloss,
      'tokens': instance.tokens,
      'tags': instance.tags,
      'difficulty': instance.difficulty,
      'example_audio': instance.exampleAudio,
    };
