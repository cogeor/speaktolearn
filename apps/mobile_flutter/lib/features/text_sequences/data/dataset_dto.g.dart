// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'dataset_dto.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

DatasetDto _$DatasetDtoFromJson(Map<String, dynamic> json) => DatasetDto(
  schemaVersion: json['schema_version'] as String,
  datasetId: json['dataset_id'] as String,
  language: json['language'] as String,
  generatedAt: json['generated_at'] as String,
  items: (json['items'] as List<dynamic>)
      .map((e) => e as Map<String, dynamic>)
      .toList(),
);

Map<String, dynamic> _$DatasetDtoToJson(DatasetDto instance) =>
    <String, dynamic>{
      'schema_version': instance.schemaVersion,
      'dataset_id': instance.datasetId,
      'language': instance.language,
      'generated_at': instance.generatedAt,
      'items': instance.items,
    };
