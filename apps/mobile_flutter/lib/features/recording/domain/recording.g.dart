// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'recording.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

_$RecordingImpl _$$RecordingImplFromJson(Map<String, dynamic> json) =>
    _$RecordingImpl(
      id: json['id'] as String,
      textSequenceId: json['textSequenceId'] as String,
      createdAt: DateTime.parse(json['createdAt'] as String),
      filePath: json['filePath'] as String,
      durationMs: (json['durationMs'] as num?)?.toInt(),
      sampleRate: (json['sampleRate'] as num?)?.toInt(),
      mimeType: json['mimeType'] as String?,
    );

Map<String, dynamic> _$$RecordingImplToJson(_$RecordingImpl instance) =>
    <String, dynamic>{
      'id': instance.id,
      'textSequenceId': instance.textSequenceId,
      'createdAt': instance.createdAt.toIso8601String(),
      'filePath': instance.filePath,
      'durationMs': instance.durationMs,
      'sampleRate': instance.sampleRate,
      'mimeType': instance.mimeType,
    };
