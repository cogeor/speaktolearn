// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'score_attempt.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

_$ScoreAttemptImpl _$$ScoreAttemptImplFromJson(Map<String, dynamic> json) =>
    _$ScoreAttemptImpl(
      id: json['id'] as String,
      textSequenceId: json['textSequenceId'] as String,
      gradedAt: DateTime.parse(json['gradedAt'] as String),
      score: (json['score'] as num).toInt(),
      method: json['method'] as String,
      recognizedText: json['recognizedText'] as String?,
      details: json['details'] as Map<String, dynamic>?,
    );

Map<String, dynamic> _$$ScoreAttemptImplToJson(_$ScoreAttemptImpl instance) =>
    <String, dynamic>{
      'id': instance.id,
      'textSequenceId': instance.textSequenceId,
      'gradedAt': instance.gradedAt.toIso8601String(),
      'score': instance.score,
      'method': instance.method,
      'recognizedText': instance.recognizedText,
      'details': instance.details,
    };
