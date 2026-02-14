// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'grade.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

_$GradeImpl _$$GradeImplFromJson(Map<String, dynamic> json) => _$GradeImpl(
  overall: (json['overall'] as num).toInt(),
  method: json['method'] as String,
  accuracy: (json['accuracy'] as num?)?.toInt(),
  completeness: (json['completeness'] as num?)?.toInt(),
  recognizedText: json['recognizedText'] as String?,
  details: json['details'] as Map<String, dynamic>?,
  characterScores: (json['characterScores'] as List<dynamic>?)
      ?.map((e) => (e as num).toDouble())
      .toList(),
);

Map<String, dynamic> _$$GradeImplToJson(_$GradeImpl instance) =>
    <String, dynamic>{
      'overall': instance.overall,
      'method': instance.method,
      'accuracy': instance.accuracy,
      'completeness': instance.completeness,
      'recognizedText': instance.recognizedText,
      'details': instance.details,
      'characterScores': instance.characterScores,
    };
