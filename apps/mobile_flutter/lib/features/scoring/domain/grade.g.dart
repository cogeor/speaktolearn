// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'grade.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

_$GradeImpl _$$GradeImplFromJson(Map<String, dynamic> json) => _$GradeImpl(
  overall: (json['overall'] as num).toInt(),
  method: json['method'] as String,
  recognizedText: json['recognizedText'] as String?,
  details: json['details'] as Map<String, dynamic>?,
);

Map<String, dynamic> _$$GradeImplToJson(_$GradeImpl instance) =>
    <String, dynamic>{
      'overall': instance.overall,
      'method': instance.method,
      'recognizedText': instance.recognizedText,
      'details': instance.details,
    };
