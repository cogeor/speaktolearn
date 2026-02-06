// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'text_sequence_progress.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

_$TextSequenceProgressImpl _$$TextSequenceProgressImplFromJson(
  Map<String, dynamic> json,
) => _$TextSequenceProgressImpl(
  tracked: json['tracked'] as bool,
  bestScore: (json['bestScore'] as num?)?.toInt(),
  bestAttemptId: json['bestAttemptId'] as String?,
  lastAttemptAt: json['lastAttemptAt'] == null
      ? null
      : DateTime.parse(json['lastAttemptAt'] as String),
  attemptCount: (json['attemptCount'] as num?)?.toInt() ?? 0,
);

Map<String, dynamic> _$$TextSequenceProgressImplToJson(
  _$TextSequenceProgressImpl instance,
) => <String, dynamic>{
  'tracked': instance.tracked,
  'bestScore': instance.bestScore,
  'bestAttemptId': instance.bestAttemptId,
  'lastAttemptAt': instance.lastAttemptAt?.toIso8601String(),
  'attemptCount': instance.attemptCount,
};
