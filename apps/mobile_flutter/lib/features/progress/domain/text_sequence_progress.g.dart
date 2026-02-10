// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'text_sequence_progress.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

_$TextSequenceProgressImpl _$$TextSequenceProgressImplFromJson(
  Map<String, dynamic> json,
) => _$TextSequenceProgressImpl(
  tracked: json['tracked'] as bool,
  lastAttemptAt: json['lastAttemptAt'] == null
      ? null
      : DateTime.parse(json['lastAttemptAt'] as String),
  attemptCount: (json['attemptCount'] as num?)?.toInt() ?? 0,
  lastRating: $enumDecodeNullable(_$SentenceRatingEnumMap, json['lastRating']),
);

Map<String, dynamic> _$$TextSequenceProgressImplToJson(
  _$TextSequenceProgressImpl instance,
) => <String, dynamic>{
  'tracked': instance.tracked,
  'lastAttemptAt': instance.lastAttemptAt?.toIso8601String(),
  'attemptCount': instance.attemptCount,
  'lastRating': _$SentenceRatingEnumMap[instance.lastRating],
};

const _$SentenceRatingEnumMap = {
  SentenceRating.hard: 'hard',
  SentenceRating.almost: 'almost',
  SentenceRating.good: 'good',
  SentenceRating.easy: 'easy',
};
