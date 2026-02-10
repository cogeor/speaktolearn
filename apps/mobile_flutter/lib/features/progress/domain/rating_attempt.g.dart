// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'rating_attempt.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

_$RatingAttemptImpl _$$RatingAttemptImplFromJson(Map<String, dynamic> json) =>
    _$RatingAttemptImpl(
      id: json['id'] as String,
      textSequenceId: json['textSequenceId'] as String,
      gradedAt: DateTime.parse(json['gradedAt'] as String),
      rating: $enumDecode(_$SentenceRatingEnumMap, json['rating']),
    );

Map<String, dynamic> _$$RatingAttemptImplToJson(_$RatingAttemptImpl instance) =>
    <String, dynamic>{
      'id': instance.id,
      'textSequenceId': instance.textSequenceId,
      'gradedAt': instance.gradedAt.toIso8601String(),
      'rating': _$SentenceRatingEnumMap[instance.rating]!,
    };

const _$SentenceRatingEnumMap = {
  SentenceRating.hard: 'hard',
  SentenceRating.almost: 'almost',
  SentenceRating.good: 'good',
  SentenceRating.easy: 'easy',
};
