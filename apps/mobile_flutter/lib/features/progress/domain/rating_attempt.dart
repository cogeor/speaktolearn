import 'package:freezed_annotation/freezed_annotation.dart';

import 'sentence_rating.dart';

part 'rating_attempt.freezed.dart';
part 'rating_attempt.g.dart';

/// A self-reported rating attempt for a sentence.
///
/// Records when a user rates their own performance on a sentence
/// using the [SentenceRating] scale (hard, almost, good, easy).
@freezed
class RatingAttempt with _$RatingAttempt {
  const factory RatingAttempt({
    /// Unique identifier for this attempt.
    required String id,

    /// The ID of the text sequence that was rated.
    required String textSequenceId,

    /// When this rating was recorded.
    required DateTime gradedAt,

    /// The self-reported rating for this attempt.
    required SentenceRating rating,
  }) = _RatingAttempt;

  factory RatingAttempt.fromJson(Map<String, dynamic> json) =>
      _$RatingAttemptFromJson(json);
}
