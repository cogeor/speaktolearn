// coverage:ignore-file
// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'rating_attempt.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

T _$identity<T>(T value) => value;

final _privateConstructorUsedError = UnsupportedError(
  'It seems like you constructed your class using `MyClass._()`. This constructor is only meant to be used by freezed and you are not supposed to need it nor use it.\nPlease check the documentation here for more information: https://github.com/rrousselGit/freezed#adding-getters-and-methods-to-our-models',
);

RatingAttempt _$RatingAttemptFromJson(Map<String, dynamic> json) {
  return _RatingAttempt.fromJson(json);
}

/// @nodoc
mixin _$RatingAttempt {
  /// Unique identifier for this attempt.
  String get id => throw _privateConstructorUsedError;

  /// The ID of the text sequence that was rated.
  String get textSequenceId => throw _privateConstructorUsedError;

  /// When this rating was recorded.
  DateTime get gradedAt => throw _privateConstructorUsedError;

  /// The self-reported rating for this attempt.
  SentenceRating get rating => throw _privateConstructorUsedError;

  /// Serializes this RatingAttempt to a JSON map.
  Map<String, dynamic> toJson() => throw _privateConstructorUsedError;

  /// Create a copy of RatingAttempt
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  $RatingAttemptCopyWith<RatingAttempt> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class $RatingAttemptCopyWith<$Res> {
  factory $RatingAttemptCopyWith(
    RatingAttempt value,
    $Res Function(RatingAttempt) then,
  ) = _$RatingAttemptCopyWithImpl<$Res, RatingAttempt>;
  @useResult
  $Res call({
    String id,
    String textSequenceId,
    DateTime gradedAt,
    SentenceRating rating,
  });
}

/// @nodoc
class _$RatingAttemptCopyWithImpl<$Res, $Val extends RatingAttempt>
    implements $RatingAttemptCopyWith<$Res> {
  _$RatingAttemptCopyWithImpl(this._value, this._then);

  // ignore: unused_field
  final $Val _value;
  // ignore: unused_field
  final $Res Function($Val) _then;

  /// Create a copy of RatingAttempt
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? id = null,
    Object? textSequenceId = null,
    Object? gradedAt = null,
    Object? rating = null,
  }) {
    return _then(
      _value.copyWith(
            id: null == id
                ? _value.id
                : id // ignore: cast_nullable_to_non_nullable
                      as String,
            textSequenceId: null == textSequenceId
                ? _value.textSequenceId
                : textSequenceId // ignore: cast_nullable_to_non_nullable
                      as String,
            gradedAt: null == gradedAt
                ? _value.gradedAt
                : gradedAt // ignore: cast_nullable_to_non_nullable
                      as DateTime,
            rating: null == rating
                ? _value.rating
                : rating // ignore: cast_nullable_to_non_nullable
                      as SentenceRating,
          )
          as $Val,
    );
  }
}

/// @nodoc
abstract class _$$RatingAttemptImplCopyWith<$Res>
    implements $RatingAttemptCopyWith<$Res> {
  factory _$$RatingAttemptImplCopyWith(
    _$RatingAttemptImpl value,
    $Res Function(_$RatingAttemptImpl) then,
  ) = __$$RatingAttemptImplCopyWithImpl<$Res>;
  @override
  @useResult
  $Res call({
    String id,
    String textSequenceId,
    DateTime gradedAt,
    SentenceRating rating,
  });
}

/// @nodoc
class __$$RatingAttemptImplCopyWithImpl<$Res>
    extends _$RatingAttemptCopyWithImpl<$Res, _$RatingAttemptImpl>
    implements _$$RatingAttemptImplCopyWith<$Res> {
  __$$RatingAttemptImplCopyWithImpl(
    _$RatingAttemptImpl _value,
    $Res Function(_$RatingAttemptImpl) _then,
  ) : super(_value, _then);

  /// Create a copy of RatingAttempt
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? id = null,
    Object? textSequenceId = null,
    Object? gradedAt = null,
    Object? rating = null,
  }) {
    return _then(
      _$RatingAttemptImpl(
        id: null == id
            ? _value.id
            : id // ignore: cast_nullable_to_non_nullable
                  as String,
        textSequenceId: null == textSequenceId
            ? _value.textSequenceId
            : textSequenceId // ignore: cast_nullable_to_non_nullable
                  as String,
        gradedAt: null == gradedAt
            ? _value.gradedAt
            : gradedAt // ignore: cast_nullable_to_non_nullable
                  as DateTime,
        rating: null == rating
            ? _value.rating
            : rating // ignore: cast_nullable_to_non_nullable
                  as SentenceRating,
      ),
    );
  }
}

/// @nodoc
@JsonSerializable()
class _$RatingAttemptImpl implements _RatingAttempt {
  const _$RatingAttemptImpl({
    required this.id,
    required this.textSequenceId,
    required this.gradedAt,
    required this.rating,
  });

  factory _$RatingAttemptImpl.fromJson(Map<String, dynamic> json) =>
      _$$RatingAttemptImplFromJson(json);

  /// Unique identifier for this attempt.
  @override
  final String id;

  /// The ID of the text sequence that was rated.
  @override
  final String textSequenceId;

  /// When this rating was recorded.
  @override
  final DateTime gradedAt;

  /// The self-reported rating for this attempt.
  @override
  final SentenceRating rating;

  @override
  String toString() {
    return 'RatingAttempt(id: $id, textSequenceId: $textSequenceId, gradedAt: $gradedAt, rating: $rating)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$RatingAttemptImpl &&
            (identical(other.id, id) || other.id == id) &&
            (identical(other.textSequenceId, textSequenceId) ||
                other.textSequenceId == textSequenceId) &&
            (identical(other.gradedAt, gradedAt) ||
                other.gradedAt == gradedAt) &&
            (identical(other.rating, rating) || other.rating == rating));
  }

  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  int get hashCode =>
      Object.hash(runtimeType, id, textSequenceId, gradedAt, rating);

  /// Create a copy of RatingAttempt
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$RatingAttemptImplCopyWith<_$RatingAttemptImpl> get copyWith =>
      __$$RatingAttemptImplCopyWithImpl<_$RatingAttemptImpl>(this, _$identity);

  @override
  Map<String, dynamic> toJson() {
    return _$$RatingAttemptImplToJson(this);
  }
}

abstract class _RatingAttempt implements RatingAttempt {
  const factory _RatingAttempt({
    required final String id,
    required final String textSequenceId,
    required final DateTime gradedAt,
    required final SentenceRating rating,
  }) = _$RatingAttemptImpl;

  factory _RatingAttempt.fromJson(Map<String, dynamic> json) =
      _$RatingAttemptImpl.fromJson;

  /// Unique identifier for this attempt.
  @override
  String get id;

  /// The ID of the text sequence that was rated.
  @override
  String get textSequenceId;

  /// When this rating was recorded.
  @override
  DateTime get gradedAt;

  /// The self-reported rating for this attempt.
  @override
  SentenceRating get rating;

  /// Create a copy of RatingAttempt
  /// with the given fields replaced by the non-null parameter values.
  @override
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$RatingAttemptImplCopyWith<_$RatingAttemptImpl> get copyWith =>
      throw _privateConstructorUsedError;
}
