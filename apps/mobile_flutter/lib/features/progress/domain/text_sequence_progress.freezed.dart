// coverage:ignore-file
// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'text_sequence_progress.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

T _$identity<T>(T value) => value;

final _privateConstructorUsedError = UnsupportedError(
  'It seems like you constructed your class using `MyClass._()`. This constructor is only meant to be used by freezed and you are not supposed to need it nor use it.\nPlease check the documentation here for more information: https://github.com/rrousselGit/freezed#adding-getters-and-methods-to-our-models',
);

TextSequenceProgress _$TextSequenceProgressFromJson(Map<String, dynamic> json) {
  return _TextSequenceProgress.fromJson(json);
}

/// @nodoc
mixin _$TextSequenceProgress {
  bool get tracked => throw _privateConstructorUsedError;
  int? get bestScore => throw _privateConstructorUsedError;
  String? get bestAttemptId => throw _privateConstructorUsedError;
  DateTime? get lastAttemptAt => throw _privateConstructorUsedError;
  int get attemptCount => throw _privateConstructorUsedError;

  /// Serializes this TextSequenceProgress to a JSON map.
  Map<String, dynamic> toJson() => throw _privateConstructorUsedError;

  /// Create a copy of TextSequenceProgress
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  $TextSequenceProgressCopyWith<TextSequenceProgress> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class $TextSequenceProgressCopyWith<$Res> {
  factory $TextSequenceProgressCopyWith(
    TextSequenceProgress value,
    $Res Function(TextSequenceProgress) then,
  ) = _$TextSequenceProgressCopyWithImpl<$Res, TextSequenceProgress>;
  @useResult
  $Res call({
    bool tracked,
    int? bestScore,
    String? bestAttemptId,
    DateTime? lastAttemptAt,
    int attemptCount,
  });
}

/// @nodoc
class _$TextSequenceProgressCopyWithImpl<
  $Res,
  $Val extends TextSequenceProgress
>
    implements $TextSequenceProgressCopyWith<$Res> {
  _$TextSequenceProgressCopyWithImpl(this._value, this._then);

  // ignore: unused_field
  final $Val _value;
  // ignore: unused_field
  final $Res Function($Val) _then;

  /// Create a copy of TextSequenceProgress
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? tracked = null,
    Object? bestScore = freezed,
    Object? bestAttemptId = freezed,
    Object? lastAttemptAt = freezed,
    Object? attemptCount = null,
  }) {
    return _then(
      _value.copyWith(
            tracked: null == tracked
                ? _value.tracked
                : tracked // ignore: cast_nullable_to_non_nullable
                      as bool,
            bestScore: freezed == bestScore
                ? _value.bestScore
                : bestScore // ignore: cast_nullable_to_non_nullable
                      as int?,
            bestAttemptId: freezed == bestAttemptId
                ? _value.bestAttemptId
                : bestAttemptId // ignore: cast_nullable_to_non_nullable
                      as String?,
            lastAttemptAt: freezed == lastAttemptAt
                ? _value.lastAttemptAt
                : lastAttemptAt // ignore: cast_nullable_to_non_nullable
                      as DateTime?,
            attemptCount: null == attemptCount
                ? _value.attemptCount
                : attemptCount // ignore: cast_nullable_to_non_nullable
                      as int,
          )
          as $Val,
    );
  }
}

/// @nodoc
abstract class _$$TextSequenceProgressImplCopyWith<$Res>
    implements $TextSequenceProgressCopyWith<$Res> {
  factory _$$TextSequenceProgressImplCopyWith(
    _$TextSequenceProgressImpl value,
    $Res Function(_$TextSequenceProgressImpl) then,
  ) = __$$TextSequenceProgressImplCopyWithImpl<$Res>;
  @override
  @useResult
  $Res call({
    bool tracked,
    int? bestScore,
    String? bestAttemptId,
    DateTime? lastAttemptAt,
    int attemptCount,
  });
}

/// @nodoc
class __$$TextSequenceProgressImplCopyWithImpl<$Res>
    extends _$TextSequenceProgressCopyWithImpl<$Res, _$TextSequenceProgressImpl>
    implements _$$TextSequenceProgressImplCopyWith<$Res> {
  __$$TextSequenceProgressImplCopyWithImpl(
    _$TextSequenceProgressImpl _value,
    $Res Function(_$TextSequenceProgressImpl) _then,
  ) : super(_value, _then);

  /// Create a copy of TextSequenceProgress
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? tracked = null,
    Object? bestScore = freezed,
    Object? bestAttemptId = freezed,
    Object? lastAttemptAt = freezed,
    Object? attemptCount = null,
  }) {
    return _then(
      _$TextSequenceProgressImpl(
        tracked: null == tracked
            ? _value.tracked
            : tracked // ignore: cast_nullable_to_non_nullable
                  as bool,
        bestScore: freezed == bestScore
            ? _value.bestScore
            : bestScore // ignore: cast_nullable_to_non_nullable
                  as int?,
        bestAttemptId: freezed == bestAttemptId
            ? _value.bestAttemptId
            : bestAttemptId // ignore: cast_nullable_to_non_nullable
                  as String?,
        lastAttemptAt: freezed == lastAttemptAt
            ? _value.lastAttemptAt
            : lastAttemptAt // ignore: cast_nullable_to_non_nullable
                  as DateTime?,
        attemptCount: null == attemptCount
            ? _value.attemptCount
            : attemptCount // ignore: cast_nullable_to_non_nullable
                  as int,
      ),
    );
  }
}

/// @nodoc
@JsonSerializable()
class _$TextSequenceProgressImpl extends _TextSequenceProgress {
  const _$TextSequenceProgressImpl({
    required this.tracked,
    this.bestScore,
    this.bestAttemptId,
    this.lastAttemptAt,
    this.attemptCount = 0,
  }) : super._();

  factory _$TextSequenceProgressImpl.fromJson(Map<String, dynamic> json) =>
      _$$TextSequenceProgressImplFromJson(json);

  @override
  final bool tracked;
  @override
  final int? bestScore;
  @override
  final String? bestAttemptId;
  @override
  final DateTime? lastAttemptAt;
  @override
  @JsonKey()
  final int attemptCount;

  @override
  String toString() {
    return 'TextSequenceProgress(tracked: $tracked, bestScore: $bestScore, bestAttemptId: $bestAttemptId, lastAttemptAt: $lastAttemptAt, attemptCount: $attemptCount)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$TextSequenceProgressImpl &&
            (identical(other.tracked, tracked) || other.tracked == tracked) &&
            (identical(other.bestScore, bestScore) ||
                other.bestScore == bestScore) &&
            (identical(other.bestAttemptId, bestAttemptId) ||
                other.bestAttemptId == bestAttemptId) &&
            (identical(other.lastAttemptAt, lastAttemptAt) ||
                other.lastAttemptAt == lastAttemptAt) &&
            (identical(other.attemptCount, attemptCount) ||
                other.attemptCount == attemptCount));
  }

  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  int get hashCode => Object.hash(
    runtimeType,
    tracked,
    bestScore,
    bestAttemptId,
    lastAttemptAt,
    attemptCount,
  );

  /// Create a copy of TextSequenceProgress
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$TextSequenceProgressImplCopyWith<_$TextSequenceProgressImpl>
  get copyWith =>
      __$$TextSequenceProgressImplCopyWithImpl<_$TextSequenceProgressImpl>(
        this,
        _$identity,
      );

  @override
  Map<String, dynamic> toJson() {
    return _$$TextSequenceProgressImplToJson(this);
  }
}

abstract class _TextSequenceProgress extends TextSequenceProgress {
  const factory _TextSequenceProgress({
    required final bool tracked,
    final int? bestScore,
    final String? bestAttemptId,
    final DateTime? lastAttemptAt,
    final int attemptCount,
  }) = _$TextSequenceProgressImpl;
  const _TextSequenceProgress._() : super._();

  factory _TextSequenceProgress.fromJson(Map<String, dynamic> json) =
      _$TextSequenceProgressImpl.fromJson;

  @override
  bool get tracked;
  @override
  int? get bestScore;
  @override
  String? get bestAttemptId;
  @override
  DateTime? get lastAttemptAt;
  @override
  int get attemptCount;

  /// Create a copy of TextSequenceProgress
  /// with the given fields replaced by the non-null parameter values.
  @override
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$TextSequenceProgressImplCopyWith<_$TextSequenceProgressImpl>
  get copyWith => throw _privateConstructorUsedError;
}
