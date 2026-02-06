// coverage:ignore-file
// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'score_attempt.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

T _$identity<T>(T value) => value;

final _privateConstructorUsedError = UnsupportedError(
  'It seems like you constructed your class using `MyClass._()`. This constructor is only meant to be used by freezed and you are not supposed to need it nor use it.\nPlease check the documentation here for more information: https://github.com/rrousselGit/freezed#adding-getters-and-methods-to-our-models',
);

ScoreAttempt _$ScoreAttemptFromJson(Map<String, dynamic> json) {
  return _ScoreAttempt.fromJson(json);
}

/// @nodoc
mixin _$ScoreAttempt {
  String get id => throw _privateConstructorUsedError;
  String get textSequenceId => throw _privateConstructorUsedError;
  DateTime get gradedAt => throw _privateConstructorUsedError;
  int get score => throw _privateConstructorUsedError;
  String get method => throw _privateConstructorUsedError;
  String? get recognizedText => throw _privateConstructorUsedError;
  Map<String, dynamic>? get details => throw _privateConstructorUsedError;

  /// Serializes this ScoreAttempt to a JSON map.
  Map<String, dynamic> toJson() => throw _privateConstructorUsedError;

  /// Create a copy of ScoreAttempt
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  $ScoreAttemptCopyWith<ScoreAttempt> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class $ScoreAttemptCopyWith<$Res> {
  factory $ScoreAttemptCopyWith(
    ScoreAttempt value,
    $Res Function(ScoreAttempt) then,
  ) = _$ScoreAttemptCopyWithImpl<$Res, ScoreAttempt>;
  @useResult
  $Res call({
    String id,
    String textSequenceId,
    DateTime gradedAt,
    int score,
    String method,
    String? recognizedText,
    Map<String, dynamic>? details,
  });
}

/// @nodoc
class _$ScoreAttemptCopyWithImpl<$Res, $Val extends ScoreAttempt>
    implements $ScoreAttemptCopyWith<$Res> {
  _$ScoreAttemptCopyWithImpl(this._value, this._then);

  // ignore: unused_field
  final $Val _value;
  // ignore: unused_field
  final $Res Function($Val) _then;

  /// Create a copy of ScoreAttempt
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? id = null,
    Object? textSequenceId = null,
    Object? gradedAt = null,
    Object? score = null,
    Object? method = null,
    Object? recognizedText = freezed,
    Object? details = freezed,
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
            score: null == score
                ? _value.score
                : score // ignore: cast_nullable_to_non_nullable
                      as int,
            method: null == method
                ? _value.method
                : method // ignore: cast_nullable_to_non_nullable
                      as String,
            recognizedText: freezed == recognizedText
                ? _value.recognizedText
                : recognizedText // ignore: cast_nullable_to_non_nullable
                      as String?,
            details: freezed == details
                ? _value.details
                : details // ignore: cast_nullable_to_non_nullable
                      as Map<String, dynamic>?,
          )
          as $Val,
    );
  }
}

/// @nodoc
abstract class _$$ScoreAttemptImplCopyWith<$Res>
    implements $ScoreAttemptCopyWith<$Res> {
  factory _$$ScoreAttemptImplCopyWith(
    _$ScoreAttemptImpl value,
    $Res Function(_$ScoreAttemptImpl) then,
  ) = __$$ScoreAttemptImplCopyWithImpl<$Res>;
  @override
  @useResult
  $Res call({
    String id,
    String textSequenceId,
    DateTime gradedAt,
    int score,
    String method,
    String? recognizedText,
    Map<String, dynamic>? details,
  });
}

/// @nodoc
class __$$ScoreAttemptImplCopyWithImpl<$Res>
    extends _$ScoreAttemptCopyWithImpl<$Res, _$ScoreAttemptImpl>
    implements _$$ScoreAttemptImplCopyWith<$Res> {
  __$$ScoreAttemptImplCopyWithImpl(
    _$ScoreAttemptImpl _value,
    $Res Function(_$ScoreAttemptImpl) _then,
  ) : super(_value, _then);

  /// Create a copy of ScoreAttempt
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? id = null,
    Object? textSequenceId = null,
    Object? gradedAt = null,
    Object? score = null,
    Object? method = null,
    Object? recognizedText = freezed,
    Object? details = freezed,
  }) {
    return _then(
      _$ScoreAttemptImpl(
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
        score: null == score
            ? _value.score
            : score // ignore: cast_nullable_to_non_nullable
                  as int,
        method: null == method
            ? _value.method
            : method // ignore: cast_nullable_to_non_nullable
                  as String,
        recognizedText: freezed == recognizedText
            ? _value.recognizedText
            : recognizedText // ignore: cast_nullable_to_non_nullable
                  as String?,
        details: freezed == details
            ? _value._details
            : details // ignore: cast_nullable_to_non_nullable
                  as Map<String, dynamic>?,
      ),
    );
  }
}

/// @nodoc
@JsonSerializable()
class _$ScoreAttemptImpl implements _ScoreAttempt {
  const _$ScoreAttemptImpl({
    required this.id,
    required this.textSequenceId,
    required this.gradedAt,
    required this.score,
    required this.method,
    this.recognizedText,
    final Map<String, dynamic>? details,
  }) : _details = details;

  factory _$ScoreAttemptImpl.fromJson(Map<String, dynamic> json) =>
      _$$ScoreAttemptImplFromJson(json);

  @override
  final String id;
  @override
  final String textSequenceId;
  @override
  final DateTime gradedAt;
  @override
  final int score;
  @override
  final String method;
  @override
  final String? recognizedText;
  final Map<String, dynamic>? _details;
  @override
  Map<String, dynamic>? get details {
    final value = _details;
    if (value == null) return null;
    if (_details is EqualUnmodifiableMapView) return _details;
    // ignore: implicit_dynamic_type
    return EqualUnmodifiableMapView(value);
  }

  @override
  String toString() {
    return 'ScoreAttempt(id: $id, textSequenceId: $textSequenceId, gradedAt: $gradedAt, score: $score, method: $method, recognizedText: $recognizedText, details: $details)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$ScoreAttemptImpl &&
            (identical(other.id, id) || other.id == id) &&
            (identical(other.textSequenceId, textSequenceId) ||
                other.textSequenceId == textSequenceId) &&
            (identical(other.gradedAt, gradedAt) ||
                other.gradedAt == gradedAt) &&
            (identical(other.score, score) || other.score == score) &&
            (identical(other.method, method) || other.method == method) &&
            (identical(other.recognizedText, recognizedText) ||
                other.recognizedText == recognizedText) &&
            const DeepCollectionEquality().equals(other._details, _details));
  }

  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  int get hashCode => Object.hash(
    runtimeType,
    id,
    textSequenceId,
    gradedAt,
    score,
    method,
    recognizedText,
    const DeepCollectionEquality().hash(_details),
  );

  /// Create a copy of ScoreAttempt
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$ScoreAttemptImplCopyWith<_$ScoreAttemptImpl> get copyWith =>
      __$$ScoreAttemptImplCopyWithImpl<_$ScoreAttemptImpl>(this, _$identity);

  @override
  Map<String, dynamic> toJson() {
    return _$$ScoreAttemptImplToJson(this);
  }
}

abstract class _ScoreAttempt implements ScoreAttempt {
  const factory _ScoreAttempt({
    required final String id,
    required final String textSequenceId,
    required final DateTime gradedAt,
    required final int score,
    required final String method,
    final String? recognizedText,
    final Map<String, dynamic>? details,
  }) = _$ScoreAttemptImpl;

  factory _ScoreAttempt.fromJson(Map<String, dynamic> json) =
      _$ScoreAttemptImpl.fromJson;

  @override
  String get id;
  @override
  String get textSequenceId;
  @override
  DateTime get gradedAt;
  @override
  int get score;
  @override
  String get method;
  @override
  String? get recognizedText;
  @override
  Map<String, dynamic>? get details;

  /// Create a copy of ScoreAttempt
  /// with the given fields replaced by the non-null parameter values.
  @override
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$ScoreAttemptImplCopyWith<_$ScoreAttemptImpl> get copyWith =>
      throw _privateConstructorUsedError;
}
