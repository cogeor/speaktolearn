// coverage:ignore-file
// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'recording_state.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

T _$identity<T>(T value) => value;

final _privateConstructorUsedError = UnsupportedError(
  'It seems like you constructed your class using `MyClass._()`. This constructor is only meant to be used by freezed and you are not supposed to need it nor use it.\nPlease check the documentation here for more information: https://github.com/rrousselGit/freezed#adding-getters-and-methods-to-our-models',
);

/// @nodoc
mixin _$RecordingState {
  bool get isRecording => throw _privateConstructorUsedError;
  bool get isScoring => throw _privateConstructorUsedError;
  bool get isPlaying => throw _privateConstructorUsedError;
  bool get hasLatestRecording => throw _privateConstructorUsedError;
  String? get error => throw _privateConstructorUsedError;

  /// Create a copy of RecordingState
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  $RecordingStateCopyWith<RecordingState> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class $RecordingStateCopyWith<$Res> {
  factory $RecordingStateCopyWith(
    RecordingState value,
    $Res Function(RecordingState) then,
  ) = _$RecordingStateCopyWithImpl<$Res, RecordingState>;
  @useResult
  $Res call({
    bool isRecording,
    bool isScoring,
    bool isPlaying,
    bool hasLatestRecording,
    String? error,
  });
}

/// @nodoc
class _$RecordingStateCopyWithImpl<$Res, $Val extends RecordingState>
    implements $RecordingStateCopyWith<$Res> {
  _$RecordingStateCopyWithImpl(this._value, this._then);

  // ignore: unused_field
  final $Val _value;
  // ignore: unused_field
  final $Res Function($Val) _then;

  /// Create a copy of RecordingState
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? isRecording = null,
    Object? isScoring = null,
    Object? isPlaying = null,
    Object? hasLatestRecording = null,
    Object? error = freezed,
  }) {
    return _then(
      _value.copyWith(
            isRecording: null == isRecording
                ? _value.isRecording
                : isRecording // ignore: cast_nullable_to_non_nullable
                      as bool,
            isScoring: null == isScoring
                ? _value.isScoring
                : isScoring // ignore: cast_nullable_to_non_nullable
                      as bool,
            isPlaying: null == isPlaying
                ? _value.isPlaying
                : isPlaying // ignore: cast_nullable_to_non_nullable
                      as bool,
            hasLatestRecording: null == hasLatestRecording
                ? _value.hasLatestRecording
                : hasLatestRecording // ignore: cast_nullable_to_non_nullable
                      as bool,
            error: freezed == error
                ? _value.error
                : error // ignore: cast_nullable_to_non_nullable
                      as String?,
          )
          as $Val,
    );
  }
}

/// @nodoc
abstract class _$$RecordingStateImplCopyWith<$Res>
    implements $RecordingStateCopyWith<$Res> {
  factory _$$RecordingStateImplCopyWith(
    _$RecordingStateImpl value,
    $Res Function(_$RecordingStateImpl) then,
  ) = __$$RecordingStateImplCopyWithImpl<$Res>;
  @override
  @useResult
  $Res call({
    bool isRecording,
    bool isScoring,
    bool isPlaying,
    bool hasLatestRecording,
    String? error,
  });
}

/// @nodoc
class __$$RecordingStateImplCopyWithImpl<$Res>
    extends _$RecordingStateCopyWithImpl<$Res, _$RecordingStateImpl>
    implements _$$RecordingStateImplCopyWith<$Res> {
  __$$RecordingStateImplCopyWithImpl(
    _$RecordingStateImpl _value,
    $Res Function(_$RecordingStateImpl) _then,
  ) : super(_value, _then);

  /// Create a copy of RecordingState
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? isRecording = null,
    Object? isScoring = null,
    Object? isPlaying = null,
    Object? hasLatestRecording = null,
    Object? error = freezed,
  }) {
    return _then(
      _$RecordingStateImpl(
        isRecording: null == isRecording
            ? _value.isRecording
            : isRecording // ignore: cast_nullable_to_non_nullable
                  as bool,
        isScoring: null == isScoring
            ? _value.isScoring
            : isScoring // ignore: cast_nullable_to_non_nullable
                  as bool,
        isPlaying: null == isPlaying
            ? _value.isPlaying
            : isPlaying // ignore: cast_nullable_to_non_nullable
                  as bool,
        hasLatestRecording: null == hasLatestRecording
            ? _value.hasLatestRecording
            : hasLatestRecording // ignore: cast_nullable_to_non_nullable
                  as bool,
        error: freezed == error
            ? _value.error
            : error // ignore: cast_nullable_to_non_nullable
                  as String?,
      ),
    );
  }
}

/// @nodoc

class _$RecordingStateImpl implements _RecordingState {
  const _$RecordingStateImpl({
    this.isRecording = false,
    this.isScoring = false,
    this.isPlaying = false,
    this.hasLatestRecording = false,
    this.error,
  });

  @override
  @JsonKey()
  final bool isRecording;
  @override
  @JsonKey()
  final bool isScoring;
  @override
  @JsonKey()
  final bool isPlaying;
  @override
  @JsonKey()
  final bool hasLatestRecording;
  @override
  final String? error;

  @override
  String toString() {
    return 'RecordingState(isRecording: $isRecording, isScoring: $isScoring, isPlaying: $isPlaying, hasLatestRecording: $hasLatestRecording, error: $error)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$RecordingStateImpl &&
            (identical(other.isRecording, isRecording) ||
                other.isRecording == isRecording) &&
            (identical(other.isScoring, isScoring) ||
                other.isScoring == isScoring) &&
            (identical(other.isPlaying, isPlaying) ||
                other.isPlaying == isPlaying) &&
            (identical(other.hasLatestRecording, hasLatestRecording) ||
                other.hasLatestRecording == hasLatestRecording) &&
            (identical(other.error, error) || other.error == error));
  }

  @override
  int get hashCode => Object.hash(
    runtimeType,
    isRecording,
    isScoring,
    isPlaying,
    hasLatestRecording,
    error,
  );

  /// Create a copy of RecordingState
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$RecordingStateImplCopyWith<_$RecordingStateImpl> get copyWith =>
      __$$RecordingStateImplCopyWithImpl<_$RecordingStateImpl>(
        this,
        _$identity,
      );
}

abstract class _RecordingState implements RecordingState {
  const factory _RecordingState({
    final bool isRecording,
    final bool isScoring,
    final bool isPlaying,
    final bool hasLatestRecording,
    final String? error,
  }) = _$RecordingStateImpl;

  @override
  bool get isRecording;
  @override
  bool get isScoring;
  @override
  bool get isPlaying;
  @override
  bool get hasLatestRecording;
  @override
  String? get error;

  /// Create a copy of RecordingState
  /// with the given fields replaced by the non-null parameter values.
  @override
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$RecordingStateImplCopyWith<_$RecordingStateImpl> get copyWith =>
      throw _privateConstructorUsedError;
}
