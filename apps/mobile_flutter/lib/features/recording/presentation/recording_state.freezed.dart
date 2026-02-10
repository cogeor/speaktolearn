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
  /// The current phase in the recording state machine.
  RecordingPhase get phase => throw _privateConstructorUsedError;
  bool get isPlaying => throw _privateConstructorUsedError;
  bool get hasLatestRecording => throw _privateConstructorUsedError;

  /// Whether the user has played back their recording at least once.
  /// Used to determine if rating buttons should be visible.
  bool get hasPlayedBack => throw _privateConstructorUsedError;
  String? get error => throw _privateConstructorUsedError;

  /// Remaining seconds in the auto-stop countdown. Null when not recording.
  int? get remainingSeconds => throw _privateConstructorUsedError;

  /// Total duration in seconds for progress calculation. Null when not recording.
  int? get totalDurationSeconds => throw _privateConstructorUsedError;

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
    RecordingPhase phase,
    bool isPlaying,
    bool hasLatestRecording,
    bool hasPlayedBack,
    String? error,
    int? remainingSeconds,
    int? totalDurationSeconds,
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
    Object? phase = null,
    Object? isPlaying = null,
    Object? hasLatestRecording = null,
    Object? hasPlayedBack = null,
    Object? error = freezed,
    Object? remainingSeconds = freezed,
    Object? totalDurationSeconds = freezed,
  }) {
    return _then(
      _value.copyWith(
            phase: null == phase
                ? _value.phase
                : phase // ignore: cast_nullable_to_non_nullable
                      as RecordingPhase,
            isPlaying: null == isPlaying
                ? _value.isPlaying
                : isPlaying // ignore: cast_nullable_to_non_nullable
                      as bool,
            hasLatestRecording: null == hasLatestRecording
                ? _value.hasLatestRecording
                : hasLatestRecording // ignore: cast_nullable_to_non_nullable
                      as bool,
            hasPlayedBack: null == hasPlayedBack
                ? _value.hasPlayedBack
                : hasPlayedBack // ignore: cast_nullable_to_non_nullable
                      as bool,
            error: freezed == error
                ? _value.error
                : error // ignore: cast_nullable_to_non_nullable
                      as String?,
            remainingSeconds: freezed == remainingSeconds
                ? _value.remainingSeconds
                : remainingSeconds // ignore: cast_nullable_to_non_nullable
                      as int?,
            totalDurationSeconds: freezed == totalDurationSeconds
                ? _value.totalDurationSeconds
                : totalDurationSeconds // ignore: cast_nullable_to_non_nullable
                      as int?,
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
    RecordingPhase phase,
    bool isPlaying,
    bool hasLatestRecording,
    bool hasPlayedBack,
    String? error,
    int? remainingSeconds,
    int? totalDurationSeconds,
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
    Object? phase = null,
    Object? isPlaying = null,
    Object? hasLatestRecording = null,
    Object? hasPlayedBack = null,
    Object? error = freezed,
    Object? remainingSeconds = freezed,
    Object? totalDurationSeconds = freezed,
  }) {
    return _then(
      _$RecordingStateImpl(
        phase: null == phase
            ? _value.phase
            : phase // ignore: cast_nullable_to_non_nullable
                  as RecordingPhase,
        isPlaying: null == isPlaying
            ? _value.isPlaying
            : isPlaying // ignore: cast_nullable_to_non_nullable
                  as bool,
        hasLatestRecording: null == hasLatestRecording
            ? _value.hasLatestRecording
            : hasLatestRecording // ignore: cast_nullable_to_non_nullable
                  as bool,
        hasPlayedBack: null == hasPlayedBack
            ? _value.hasPlayedBack
            : hasPlayedBack // ignore: cast_nullable_to_non_nullable
                  as bool,
        error: freezed == error
            ? _value.error
            : error // ignore: cast_nullable_to_non_nullable
                  as String?,
        remainingSeconds: freezed == remainingSeconds
            ? _value.remainingSeconds
            : remainingSeconds // ignore: cast_nullable_to_non_nullable
                  as int?,
        totalDurationSeconds: freezed == totalDurationSeconds
            ? _value.totalDurationSeconds
            : totalDurationSeconds // ignore: cast_nullable_to_non_nullable
                  as int?,
      ),
    );
  }
}

/// @nodoc

class _$RecordingStateImpl extends _RecordingState {
  const _$RecordingStateImpl({
    this.phase = RecordingPhase.idle,
    this.isPlaying = false,
    this.hasLatestRecording = false,
    this.hasPlayedBack = false,
    this.error,
    this.remainingSeconds,
    this.totalDurationSeconds,
  }) : super._();

  /// The current phase in the recording state machine.
  @override
  @JsonKey()
  final RecordingPhase phase;
  @override
  @JsonKey()
  final bool isPlaying;
  @override
  @JsonKey()
  final bool hasLatestRecording;

  /// Whether the user has played back their recording at least once.
  /// Used to determine if rating buttons should be visible.
  @override
  @JsonKey()
  final bool hasPlayedBack;
  @override
  final String? error;

  /// Remaining seconds in the auto-stop countdown. Null when not recording.
  @override
  final int? remainingSeconds;

  /// Total duration in seconds for progress calculation. Null when not recording.
  @override
  final int? totalDurationSeconds;

  @override
  String toString() {
    return 'RecordingState(phase: $phase, isPlaying: $isPlaying, hasLatestRecording: $hasLatestRecording, hasPlayedBack: $hasPlayedBack, error: $error, remainingSeconds: $remainingSeconds, totalDurationSeconds: $totalDurationSeconds)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$RecordingStateImpl &&
            (identical(other.phase, phase) || other.phase == phase) &&
            (identical(other.isPlaying, isPlaying) ||
                other.isPlaying == isPlaying) &&
            (identical(other.hasLatestRecording, hasLatestRecording) ||
                other.hasLatestRecording == hasLatestRecording) &&
            (identical(other.hasPlayedBack, hasPlayedBack) ||
                other.hasPlayedBack == hasPlayedBack) &&
            (identical(other.error, error) || other.error == error) &&
            (identical(other.remainingSeconds, remainingSeconds) ||
                other.remainingSeconds == remainingSeconds) &&
            (identical(other.totalDurationSeconds, totalDurationSeconds) ||
                other.totalDurationSeconds == totalDurationSeconds));
  }

  @override
  int get hashCode => Object.hash(
    runtimeType,
    phase,
    isPlaying,
    hasLatestRecording,
    hasPlayedBack,
    error,
    remainingSeconds,
    totalDurationSeconds,
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

abstract class _RecordingState extends RecordingState {
  const factory _RecordingState({
    final RecordingPhase phase,
    final bool isPlaying,
    final bool hasLatestRecording,
    final bool hasPlayedBack,
    final String? error,
    final int? remainingSeconds,
    final int? totalDurationSeconds,
  }) = _$RecordingStateImpl;
  const _RecordingState._() : super._();

  /// The current phase in the recording state machine.
  @override
  RecordingPhase get phase;
  @override
  bool get isPlaying;
  @override
  bool get hasLatestRecording;

  /// Whether the user has played back their recording at least once.
  /// Used to determine if rating buttons should be visible.
  @override
  bool get hasPlayedBack;
  @override
  String? get error;

  /// Remaining seconds in the auto-stop countdown. Null when not recording.
  @override
  int? get remainingSeconds;

  /// Total duration in seconds for progress calculation. Null when not recording.
  @override
  int? get totalDurationSeconds;

  /// Create a copy of RecordingState
  /// with the given fields replaced by the non-null parameter values.
  @override
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$RecordingStateImplCopyWith<_$RecordingStateImpl> get copyWith =>
      throw _privateConstructorUsedError;
}
