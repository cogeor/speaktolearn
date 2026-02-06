// coverage:ignore-file
// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'example_audio_controller.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

T _$identity<T>(T value) => value;

final _privateConstructorUsedError = UnsupportedError(
  'It seems like you constructed your class using `MyClass._()`. This constructor is only meant to be used by freezed and you are not supposed to need it nor use it.\nPlease check the documentation here for more information: https://github.com/rrousselGit/freezed#adding-getters-and-methods-to-our-models',
);

/// @nodoc
mixin _$ExampleAudioState {
  bool get isPlaying => throw _privateConstructorUsedError;
  String? get currentSequenceId => throw _privateConstructorUsedError;
  String? get currentVoiceId => throw _privateConstructorUsedError;

  /// Create a copy of ExampleAudioState
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  $ExampleAudioStateCopyWith<ExampleAudioState> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class $ExampleAudioStateCopyWith<$Res> {
  factory $ExampleAudioStateCopyWith(
    ExampleAudioState value,
    $Res Function(ExampleAudioState) then,
  ) = _$ExampleAudioStateCopyWithImpl<$Res, ExampleAudioState>;
  @useResult
  $Res call({
    bool isPlaying,
    String? currentSequenceId,
    String? currentVoiceId,
  });
}

/// @nodoc
class _$ExampleAudioStateCopyWithImpl<$Res, $Val extends ExampleAudioState>
    implements $ExampleAudioStateCopyWith<$Res> {
  _$ExampleAudioStateCopyWithImpl(this._value, this._then);

  // ignore: unused_field
  final $Val _value;
  // ignore: unused_field
  final $Res Function($Val) _then;

  /// Create a copy of ExampleAudioState
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? isPlaying = null,
    Object? currentSequenceId = freezed,
    Object? currentVoiceId = freezed,
  }) {
    return _then(
      _value.copyWith(
            isPlaying: null == isPlaying
                ? _value.isPlaying
                : isPlaying // ignore: cast_nullable_to_non_nullable
                      as bool,
            currentSequenceId: freezed == currentSequenceId
                ? _value.currentSequenceId
                : currentSequenceId // ignore: cast_nullable_to_non_nullable
                      as String?,
            currentVoiceId: freezed == currentVoiceId
                ? _value.currentVoiceId
                : currentVoiceId // ignore: cast_nullable_to_non_nullable
                      as String?,
          )
          as $Val,
    );
  }
}

/// @nodoc
abstract class _$$ExampleAudioStateImplCopyWith<$Res>
    implements $ExampleAudioStateCopyWith<$Res> {
  factory _$$ExampleAudioStateImplCopyWith(
    _$ExampleAudioStateImpl value,
    $Res Function(_$ExampleAudioStateImpl) then,
  ) = __$$ExampleAudioStateImplCopyWithImpl<$Res>;
  @override
  @useResult
  $Res call({
    bool isPlaying,
    String? currentSequenceId,
    String? currentVoiceId,
  });
}

/// @nodoc
class __$$ExampleAudioStateImplCopyWithImpl<$Res>
    extends _$ExampleAudioStateCopyWithImpl<$Res, _$ExampleAudioStateImpl>
    implements _$$ExampleAudioStateImplCopyWith<$Res> {
  __$$ExampleAudioStateImplCopyWithImpl(
    _$ExampleAudioStateImpl _value,
    $Res Function(_$ExampleAudioStateImpl) _then,
  ) : super(_value, _then);

  /// Create a copy of ExampleAudioState
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? isPlaying = null,
    Object? currentSequenceId = freezed,
    Object? currentVoiceId = freezed,
  }) {
    return _then(
      _$ExampleAudioStateImpl(
        isPlaying: null == isPlaying
            ? _value.isPlaying
            : isPlaying // ignore: cast_nullable_to_non_nullable
                  as bool,
        currentSequenceId: freezed == currentSequenceId
            ? _value.currentSequenceId
            : currentSequenceId // ignore: cast_nullable_to_non_nullable
                  as String?,
        currentVoiceId: freezed == currentVoiceId
            ? _value.currentVoiceId
            : currentVoiceId // ignore: cast_nullable_to_non_nullable
                  as String?,
      ),
    );
  }
}

/// @nodoc

class _$ExampleAudioStateImpl implements _ExampleAudioState {
  const _$ExampleAudioStateImpl({
    this.isPlaying = false,
    this.currentSequenceId,
    this.currentVoiceId,
  });

  @override
  @JsonKey()
  final bool isPlaying;
  @override
  final String? currentSequenceId;
  @override
  final String? currentVoiceId;

  @override
  String toString() {
    return 'ExampleAudioState(isPlaying: $isPlaying, currentSequenceId: $currentSequenceId, currentVoiceId: $currentVoiceId)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$ExampleAudioStateImpl &&
            (identical(other.isPlaying, isPlaying) ||
                other.isPlaying == isPlaying) &&
            (identical(other.currentSequenceId, currentSequenceId) ||
                other.currentSequenceId == currentSequenceId) &&
            (identical(other.currentVoiceId, currentVoiceId) ||
                other.currentVoiceId == currentVoiceId));
  }

  @override
  int get hashCode =>
      Object.hash(runtimeType, isPlaying, currentSequenceId, currentVoiceId);

  /// Create a copy of ExampleAudioState
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$ExampleAudioStateImplCopyWith<_$ExampleAudioStateImpl> get copyWith =>
      __$$ExampleAudioStateImplCopyWithImpl<_$ExampleAudioStateImpl>(
        this,
        _$identity,
      );
}

abstract class _ExampleAudioState implements ExampleAudioState {
  const factory _ExampleAudioState({
    final bool isPlaying,
    final String? currentSequenceId,
    final String? currentVoiceId,
  }) = _$ExampleAudioStateImpl;

  @override
  bool get isPlaying;
  @override
  String? get currentSequenceId;
  @override
  String? get currentVoiceId;

  /// Create a copy of ExampleAudioState
  /// with the given fields replaced by the non-null parameter values.
  @override
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$ExampleAudioStateImplCopyWith<_$ExampleAudioStateImpl> get copyWith =>
      throw _privateConstructorUsedError;
}
