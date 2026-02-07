// coverage:ignore-file
// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'score_pronunciation_use_case.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

T _$identity<T>(T value) => value;

final _privateConstructorUsedError = UnsupportedError(
  'It seems like you constructed your class using `MyClass._()`. This constructor is only meant to be used by freezed and you are not supposed to need it nor use it.\nPlease check the documentation here for more information: https://github.com/rrousselGit/freezed#adding-getters-and-methods-to-our-models',
);

/// @nodoc
mixin _$ScorePronunciationParams {
  TextSequence get textSequence => throw _privateConstructorUsedError;
  Recording get recording => throw _privateConstructorUsedError;

  /// Create a copy of ScorePronunciationParams
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  $ScorePronunciationParamsCopyWith<ScorePronunciationParams> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class $ScorePronunciationParamsCopyWith<$Res> {
  factory $ScorePronunciationParamsCopyWith(
    ScorePronunciationParams value,
    $Res Function(ScorePronunciationParams) then,
  ) = _$ScorePronunciationParamsCopyWithImpl<$Res, ScorePronunciationParams>;
  @useResult
  $Res call({TextSequence textSequence, Recording recording});

  $TextSequenceCopyWith<$Res> get textSequence;
  $RecordingCopyWith<$Res> get recording;
}

/// @nodoc
class _$ScorePronunciationParamsCopyWithImpl<
  $Res,
  $Val extends ScorePronunciationParams
>
    implements $ScorePronunciationParamsCopyWith<$Res> {
  _$ScorePronunciationParamsCopyWithImpl(this._value, this._then);

  // ignore: unused_field
  final $Val _value;
  // ignore: unused_field
  final $Res Function($Val) _then;

  /// Create a copy of ScorePronunciationParams
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({Object? textSequence = null, Object? recording = null}) {
    return _then(
      _value.copyWith(
            textSequence: null == textSequence
                ? _value.textSequence
                : textSequence // ignore: cast_nullable_to_non_nullable
                      as TextSequence,
            recording: null == recording
                ? _value.recording
                : recording // ignore: cast_nullable_to_non_nullable
                      as Recording,
          )
          as $Val,
    );
  }

  /// Create a copy of ScorePronunciationParams
  /// with the given fields replaced by the non-null parameter values.
  @override
  @pragma('vm:prefer-inline')
  $TextSequenceCopyWith<$Res> get textSequence {
    return $TextSequenceCopyWith<$Res>(_value.textSequence, (value) {
      return _then(_value.copyWith(textSequence: value) as $Val);
    });
  }

  /// Create a copy of ScorePronunciationParams
  /// with the given fields replaced by the non-null parameter values.
  @override
  @pragma('vm:prefer-inline')
  $RecordingCopyWith<$Res> get recording {
    return $RecordingCopyWith<$Res>(_value.recording, (value) {
      return _then(_value.copyWith(recording: value) as $Val);
    });
  }
}

/// @nodoc
abstract class _$$ScorePronunciationParamsImplCopyWith<$Res>
    implements $ScorePronunciationParamsCopyWith<$Res> {
  factory _$$ScorePronunciationParamsImplCopyWith(
    _$ScorePronunciationParamsImpl value,
    $Res Function(_$ScorePronunciationParamsImpl) then,
  ) = __$$ScorePronunciationParamsImplCopyWithImpl<$Res>;
  @override
  @useResult
  $Res call({TextSequence textSequence, Recording recording});

  @override
  $TextSequenceCopyWith<$Res> get textSequence;
  @override
  $RecordingCopyWith<$Res> get recording;
}

/// @nodoc
class __$$ScorePronunciationParamsImplCopyWithImpl<$Res>
    extends
        _$ScorePronunciationParamsCopyWithImpl<
          $Res,
          _$ScorePronunciationParamsImpl
        >
    implements _$$ScorePronunciationParamsImplCopyWith<$Res> {
  __$$ScorePronunciationParamsImplCopyWithImpl(
    _$ScorePronunciationParamsImpl _value,
    $Res Function(_$ScorePronunciationParamsImpl) _then,
  ) : super(_value, _then);

  /// Create a copy of ScorePronunciationParams
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({Object? textSequence = null, Object? recording = null}) {
    return _then(
      _$ScorePronunciationParamsImpl(
        textSequence: null == textSequence
            ? _value.textSequence
            : textSequence // ignore: cast_nullable_to_non_nullable
                  as TextSequence,
        recording: null == recording
            ? _value.recording
            : recording // ignore: cast_nullable_to_non_nullable
                  as Recording,
      ),
    );
  }
}

/// @nodoc

class _$ScorePronunciationParamsImpl implements _ScorePronunciationParams {
  const _$ScorePronunciationParamsImpl({
    required this.textSequence,
    required this.recording,
  });

  @override
  final TextSequence textSequence;
  @override
  final Recording recording;

  @override
  String toString() {
    return 'ScorePronunciationParams(textSequence: $textSequence, recording: $recording)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$ScorePronunciationParamsImpl &&
            (identical(other.textSequence, textSequence) ||
                other.textSequence == textSequence) &&
            (identical(other.recording, recording) ||
                other.recording == recording));
  }

  @override
  int get hashCode => Object.hash(runtimeType, textSequence, recording);

  /// Create a copy of ScorePronunciationParams
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$ScorePronunciationParamsImplCopyWith<_$ScorePronunciationParamsImpl>
  get copyWith =>
      __$$ScorePronunciationParamsImplCopyWithImpl<
        _$ScorePronunciationParamsImpl
      >(this, _$identity);
}

abstract class _ScorePronunciationParams implements ScorePronunciationParams {
  const factory _ScorePronunciationParams({
    required final TextSequence textSequence,
    required final Recording recording,
  }) = _$ScorePronunciationParamsImpl;

  @override
  TextSequence get textSequence;
  @override
  Recording get recording;

  /// Create a copy of ScorePronunciationParams
  /// with the given fields replaced by the non-null parameter values.
  @override
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$ScorePronunciationParamsImplCopyWith<_$ScorePronunciationParamsImpl>
  get copyWith => throw _privateConstructorUsedError;
}
