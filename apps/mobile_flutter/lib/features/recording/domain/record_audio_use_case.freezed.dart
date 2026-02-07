// coverage:ignore-file
// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'record_audio_use_case.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

T _$identity<T>(T value) => value;

final _privateConstructorUsedError = UnsupportedError(
  'It seems like you constructed your class using `MyClass._()`. This constructor is only meant to be used by freezed and you are not supposed to need it nor use it.\nPlease check the documentation here for more information: https://github.com/rrousselGit/freezed#adding-getters-and-methods-to-our-models',
);

/// @nodoc
mixin _$RecordAudioParams {
  String get textSequenceId => throw _privateConstructorUsedError;

  /// Create a copy of RecordAudioParams
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  $RecordAudioParamsCopyWith<RecordAudioParams> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class $RecordAudioParamsCopyWith<$Res> {
  factory $RecordAudioParamsCopyWith(
    RecordAudioParams value,
    $Res Function(RecordAudioParams) then,
  ) = _$RecordAudioParamsCopyWithImpl<$Res, RecordAudioParams>;
  @useResult
  $Res call({String textSequenceId});
}

/// @nodoc
class _$RecordAudioParamsCopyWithImpl<$Res, $Val extends RecordAudioParams>
    implements $RecordAudioParamsCopyWith<$Res> {
  _$RecordAudioParamsCopyWithImpl(this._value, this._then);

  // ignore: unused_field
  final $Val _value;
  // ignore: unused_field
  final $Res Function($Val) _then;

  /// Create a copy of RecordAudioParams
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({Object? textSequenceId = null}) {
    return _then(
      _value.copyWith(
            textSequenceId: null == textSequenceId
                ? _value.textSequenceId
                : textSequenceId // ignore: cast_nullable_to_non_nullable
                      as String,
          )
          as $Val,
    );
  }
}

/// @nodoc
abstract class _$$RecordAudioParamsImplCopyWith<$Res>
    implements $RecordAudioParamsCopyWith<$Res> {
  factory _$$RecordAudioParamsImplCopyWith(
    _$RecordAudioParamsImpl value,
    $Res Function(_$RecordAudioParamsImpl) then,
  ) = __$$RecordAudioParamsImplCopyWithImpl<$Res>;
  @override
  @useResult
  $Res call({String textSequenceId});
}

/// @nodoc
class __$$RecordAudioParamsImplCopyWithImpl<$Res>
    extends _$RecordAudioParamsCopyWithImpl<$Res, _$RecordAudioParamsImpl>
    implements _$$RecordAudioParamsImplCopyWith<$Res> {
  __$$RecordAudioParamsImplCopyWithImpl(
    _$RecordAudioParamsImpl _value,
    $Res Function(_$RecordAudioParamsImpl) _then,
  ) : super(_value, _then);

  /// Create a copy of RecordAudioParams
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({Object? textSequenceId = null}) {
    return _then(
      _$RecordAudioParamsImpl(
        textSequenceId: null == textSequenceId
            ? _value.textSequenceId
            : textSequenceId // ignore: cast_nullable_to_non_nullable
                  as String,
      ),
    );
  }
}

/// @nodoc

class _$RecordAudioParamsImpl implements _RecordAudioParams {
  const _$RecordAudioParamsImpl({required this.textSequenceId});

  @override
  final String textSequenceId;

  @override
  String toString() {
    return 'RecordAudioParams(textSequenceId: $textSequenceId)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$RecordAudioParamsImpl &&
            (identical(other.textSequenceId, textSequenceId) ||
                other.textSequenceId == textSequenceId));
  }

  @override
  int get hashCode => Object.hash(runtimeType, textSequenceId);

  /// Create a copy of RecordAudioParams
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$RecordAudioParamsImplCopyWith<_$RecordAudioParamsImpl> get copyWith =>
      __$$RecordAudioParamsImplCopyWithImpl<_$RecordAudioParamsImpl>(
        this,
        _$identity,
      );
}

abstract class _RecordAudioParams implements RecordAudioParams {
  const factory _RecordAudioParams({required final String textSequenceId}) =
      _$RecordAudioParamsImpl;

  @override
  String get textSequenceId;

  /// Create a copy of RecordAudioParams
  /// with the given fields replaced by the non-null parameter values.
  @override
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$RecordAudioParamsImplCopyWith<_$RecordAudioParamsImpl> get copyWith =>
      throw _privateConstructorUsedError;
}
