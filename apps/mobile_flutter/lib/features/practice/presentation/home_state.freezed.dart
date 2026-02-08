// coverage:ignore-file
// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'home_state.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

T _$identity<T>(T value) => value;

final _privateConstructorUsedError = UnsupportedError(
  'It seems like you constructed your class using `MyClass._()`. This constructor is only meant to be used by freezed and you are not supposed to need it nor use it.\nPlease check the documentation here for more information: https://github.com/rrousselGit/freezed#adding-getters-and-methods-to-our-models',
);

/// @nodoc
mixin _$HomeState {
  TextSequence? get current => throw _privateConstructorUsedError;
  TextSequenceProgress? get currentProgress =>
      throw _privateConstructorUsedError;
  bool get isLoading => throw _privateConstructorUsedError;

  /// Create a copy of HomeState
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  $HomeStateCopyWith<HomeState> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class $HomeStateCopyWith<$Res> {
  factory $HomeStateCopyWith(HomeState value, $Res Function(HomeState) then) =
      _$HomeStateCopyWithImpl<$Res, HomeState>;
  @useResult
  $Res call({
    TextSequence? current,
    TextSequenceProgress? currentProgress,
    bool isLoading,
  });

  $TextSequenceCopyWith<$Res>? get current;
  $TextSequenceProgressCopyWith<$Res>? get currentProgress;
}

/// @nodoc
class _$HomeStateCopyWithImpl<$Res, $Val extends HomeState>
    implements $HomeStateCopyWith<$Res> {
  _$HomeStateCopyWithImpl(this._value, this._then);

  // ignore: unused_field
  final $Val _value;
  // ignore: unused_field
  final $Res Function($Val) _then;

  /// Create a copy of HomeState
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? current = freezed,
    Object? currentProgress = freezed,
    Object? isLoading = null,
  }) {
    return _then(
      _value.copyWith(
            current: freezed == current
                ? _value.current
                : current // ignore: cast_nullable_to_non_nullable
                      as TextSequence?,
            currentProgress: freezed == currentProgress
                ? _value.currentProgress
                : currentProgress // ignore: cast_nullable_to_non_nullable
                      as TextSequenceProgress?,
            isLoading: null == isLoading
                ? _value.isLoading
                : isLoading // ignore: cast_nullable_to_non_nullable
                      as bool,
          )
          as $Val,
    );
  }

  /// Create a copy of HomeState
  /// with the given fields replaced by the non-null parameter values.
  @override
  @pragma('vm:prefer-inline')
  $TextSequenceCopyWith<$Res>? get current {
    if (_value.current == null) {
      return null;
    }

    return $TextSequenceCopyWith<$Res>(_value.current!, (value) {
      return _then(_value.copyWith(current: value) as $Val);
    });
  }

  /// Create a copy of HomeState
  /// with the given fields replaced by the non-null parameter values.
  @override
  @pragma('vm:prefer-inline')
  $TextSequenceProgressCopyWith<$Res>? get currentProgress {
    if (_value.currentProgress == null) {
      return null;
    }

    return $TextSequenceProgressCopyWith<$Res>(_value.currentProgress!, (
      value,
    ) {
      return _then(_value.copyWith(currentProgress: value) as $Val);
    });
  }
}

/// @nodoc
abstract class _$$HomeStateImplCopyWith<$Res>
    implements $HomeStateCopyWith<$Res> {
  factory _$$HomeStateImplCopyWith(
    _$HomeStateImpl value,
    $Res Function(_$HomeStateImpl) then,
  ) = __$$HomeStateImplCopyWithImpl<$Res>;
  @override
  @useResult
  $Res call({
    TextSequence? current,
    TextSequenceProgress? currentProgress,
    bool isLoading,
  });

  @override
  $TextSequenceCopyWith<$Res>? get current;
  @override
  $TextSequenceProgressCopyWith<$Res>? get currentProgress;
}

/// @nodoc
class __$$HomeStateImplCopyWithImpl<$Res>
    extends _$HomeStateCopyWithImpl<$Res, _$HomeStateImpl>
    implements _$$HomeStateImplCopyWith<$Res> {
  __$$HomeStateImplCopyWithImpl(
    _$HomeStateImpl _value,
    $Res Function(_$HomeStateImpl) _then,
  ) : super(_value, _then);

  /// Create a copy of HomeState
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? current = freezed,
    Object? currentProgress = freezed,
    Object? isLoading = null,
  }) {
    return _then(
      _$HomeStateImpl(
        current: freezed == current
            ? _value.current
            : current // ignore: cast_nullable_to_non_nullable
                  as TextSequence?,
        currentProgress: freezed == currentProgress
            ? _value.currentProgress
            : currentProgress // ignore: cast_nullable_to_non_nullable
                  as TextSequenceProgress?,
        isLoading: null == isLoading
            ? _value.isLoading
            : isLoading // ignore: cast_nullable_to_non_nullable
                  as bool,
      ),
    );
  }
}

/// @nodoc

class _$HomeStateImpl implements _HomeState {
  const _$HomeStateImpl({
    this.current,
    this.currentProgress,
    this.isLoading = true,
  });

  @override
  final TextSequence? current;
  @override
  final TextSequenceProgress? currentProgress;
  @override
  @JsonKey()
  final bool isLoading;

  @override
  String toString() {
    return 'HomeState(current: $current, currentProgress: $currentProgress, isLoading: $isLoading)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$HomeStateImpl &&
            (identical(other.current, current) || other.current == current) &&
            (identical(other.currentProgress, currentProgress) ||
                other.currentProgress == currentProgress) &&
            (identical(other.isLoading, isLoading) ||
                other.isLoading == isLoading));
  }

  @override
  int get hashCode =>
      Object.hash(runtimeType, current, currentProgress, isLoading);

  /// Create a copy of HomeState
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$HomeStateImplCopyWith<_$HomeStateImpl> get copyWith =>
      __$$HomeStateImplCopyWithImpl<_$HomeStateImpl>(this, _$identity);
}

abstract class _HomeState implements HomeState {
  const factory _HomeState({
    final TextSequence? current,
    final TextSequenceProgress? currentProgress,
    final bool isLoading,
  }) = _$HomeStateImpl;

  @override
  TextSequence? get current;
  @override
  TextSequenceProgress? get currentProgress;
  @override
  bool get isLoading;

  /// Create a copy of HomeState
  /// with the given fields replaced by the non-null parameter values.
  @override
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$HomeStateImplCopyWith<_$HomeStateImpl> get copyWith =>
      throw _privateConstructorUsedError;
}
