// coverage:ignore-file
// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'sequence_list_item.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

T _$identity<T>(T value) => value;

final _privateConstructorUsedError = UnsupportedError(
  'It seems like you constructed your class using `MyClass._()`. This constructor is only meant to be used by freezed and you are not supposed to need it nor use it.\nPlease check the documentation here for more information: https://github.com/rrousselGit/freezed#adding-getters-and-methods-to-our-models',
);

/// @nodoc
mixin _$SequenceListItem {
  String get id => throw _privateConstructorUsedError;
  String get text => throw _privateConstructorUsedError;
  SentenceRating? get lastRating => throw _privateConstructorUsedError;
  int? get hskLevel => throw _privateConstructorUsedError;

  /// Create a copy of SequenceListItem
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  $SequenceListItemCopyWith<SequenceListItem> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class $SequenceListItemCopyWith<$Res> {
  factory $SequenceListItemCopyWith(
    SequenceListItem value,
    $Res Function(SequenceListItem) then,
  ) = _$SequenceListItemCopyWithImpl<$Res, SequenceListItem>;
  @useResult
  $Res call({
    String id,
    String text,
    SentenceRating? lastRating,
    int? hskLevel,
  });
}

/// @nodoc
class _$SequenceListItemCopyWithImpl<$Res, $Val extends SequenceListItem>
    implements $SequenceListItemCopyWith<$Res> {
  _$SequenceListItemCopyWithImpl(this._value, this._then);

  // ignore: unused_field
  final $Val _value;
  // ignore: unused_field
  final $Res Function($Val) _then;

  /// Create a copy of SequenceListItem
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? id = null,
    Object? text = null,
    Object? lastRating = freezed,
    Object? hskLevel = freezed,
  }) {
    return _then(
      _value.copyWith(
            id: null == id
                ? _value.id
                : id // ignore: cast_nullable_to_non_nullable
                      as String,
            text: null == text
                ? _value.text
                : text // ignore: cast_nullable_to_non_nullable
                      as String,
            lastRating: freezed == lastRating
                ? _value.lastRating
                : lastRating // ignore: cast_nullable_to_non_nullable
                      as SentenceRating?,
            hskLevel: freezed == hskLevel
                ? _value.hskLevel
                : hskLevel // ignore: cast_nullable_to_non_nullable
                      as int?,
          )
          as $Val,
    );
  }
}

/// @nodoc
abstract class _$$SequenceListItemImplCopyWith<$Res>
    implements $SequenceListItemCopyWith<$Res> {
  factory _$$SequenceListItemImplCopyWith(
    _$SequenceListItemImpl value,
    $Res Function(_$SequenceListItemImpl) then,
  ) = __$$SequenceListItemImplCopyWithImpl<$Res>;
  @override
  @useResult
  $Res call({
    String id,
    String text,
    SentenceRating? lastRating,
    int? hskLevel,
  });
}

/// @nodoc
class __$$SequenceListItemImplCopyWithImpl<$Res>
    extends _$SequenceListItemCopyWithImpl<$Res, _$SequenceListItemImpl>
    implements _$$SequenceListItemImplCopyWith<$Res> {
  __$$SequenceListItemImplCopyWithImpl(
    _$SequenceListItemImpl _value,
    $Res Function(_$SequenceListItemImpl) _then,
  ) : super(_value, _then);

  /// Create a copy of SequenceListItem
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? id = null,
    Object? text = null,
    Object? lastRating = freezed,
    Object? hskLevel = freezed,
  }) {
    return _then(
      _$SequenceListItemImpl(
        id: null == id
            ? _value.id
            : id // ignore: cast_nullable_to_non_nullable
                  as String,
        text: null == text
            ? _value.text
            : text // ignore: cast_nullable_to_non_nullable
                  as String,
        lastRating: freezed == lastRating
            ? _value.lastRating
            : lastRating // ignore: cast_nullable_to_non_nullable
                  as SentenceRating?,
        hskLevel: freezed == hskLevel
            ? _value.hskLevel
            : hskLevel // ignore: cast_nullable_to_non_nullable
                  as int?,
      ),
    );
  }
}

/// @nodoc

class _$SequenceListItemImpl implements _SequenceListItem {
  const _$SequenceListItemImpl({
    required this.id,
    required this.text,
    this.lastRating,
    this.hskLevel,
  });

  @override
  final String id;
  @override
  final String text;
  @override
  final SentenceRating? lastRating;
  @override
  final int? hskLevel;

  @override
  String toString() {
    return 'SequenceListItem(id: $id, text: $text, lastRating: $lastRating, hskLevel: $hskLevel)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$SequenceListItemImpl &&
            (identical(other.id, id) || other.id == id) &&
            (identical(other.text, text) || other.text == text) &&
            (identical(other.lastRating, lastRating) ||
                other.lastRating == lastRating) &&
            (identical(other.hskLevel, hskLevel) ||
                other.hskLevel == hskLevel));
  }

  @override
  int get hashCode => Object.hash(runtimeType, id, text, lastRating, hskLevel);

  /// Create a copy of SequenceListItem
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$SequenceListItemImplCopyWith<_$SequenceListItemImpl> get copyWith =>
      __$$SequenceListItemImplCopyWithImpl<_$SequenceListItemImpl>(
        this,
        _$identity,
      );
}

abstract class _SequenceListItem implements SequenceListItem {
  const factory _SequenceListItem({
    required final String id,
    required final String text,
    final SentenceRating? lastRating,
    final int? hskLevel,
  }) = _$SequenceListItemImpl;

  @override
  String get id;
  @override
  String get text;
  @override
  SentenceRating? get lastRating;
  @override
  int? get hskLevel;

  /// Create a copy of SequenceListItem
  /// with the given fields replaced by the non-null parameter values.
  @override
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$SequenceListItemImplCopyWith<_$SequenceListItemImpl> get copyWith =>
      throw _privateConstructorUsedError;
}
