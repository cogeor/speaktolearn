// coverage:ignore-file
// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'text_sequence.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

T _$identity<T>(T value) => value;

final _privateConstructorUsedError = UnsupportedError(
  'It seems like you constructed your class using `MyClass._()`. This constructor is only meant to be used by freezed and you are not supposed to need it nor use it.\nPlease check the documentation here for more information: https://github.com/rrousselGit/freezed#adding-getters-and-methods-to-our-models',
);

TextSequence _$TextSequenceFromJson(Map<String, dynamic> json) {
  return _TextSequence.fromJson(json);
}

/// @nodoc
mixin _$TextSequence {
  String get id => throw _privateConstructorUsedError;
  String get text => throw _privateConstructorUsedError;
  String get language => throw _privateConstructorUsedError;
  String? get romanization => throw _privateConstructorUsedError;
  Map<String, String>? get gloss => throw _privateConstructorUsedError;
  List<String>? get tokens => throw _privateConstructorUsedError;
  List<String>? get tags => throw _privateConstructorUsedError;
  int? get difficulty => throw _privateConstructorUsedError;
  List<ExampleVoice>? get voices => throw _privateConstructorUsedError;

  /// Serializes this TextSequence to a JSON map.
  Map<String, dynamic> toJson() => throw _privateConstructorUsedError;

  /// Create a copy of TextSequence
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  $TextSequenceCopyWith<TextSequence> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class $TextSequenceCopyWith<$Res> {
  factory $TextSequenceCopyWith(
    TextSequence value,
    $Res Function(TextSequence) then,
  ) = _$TextSequenceCopyWithImpl<$Res, TextSequence>;
  @useResult
  $Res call({
    String id,
    String text,
    String language,
    String? romanization,
    Map<String, String>? gloss,
    List<String>? tokens,
    List<String>? tags,
    int? difficulty,
    List<ExampleVoice>? voices,
  });
}

/// @nodoc
class _$TextSequenceCopyWithImpl<$Res, $Val extends TextSequence>
    implements $TextSequenceCopyWith<$Res> {
  _$TextSequenceCopyWithImpl(this._value, this._then);

  // ignore: unused_field
  final $Val _value;
  // ignore: unused_field
  final $Res Function($Val) _then;

  /// Create a copy of TextSequence
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? id = null,
    Object? text = null,
    Object? language = null,
    Object? romanization = freezed,
    Object? gloss = freezed,
    Object? tokens = freezed,
    Object? tags = freezed,
    Object? difficulty = freezed,
    Object? voices = freezed,
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
            language: null == language
                ? _value.language
                : language // ignore: cast_nullable_to_non_nullable
                      as String,
            romanization: freezed == romanization
                ? _value.romanization
                : romanization // ignore: cast_nullable_to_non_nullable
                      as String?,
            gloss: freezed == gloss
                ? _value.gloss
                : gloss // ignore: cast_nullable_to_non_nullable
                      as Map<String, String>?,
            tokens: freezed == tokens
                ? _value.tokens
                : tokens // ignore: cast_nullable_to_non_nullable
                      as List<String>?,
            tags: freezed == tags
                ? _value.tags
                : tags // ignore: cast_nullable_to_non_nullable
                      as List<String>?,
            difficulty: freezed == difficulty
                ? _value.difficulty
                : difficulty // ignore: cast_nullable_to_non_nullable
                      as int?,
            voices: freezed == voices
                ? _value.voices
                : voices // ignore: cast_nullable_to_non_nullable
                      as List<ExampleVoice>?,
          )
          as $Val,
    );
  }
}

/// @nodoc
abstract class _$$TextSequenceImplCopyWith<$Res>
    implements $TextSequenceCopyWith<$Res> {
  factory _$$TextSequenceImplCopyWith(
    _$TextSequenceImpl value,
    $Res Function(_$TextSequenceImpl) then,
  ) = __$$TextSequenceImplCopyWithImpl<$Res>;
  @override
  @useResult
  $Res call({
    String id,
    String text,
    String language,
    String? romanization,
    Map<String, String>? gloss,
    List<String>? tokens,
    List<String>? tags,
    int? difficulty,
    List<ExampleVoice>? voices,
  });
}

/// @nodoc
class __$$TextSequenceImplCopyWithImpl<$Res>
    extends _$TextSequenceCopyWithImpl<$Res, _$TextSequenceImpl>
    implements _$$TextSequenceImplCopyWith<$Res> {
  __$$TextSequenceImplCopyWithImpl(
    _$TextSequenceImpl _value,
    $Res Function(_$TextSequenceImpl) _then,
  ) : super(_value, _then);

  /// Create a copy of TextSequence
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? id = null,
    Object? text = null,
    Object? language = null,
    Object? romanization = freezed,
    Object? gloss = freezed,
    Object? tokens = freezed,
    Object? tags = freezed,
    Object? difficulty = freezed,
    Object? voices = freezed,
  }) {
    return _then(
      _$TextSequenceImpl(
        id: null == id
            ? _value.id
            : id // ignore: cast_nullable_to_non_nullable
                  as String,
        text: null == text
            ? _value.text
            : text // ignore: cast_nullable_to_non_nullable
                  as String,
        language: null == language
            ? _value.language
            : language // ignore: cast_nullable_to_non_nullable
                  as String,
        romanization: freezed == romanization
            ? _value.romanization
            : romanization // ignore: cast_nullable_to_non_nullable
                  as String?,
        gloss: freezed == gloss
            ? _value._gloss
            : gloss // ignore: cast_nullable_to_non_nullable
                  as Map<String, String>?,
        tokens: freezed == tokens
            ? _value._tokens
            : tokens // ignore: cast_nullable_to_non_nullable
                  as List<String>?,
        tags: freezed == tags
            ? _value._tags
            : tags // ignore: cast_nullable_to_non_nullable
                  as List<String>?,
        difficulty: freezed == difficulty
            ? _value.difficulty
            : difficulty // ignore: cast_nullable_to_non_nullable
                  as int?,
        voices: freezed == voices
            ? _value._voices
            : voices // ignore: cast_nullable_to_non_nullable
                  as List<ExampleVoice>?,
      ),
    );
  }
}

/// @nodoc
@JsonSerializable()
class _$TextSequenceImpl implements _TextSequence {
  const _$TextSequenceImpl({
    required this.id,
    required this.text,
    required this.language,
    this.romanization,
    final Map<String, String>? gloss,
    final List<String>? tokens,
    final List<String>? tags,
    this.difficulty,
    final List<ExampleVoice>? voices,
  }) : _gloss = gloss,
       _tokens = tokens,
       _tags = tags,
       _voices = voices;

  factory _$TextSequenceImpl.fromJson(Map<String, dynamic> json) =>
      _$$TextSequenceImplFromJson(json);

  @override
  final String id;
  @override
  final String text;
  @override
  final String language;
  @override
  final String? romanization;
  final Map<String, String>? _gloss;
  @override
  Map<String, String>? get gloss {
    final value = _gloss;
    if (value == null) return null;
    if (_gloss is EqualUnmodifiableMapView) return _gloss;
    // ignore: implicit_dynamic_type
    return EqualUnmodifiableMapView(value);
  }

  final List<String>? _tokens;
  @override
  List<String>? get tokens {
    final value = _tokens;
    if (value == null) return null;
    if (_tokens is EqualUnmodifiableListView) return _tokens;
    // ignore: implicit_dynamic_type
    return EqualUnmodifiableListView(value);
  }

  final List<String>? _tags;
  @override
  List<String>? get tags {
    final value = _tags;
    if (value == null) return null;
    if (_tags is EqualUnmodifiableListView) return _tags;
    // ignore: implicit_dynamic_type
    return EqualUnmodifiableListView(value);
  }

  @override
  final int? difficulty;
  final List<ExampleVoice>? _voices;
  @override
  List<ExampleVoice>? get voices {
    final value = _voices;
    if (value == null) return null;
    if (_voices is EqualUnmodifiableListView) return _voices;
    // ignore: implicit_dynamic_type
    return EqualUnmodifiableListView(value);
  }

  @override
  String toString() {
    return 'TextSequence(id: $id, text: $text, language: $language, romanization: $romanization, gloss: $gloss, tokens: $tokens, tags: $tags, difficulty: $difficulty, voices: $voices)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$TextSequenceImpl &&
            (identical(other.id, id) || other.id == id) &&
            (identical(other.text, text) || other.text == text) &&
            (identical(other.language, language) ||
                other.language == language) &&
            (identical(other.romanization, romanization) ||
                other.romanization == romanization) &&
            const DeepCollectionEquality().equals(other._gloss, _gloss) &&
            const DeepCollectionEquality().equals(other._tokens, _tokens) &&
            const DeepCollectionEquality().equals(other._tags, _tags) &&
            (identical(other.difficulty, difficulty) ||
                other.difficulty == difficulty) &&
            const DeepCollectionEquality().equals(other._voices, _voices));
  }

  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  int get hashCode => Object.hash(
    runtimeType,
    id,
    text,
    language,
    romanization,
    const DeepCollectionEquality().hash(_gloss),
    const DeepCollectionEquality().hash(_tokens),
    const DeepCollectionEquality().hash(_tags),
    difficulty,
    const DeepCollectionEquality().hash(_voices),
  );

  /// Create a copy of TextSequence
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$TextSequenceImplCopyWith<_$TextSequenceImpl> get copyWith =>
      __$$TextSequenceImplCopyWithImpl<_$TextSequenceImpl>(this, _$identity);

  @override
  Map<String, dynamic> toJson() {
    return _$$TextSequenceImplToJson(this);
  }
}

abstract class _TextSequence implements TextSequence {
  const factory _TextSequence({
    required final String id,
    required final String text,
    required final String language,
    final String? romanization,
    final Map<String, String>? gloss,
    final List<String>? tokens,
    final List<String>? tags,
    final int? difficulty,
    final List<ExampleVoice>? voices,
  }) = _$TextSequenceImpl;

  factory _TextSequence.fromJson(Map<String, dynamic> json) =
      _$TextSequenceImpl.fromJson;

  @override
  String get id;
  @override
  String get text;
  @override
  String get language;
  @override
  String? get romanization;
  @override
  Map<String, String>? get gloss;
  @override
  List<String>? get tokens;
  @override
  List<String>? get tags;
  @override
  int? get difficulty;
  @override
  List<ExampleVoice>? get voices;

  /// Create a copy of TextSequence
  /// with the given fields replaced by the non-null parameter values.
  @override
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$TextSequenceImplCopyWith<_$TextSequenceImpl> get copyWith =>
      throw _privateConstructorUsedError;
}

ExampleVoice _$ExampleVoiceFromJson(Map<String, dynamic> json) {
  return _ExampleVoice.fromJson(json);
}

/// @nodoc
mixin _$ExampleVoice {
  String get id => throw _privateConstructorUsedError;
  Map<String, String>? get label => throw _privateConstructorUsedError;
  String get uri => throw _privateConstructorUsedError;
  int? get durationMs => throw _privateConstructorUsedError;

  /// Serializes this ExampleVoice to a JSON map.
  Map<String, dynamic> toJson() => throw _privateConstructorUsedError;

  /// Create a copy of ExampleVoice
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  $ExampleVoiceCopyWith<ExampleVoice> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class $ExampleVoiceCopyWith<$Res> {
  factory $ExampleVoiceCopyWith(
    ExampleVoice value,
    $Res Function(ExampleVoice) then,
  ) = _$ExampleVoiceCopyWithImpl<$Res, ExampleVoice>;
  @useResult
  $Res call({
    String id,
    Map<String, String>? label,
    String uri,
    int? durationMs,
  });
}

/// @nodoc
class _$ExampleVoiceCopyWithImpl<$Res, $Val extends ExampleVoice>
    implements $ExampleVoiceCopyWith<$Res> {
  _$ExampleVoiceCopyWithImpl(this._value, this._then);

  // ignore: unused_field
  final $Val _value;
  // ignore: unused_field
  final $Res Function($Val) _then;

  /// Create a copy of ExampleVoice
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? id = null,
    Object? label = freezed,
    Object? uri = null,
    Object? durationMs = freezed,
  }) {
    return _then(
      _value.copyWith(
            id: null == id
                ? _value.id
                : id // ignore: cast_nullable_to_non_nullable
                      as String,
            label: freezed == label
                ? _value.label
                : label // ignore: cast_nullable_to_non_nullable
                      as Map<String, String>?,
            uri: null == uri
                ? _value.uri
                : uri // ignore: cast_nullable_to_non_nullable
                      as String,
            durationMs: freezed == durationMs
                ? _value.durationMs
                : durationMs // ignore: cast_nullable_to_non_nullable
                      as int?,
          )
          as $Val,
    );
  }
}

/// @nodoc
abstract class _$$ExampleVoiceImplCopyWith<$Res>
    implements $ExampleVoiceCopyWith<$Res> {
  factory _$$ExampleVoiceImplCopyWith(
    _$ExampleVoiceImpl value,
    $Res Function(_$ExampleVoiceImpl) then,
  ) = __$$ExampleVoiceImplCopyWithImpl<$Res>;
  @override
  @useResult
  $Res call({
    String id,
    Map<String, String>? label,
    String uri,
    int? durationMs,
  });
}

/// @nodoc
class __$$ExampleVoiceImplCopyWithImpl<$Res>
    extends _$ExampleVoiceCopyWithImpl<$Res, _$ExampleVoiceImpl>
    implements _$$ExampleVoiceImplCopyWith<$Res> {
  __$$ExampleVoiceImplCopyWithImpl(
    _$ExampleVoiceImpl _value,
    $Res Function(_$ExampleVoiceImpl) _then,
  ) : super(_value, _then);

  /// Create a copy of ExampleVoice
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? id = null,
    Object? label = freezed,
    Object? uri = null,
    Object? durationMs = freezed,
  }) {
    return _then(
      _$ExampleVoiceImpl(
        id: null == id
            ? _value.id
            : id // ignore: cast_nullable_to_non_nullable
                  as String,
        label: freezed == label
            ? _value._label
            : label // ignore: cast_nullable_to_non_nullable
                  as Map<String, String>?,
        uri: null == uri
            ? _value.uri
            : uri // ignore: cast_nullable_to_non_nullable
                  as String,
        durationMs: freezed == durationMs
            ? _value.durationMs
            : durationMs // ignore: cast_nullable_to_non_nullable
                  as int?,
      ),
    );
  }
}

/// @nodoc
@JsonSerializable()
class _$ExampleVoiceImpl implements _ExampleVoice {
  const _$ExampleVoiceImpl({
    required this.id,
    final Map<String, String>? label,
    required this.uri,
    this.durationMs,
  }) : _label = label;

  factory _$ExampleVoiceImpl.fromJson(Map<String, dynamic> json) =>
      _$$ExampleVoiceImplFromJson(json);

  @override
  final String id;
  final Map<String, String>? _label;
  @override
  Map<String, String>? get label {
    final value = _label;
    if (value == null) return null;
    if (_label is EqualUnmodifiableMapView) return _label;
    // ignore: implicit_dynamic_type
    return EqualUnmodifiableMapView(value);
  }

  @override
  final String uri;
  @override
  final int? durationMs;

  @override
  String toString() {
    return 'ExampleVoice(id: $id, label: $label, uri: $uri, durationMs: $durationMs)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$ExampleVoiceImpl &&
            (identical(other.id, id) || other.id == id) &&
            const DeepCollectionEquality().equals(other._label, _label) &&
            (identical(other.uri, uri) || other.uri == uri) &&
            (identical(other.durationMs, durationMs) ||
                other.durationMs == durationMs));
  }

  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  int get hashCode => Object.hash(
    runtimeType,
    id,
    const DeepCollectionEquality().hash(_label),
    uri,
    durationMs,
  );

  /// Create a copy of ExampleVoice
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$ExampleVoiceImplCopyWith<_$ExampleVoiceImpl> get copyWith =>
      __$$ExampleVoiceImplCopyWithImpl<_$ExampleVoiceImpl>(this, _$identity);

  @override
  Map<String, dynamic> toJson() {
    return _$$ExampleVoiceImplToJson(this);
  }
}

abstract class _ExampleVoice implements ExampleVoice {
  const factory _ExampleVoice({
    required final String id,
    final Map<String, String>? label,
    required final String uri,
    final int? durationMs,
  }) = _$ExampleVoiceImpl;

  factory _ExampleVoice.fromJson(Map<String, dynamic> json) =
      _$ExampleVoiceImpl.fromJson;

  @override
  String get id;
  @override
  Map<String, String>? get label;
  @override
  String get uri;
  @override
  int? get durationMs;

  /// Create a copy of ExampleVoice
  /// with the given fields replaced by the non-null parameter values.
  @override
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$ExampleVoiceImplCopyWith<_$ExampleVoiceImpl> get copyWith =>
      throw _privateConstructorUsedError;
}
