// coverage:ignore-file
// GENERATED CODE - DO NOT MODIFY BY HAND
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'practice_stats.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

T _$identity<T>(T value) => value;

final _privateConstructorUsedError = UnsupportedError(
  'It seems like you constructed your class using `MyClass._()`. This constructor is only meant to be used by freezed and you are not supposed to need it nor use it.\nPlease check the documentation here for more information: https://github.com/rrousselGit/freezed#adding-getters-and-methods-to-our-models',
);

/// @nodoc
mixin _$PracticeStats {
  int get totalAttempts => throw _privateConstructorUsedError;
  int get sequencesPracticed => throw _privateConstructorUsedError;
  double? get averageScore => throw _privateConstructorUsedError;
  int get currentStreak => throw _privateConstructorUsedError;
  int get longestStreak => throw _privateConstructorUsedError;
  DateTime? get lastPracticeDate => throw _privateConstructorUsedError;
  Map<DateTime, int> get dailyAttempts => throw _privateConstructorUsedError;

  /// Create a copy of PracticeStats
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  $PracticeStatsCopyWith<PracticeStats> get copyWith =>
      throw _privateConstructorUsedError;
}

/// @nodoc
abstract class $PracticeStatsCopyWith<$Res> {
  factory $PracticeStatsCopyWith(
    PracticeStats value,
    $Res Function(PracticeStats) then,
  ) = _$PracticeStatsCopyWithImpl<$Res, PracticeStats>;
  @useResult
  $Res call({
    int totalAttempts,
    int sequencesPracticed,
    double? averageScore,
    int currentStreak,
    int longestStreak,
    DateTime? lastPracticeDate,
    Map<DateTime, int> dailyAttempts,
  });
}

/// @nodoc
class _$PracticeStatsCopyWithImpl<$Res, $Val extends PracticeStats>
    implements $PracticeStatsCopyWith<$Res> {
  _$PracticeStatsCopyWithImpl(this._value, this._then);

  // ignore: unused_field
  final $Val _value;
  // ignore: unused_field
  final $Res Function($Val) _then;

  /// Create a copy of PracticeStats
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? totalAttempts = null,
    Object? sequencesPracticed = null,
    Object? averageScore = freezed,
    Object? currentStreak = null,
    Object? longestStreak = null,
    Object? lastPracticeDate = freezed,
    Object? dailyAttempts = null,
  }) {
    return _then(
      _value.copyWith(
            totalAttempts: null == totalAttempts
                ? _value.totalAttempts
                : totalAttempts // ignore: cast_nullable_to_non_nullable
                      as int,
            sequencesPracticed: null == sequencesPracticed
                ? _value.sequencesPracticed
                : sequencesPracticed // ignore: cast_nullable_to_non_nullable
                      as int,
            averageScore: freezed == averageScore
                ? _value.averageScore
                : averageScore // ignore: cast_nullable_to_non_nullable
                      as double?,
            currentStreak: null == currentStreak
                ? _value.currentStreak
                : currentStreak // ignore: cast_nullable_to_non_nullable
                      as int,
            longestStreak: null == longestStreak
                ? _value.longestStreak
                : longestStreak // ignore: cast_nullable_to_non_nullable
                      as int,
            lastPracticeDate: freezed == lastPracticeDate
                ? _value.lastPracticeDate
                : lastPracticeDate // ignore: cast_nullable_to_non_nullable
                      as DateTime?,
            dailyAttempts: null == dailyAttempts
                ? _value.dailyAttempts
                : dailyAttempts // ignore: cast_nullable_to_non_nullable
                      as Map<DateTime, int>,
          )
          as $Val,
    );
  }
}

/// @nodoc
abstract class _$$PracticeStatsImplCopyWith<$Res>
    implements $PracticeStatsCopyWith<$Res> {
  factory _$$PracticeStatsImplCopyWith(
    _$PracticeStatsImpl value,
    $Res Function(_$PracticeStatsImpl) then,
  ) = __$$PracticeStatsImplCopyWithImpl<$Res>;
  @override
  @useResult
  $Res call({
    int totalAttempts,
    int sequencesPracticed,
    double? averageScore,
    int currentStreak,
    int longestStreak,
    DateTime? lastPracticeDate,
    Map<DateTime, int> dailyAttempts,
  });
}

/// @nodoc
class __$$PracticeStatsImplCopyWithImpl<$Res>
    extends _$PracticeStatsCopyWithImpl<$Res, _$PracticeStatsImpl>
    implements _$$PracticeStatsImplCopyWith<$Res> {
  __$$PracticeStatsImplCopyWithImpl(
    _$PracticeStatsImpl _value,
    $Res Function(_$PracticeStatsImpl) _then,
  ) : super(_value, _then);

  /// Create a copy of PracticeStats
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  @override
  $Res call({
    Object? totalAttempts = null,
    Object? sequencesPracticed = null,
    Object? averageScore = freezed,
    Object? currentStreak = null,
    Object? longestStreak = null,
    Object? lastPracticeDate = freezed,
    Object? dailyAttempts = null,
  }) {
    return _then(
      _$PracticeStatsImpl(
        totalAttempts: null == totalAttempts
            ? _value.totalAttempts
            : totalAttempts // ignore: cast_nullable_to_non_nullable
                  as int,
        sequencesPracticed: null == sequencesPracticed
            ? _value.sequencesPracticed
            : sequencesPracticed // ignore: cast_nullable_to_non_nullable
                  as int,
        averageScore: freezed == averageScore
            ? _value.averageScore
            : averageScore // ignore: cast_nullable_to_non_nullable
                  as double?,
        currentStreak: null == currentStreak
            ? _value.currentStreak
            : currentStreak // ignore: cast_nullable_to_non_nullable
                  as int,
        longestStreak: null == longestStreak
            ? _value.longestStreak
            : longestStreak // ignore: cast_nullable_to_non_nullable
                  as int,
        lastPracticeDate: freezed == lastPracticeDate
            ? _value.lastPracticeDate
            : lastPracticeDate // ignore: cast_nullable_to_non_nullable
                  as DateTime?,
        dailyAttempts: null == dailyAttempts
            ? _value._dailyAttempts
            : dailyAttempts // ignore: cast_nullable_to_non_nullable
                  as Map<DateTime, int>,
      ),
    );
  }
}

/// @nodoc

class _$PracticeStatsImpl implements _PracticeStats {
  const _$PracticeStatsImpl({
    this.totalAttempts = 0,
    this.sequencesPracticed = 0,
    this.averageScore,
    this.currentStreak = 0,
    this.longestStreak = 0,
    this.lastPracticeDate,
    final Map<DateTime, int> dailyAttempts = const {},
  }) : _dailyAttempts = dailyAttempts;

  @override
  @JsonKey()
  final int totalAttempts;
  @override
  @JsonKey()
  final int sequencesPracticed;
  @override
  final double? averageScore;
  @override
  @JsonKey()
  final int currentStreak;
  @override
  @JsonKey()
  final int longestStreak;
  @override
  final DateTime? lastPracticeDate;
  final Map<DateTime, int> _dailyAttempts;
  @override
  @JsonKey()
  Map<DateTime, int> get dailyAttempts {
    if (_dailyAttempts is EqualUnmodifiableMapView) return _dailyAttempts;
    // ignore: implicit_dynamic_type
    return EqualUnmodifiableMapView(_dailyAttempts);
  }

  @override
  String toString() {
    return 'PracticeStats(totalAttempts: $totalAttempts, sequencesPracticed: $sequencesPracticed, averageScore: $averageScore, currentStreak: $currentStreak, longestStreak: $longestStreak, lastPracticeDate: $lastPracticeDate, dailyAttempts: $dailyAttempts)';
  }

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is _$PracticeStatsImpl &&
            (identical(other.totalAttempts, totalAttempts) ||
                other.totalAttempts == totalAttempts) &&
            (identical(other.sequencesPracticed, sequencesPracticed) ||
                other.sequencesPracticed == sequencesPracticed) &&
            (identical(other.averageScore, averageScore) ||
                other.averageScore == averageScore) &&
            (identical(other.currentStreak, currentStreak) ||
                other.currentStreak == currentStreak) &&
            (identical(other.longestStreak, longestStreak) ||
                other.longestStreak == longestStreak) &&
            (identical(other.lastPracticeDate, lastPracticeDate) ||
                other.lastPracticeDate == lastPracticeDate) &&
            const DeepCollectionEquality().equals(
              other._dailyAttempts,
              _dailyAttempts,
            ));
  }

  @override
  int get hashCode => Object.hash(
    runtimeType,
    totalAttempts,
    sequencesPracticed,
    averageScore,
    currentStreak,
    longestStreak,
    lastPracticeDate,
    const DeepCollectionEquality().hash(_dailyAttempts),
  );

  /// Create a copy of PracticeStats
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @override
  @pragma('vm:prefer-inline')
  _$$PracticeStatsImplCopyWith<_$PracticeStatsImpl> get copyWith =>
      __$$PracticeStatsImplCopyWithImpl<_$PracticeStatsImpl>(this, _$identity);
}

abstract class _PracticeStats implements PracticeStats {
  const factory _PracticeStats({
    final int totalAttempts,
    final int sequencesPracticed,
    final double? averageScore,
    final int currentStreak,
    final int longestStreak,
    final DateTime? lastPracticeDate,
    final Map<DateTime, int> dailyAttempts,
  }) = _$PracticeStatsImpl;

  @override
  int get totalAttempts;
  @override
  int get sequencesPracticed;
  @override
  double? get averageScore;
  @override
  int get currentStreak;
  @override
  int get longestStreak;
  @override
  DateTime? get lastPracticeDate;
  @override
  Map<DateTime, int> get dailyAttempts;

  /// Create a copy of PracticeStats
  /// with the given fields replaced by the non-null parameter values.
  @override
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$PracticeStatsImplCopyWith<_$PracticeStatsImpl> get copyWith =>
      throw _privateConstructorUsedError;
}
