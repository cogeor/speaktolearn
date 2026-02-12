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
  int get hardCount => throw _privateConstructorUsedError;
  int get almostCount => throw _privateConstructorUsedError;
  int get goodCount => throw _privateConstructorUsedError;
  int get easyCount => throw _privateConstructorUsedError;
  int get currentStreak => throw _privateConstructorUsedError;
  int get longestStreak => throw _privateConstructorUsedError;
  DateTime? get lastPracticeDate => throw _privateConstructorUsedError;
  Map<DateTime, int> get dailyAttempts => throw _privateConstructorUsedError;

  /// Cumulative mastered sentences over time by HSK level.
  /// Each data point contains counts per HSK level and total.
  List<CumulativeDataPoint> get cumulativeProgress =>
      throw _privateConstructorUsedError;

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
    int hardCount,
    int almostCount,
    int goodCount,
    int easyCount,
    int currentStreak,
    int longestStreak,
    DateTime? lastPracticeDate,
    Map<DateTime, int> dailyAttempts,
    List<CumulativeDataPoint> cumulativeProgress,
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
    Object? hardCount = null,
    Object? almostCount = null,
    Object? goodCount = null,
    Object? easyCount = null,
    Object? currentStreak = null,
    Object? longestStreak = null,
    Object? lastPracticeDate = freezed,
    Object? dailyAttempts = null,
    Object? cumulativeProgress = null,
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
            hardCount: null == hardCount
                ? _value.hardCount
                : hardCount // ignore: cast_nullable_to_non_nullable
                      as int,
            almostCount: null == almostCount
                ? _value.almostCount
                : almostCount // ignore: cast_nullable_to_non_nullable
                      as int,
            goodCount: null == goodCount
                ? _value.goodCount
                : goodCount // ignore: cast_nullable_to_non_nullable
                      as int,
            easyCount: null == easyCount
                ? _value.easyCount
                : easyCount // ignore: cast_nullable_to_non_nullable
                      as int,
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
            cumulativeProgress: null == cumulativeProgress
                ? _value.cumulativeProgress
                : cumulativeProgress // ignore: cast_nullable_to_non_nullable
                      as List<CumulativeDataPoint>,
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
    int hardCount,
    int almostCount,
    int goodCount,
    int easyCount,
    int currentStreak,
    int longestStreak,
    DateTime? lastPracticeDate,
    Map<DateTime, int> dailyAttempts,
    List<CumulativeDataPoint> cumulativeProgress,
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
    Object? hardCount = null,
    Object? almostCount = null,
    Object? goodCount = null,
    Object? easyCount = null,
    Object? currentStreak = null,
    Object? longestStreak = null,
    Object? lastPracticeDate = freezed,
    Object? dailyAttempts = null,
    Object? cumulativeProgress = null,
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
        hardCount: null == hardCount
            ? _value.hardCount
            : hardCount // ignore: cast_nullable_to_non_nullable
                  as int,
        almostCount: null == almostCount
            ? _value.almostCount
            : almostCount // ignore: cast_nullable_to_non_nullable
                  as int,
        goodCount: null == goodCount
            ? _value.goodCount
            : goodCount // ignore: cast_nullable_to_non_nullable
                  as int,
        easyCount: null == easyCount
            ? _value.easyCount
            : easyCount // ignore: cast_nullable_to_non_nullable
                  as int,
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
        cumulativeProgress: null == cumulativeProgress
            ? _value._cumulativeProgress
            : cumulativeProgress // ignore: cast_nullable_to_non_nullable
                  as List<CumulativeDataPoint>,
      ),
    );
  }
}

/// @nodoc

class _$PracticeStatsImpl implements _PracticeStats {
  const _$PracticeStatsImpl({
    this.totalAttempts = 0,
    this.sequencesPracticed = 0,
    this.hardCount = 0,
    this.almostCount = 0,
    this.goodCount = 0,
    this.easyCount = 0,
    this.currentStreak = 0,
    this.longestStreak = 0,
    this.lastPracticeDate,
    final Map<DateTime, int> dailyAttempts = const {},
    final List<CumulativeDataPoint> cumulativeProgress = const [],
  }) : _dailyAttempts = dailyAttempts,
       _cumulativeProgress = cumulativeProgress;

  @override
  @JsonKey()
  final int totalAttempts;
  @override
  @JsonKey()
  final int sequencesPracticed;
  @override
  @JsonKey()
  final int hardCount;
  @override
  @JsonKey()
  final int almostCount;
  @override
  @JsonKey()
  final int goodCount;
  @override
  @JsonKey()
  final int easyCount;
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

  /// Cumulative mastered sentences over time by HSK level.
  /// Each data point contains counts per HSK level and total.
  final List<CumulativeDataPoint> _cumulativeProgress;

  /// Cumulative mastered sentences over time by HSK level.
  /// Each data point contains counts per HSK level and total.
  @override
  @JsonKey()
  List<CumulativeDataPoint> get cumulativeProgress {
    if (_cumulativeProgress is EqualUnmodifiableListView)
      return _cumulativeProgress;
    // ignore: implicit_dynamic_type
    return EqualUnmodifiableListView(_cumulativeProgress);
  }

  @override
  String toString() {
    return 'PracticeStats(totalAttempts: $totalAttempts, sequencesPracticed: $sequencesPracticed, hardCount: $hardCount, almostCount: $almostCount, goodCount: $goodCount, easyCount: $easyCount, currentStreak: $currentStreak, longestStreak: $longestStreak, lastPracticeDate: $lastPracticeDate, dailyAttempts: $dailyAttempts, cumulativeProgress: $cumulativeProgress)';
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
            (identical(other.hardCount, hardCount) ||
                other.hardCount == hardCount) &&
            (identical(other.almostCount, almostCount) ||
                other.almostCount == almostCount) &&
            (identical(other.goodCount, goodCount) ||
                other.goodCount == goodCount) &&
            (identical(other.easyCount, easyCount) ||
                other.easyCount == easyCount) &&
            (identical(other.currentStreak, currentStreak) ||
                other.currentStreak == currentStreak) &&
            (identical(other.longestStreak, longestStreak) ||
                other.longestStreak == longestStreak) &&
            (identical(other.lastPracticeDate, lastPracticeDate) ||
                other.lastPracticeDate == lastPracticeDate) &&
            const DeepCollectionEquality().equals(
              other._dailyAttempts,
              _dailyAttempts,
            ) &&
            const DeepCollectionEquality().equals(
              other._cumulativeProgress,
              _cumulativeProgress,
            ));
  }

  @override
  int get hashCode => Object.hash(
    runtimeType,
    totalAttempts,
    sequencesPracticed,
    hardCount,
    almostCount,
    goodCount,
    easyCount,
    currentStreak,
    longestStreak,
    lastPracticeDate,
    const DeepCollectionEquality().hash(_dailyAttempts),
    const DeepCollectionEquality().hash(_cumulativeProgress),
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
    final int hardCount,
    final int almostCount,
    final int goodCount,
    final int easyCount,
    final int currentStreak,
    final int longestStreak,
    final DateTime? lastPracticeDate,
    final Map<DateTime, int> dailyAttempts,
    final List<CumulativeDataPoint> cumulativeProgress,
  }) = _$PracticeStatsImpl;

  @override
  int get totalAttempts;
  @override
  int get sequencesPracticed;
  @override
  int get hardCount;
  @override
  int get almostCount;
  @override
  int get goodCount;
  @override
  int get easyCount;
  @override
  int get currentStreak;
  @override
  int get longestStreak;
  @override
  DateTime? get lastPracticeDate;
  @override
  Map<DateTime, int> get dailyAttempts;

  /// Cumulative mastered sentences over time by HSK level.
  /// Each data point contains counts per HSK level and total.
  @override
  List<CumulativeDataPoint> get cumulativeProgress;

  /// Create a copy of PracticeStats
  /// with the given fields replaced by the non-null parameter values.
  @override
  @JsonKey(includeFromJson: false, includeToJson: false)
  _$$PracticeStatsImplCopyWith<_$PracticeStatsImpl> get copyWith =>
      throw _privateConstructorUsedError;
}
