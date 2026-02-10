import 'dart:math';

import 'package:hive/hive.dart';
import 'package:uuid/uuid.dart';

import '../domain/progress_repository.dart';
import '../domain/rating_attempt.dart';
import '../domain/sentence_rating.dart';
import '../domain/text_sequence_progress.dart';

/// Hive-based implementation of [ProgressRepository].
class ProgressRepositoryImpl implements ProgressRepository {
  ProgressRepositoryImpl({
    required Box<dynamic> progressBox,
    required Box<dynamic> attemptsBox,
  }) : _progressBox = progressBox,
       _attemptsBox = attemptsBox;

  static const _maxAttempts = 50;

  final Box<dynamic> _progressBox;
  final Box<dynamic> _attemptsBox;

  @override
  Future<TextSequenceProgress> getProgress(String textSequenceId) async {
    final data = _progressBox.get(textSequenceId);
    if (data == null) {
      return TextSequenceProgress.initial();
    }
    return TextSequenceProgress.fromJson(
      Map<String, dynamic>.from(data as Map),
    );
  }

  @override
  Future<Map<String, TextSequenceProgress>> getProgressMap(
    List<String> ids,
  ) async {
    final result = <String, TextSequenceProgress>{};
    for (final id in ids) {
      result[id] = await getProgress(id);
    }
    return result;
  }

  @override
  Future<Set<String>> getTrackedIds() async {
    final result = <String>{};
    for (final key in _progressBox.keys) {
      final data = _progressBox.get(key);
      if (data != null) {
        final progress = TextSequenceProgress.fromJson(
          Map<String, dynamic>.from(data as Map),
        );
        if (progress.tracked) {
          result.add(key as String);
        }
      }
    }
    return result;
  }

  @override
  Future<List<TextSequenceProgress>> getTrackedProgress() async {
    final result = <TextSequenceProgress>[];
    for (final key in _progressBox.keys) {
      final data = _progressBox.get(key);
      if (data != null) {
        final progress = TextSequenceProgress.fromJson(
          Map<String, dynamic>.from(data as Map),
        );
        if (progress.tracked) {
          result.add(progress);
        }
      }
    }
    return result;
  }

  @override
  Future<void> toggleTracked(String textSequenceId) async {
    final current = await getProgress(textSequenceId);
    final updated = current.copyWith(tracked: !current.tracked);
    await _progressBox.put(textSequenceId, updated.toJson());
  }

  @override
  Future<void> setTracked(String textSequenceId, bool tracked) async {
    final current = await getProgress(textSequenceId);
    if (current.tracked != tracked) {
      final updated = current.copyWith(tracked: tracked);
      await _progressBox.put(textSequenceId, updated.toJson());
    }
  }

  @override
  Future<void> saveAttempt(RatingAttempt attempt) async {
    // Get existing attempts for this sequence
    final key = attempt.textSequenceId;
    final existing = await getAttempts(key);

    // Add new attempt and limit to max
    final updated = [attempt, ...existing];
    if (updated.length > _maxAttempts) {
      updated.removeRange(_maxAttempts, updated.length);
    }

    // Save attempts as list of JSON maps
    await _attemptsBox.put(key, updated.map((a) => a.toJson()).toList());

    // Update progress with new rating (also mark as tracked)
    final currentProgress = await getProgress(key);
    final updatedProgress = currentProgress.copyWith(
      tracked: true,
      lastAttemptAt: attempt.gradedAt,
      attemptCount: currentProgress.attemptCount + 1,
      lastRating: attempt.rating,
    );

    await _progressBox.put(key, updatedProgress.toJson());
  }

  @override
  Future<List<RatingAttempt>> getAttempts(
    String textSequenceId, {
    int? limit,
  }) async {
    final data = _attemptsBox.get(textSequenceId);
    if (data == null) return [];

    final list = (data as List)
        .map(
          (item) =>
              RatingAttempt.fromJson(Map<String, dynamic>.from(item as Map)),
        )
        .toList();

    if (limit != null && limit < list.length) {
      return list.sublist(0, limit);
    }
    return list;
  }

  @override
  Future<List<RatingAttempt>> getAllAttempts() async {
    final result = <RatingAttempt>[];
    for (final key in _attemptsBox.keys) {
      final data = _attemptsBox.get(key);
      if (data != null) {
        final attempts = (data as List)
            .map(
              (item) =>
                  RatingAttempt.fromJson(Map<String, dynamic>.from(item as Map)),
            )
            .toList();
        result.addAll(attempts);
      }
    }
    return result;
  }

  @override
  Future<void> generateFakeStats({
    required List<String> sequenceIds,
    int days = 60,
    int attemptsPerDay = 10,
  }) async {
    if (sequenceIds.isEmpty) return;

    // Clear existing data to avoid enum decode errors from stale data
    await clearAllStats();

    final random = Random(42); // Fixed seed for reproducibility
    final uuid = const Uuid();
    final now = DateTime.now();

    // Gap days to test streak logic (relative to start)
    const gapDays = {15, 35};

    for (var dayOffset = days; dayOffset > 0; dayOffset--) {
      // Skip gap days to test streak reset
      if (gapDays.contains(days - dayOffset)) continue;

      final dayDate = now.subtract(Duration(days: dayOffset));

      // Variable attempts per day (5-15) to make history more realistic
      final dailyAttempts = 5 + random.nextInt(11);

      for (var i = 0; i < dailyAttempts; i++) {
        // Pick a random sequence
        final sequenceId = sequenceIds[random.nextInt(sequenceIds.length)];

        // Generate random rating
        final rating =
            SentenceRating.values[random.nextInt(SentenceRating.values.length)];

        // Randomize time within the day (8am - 10pm)
        final hour = 8 + random.nextInt(14);
        final minute = random.nextInt(60);
        final gradedAt = DateTime(
          dayDate.year,
          dayDate.month,
          dayDate.day,
          hour,
          minute,
        );

        final attempt = RatingAttempt(
          id: uuid.v4(),
          textSequenceId: sequenceId,
          gradedAt: gradedAt,
          rating: rating,
        );

        await saveAttempt(attempt);
      }
    }
  }

  @override
  Future<void> clearAllStats() async {
    await _progressBox.clear();
    await _attemptsBox.clear();
  }
}
