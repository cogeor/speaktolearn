import 'package:hive/hive.dart';

import '../domain/progress_repository.dart';
import '../domain/score_attempt.dart';
import '../domain/text_sequence_progress.dart';

/// Hive-based implementation of [ProgressRepository].
class ProgressRepositoryImpl implements ProgressRepository {
  ProgressRepositoryImpl({
    required Box<dynamic> progressBox,
    required Box<dynamic> attemptsBox,
  })  : _progressBox = progressBox,
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
    return TextSequenceProgress.fromJson(Map<String, dynamic>.from(data as Map));
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
  Future<void> saveAttempt(ScoreAttempt attempt) async {
    // Get existing attempts for this sequence
    final key = attempt.textSequenceId;
    final existing = await getAttempts(key);

    // Add new attempt and limit to max
    final updated = [attempt, ...existing];
    if (updated.length > _maxAttempts) {
      updated.removeRange(_maxAttempts, updated.length);
    }

    // Save attempts as list of JSON maps
    await _attemptsBox.put(
      key,
      updated.map((a) => a.toJson()).toList(),
    );

    // Update progress with new best score if better
    final currentProgress = await getProgress(key);
    final newBest = attempt.score > (currentProgress.bestScore ?? -1);

    final updatedProgress = currentProgress.copyWith(
      lastAttemptAt: attempt.gradedAt,
      attemptCount: currentProgress.attemptCount + 1,
      bestScore: newBest ? attempt.score : currentProgress.bestScore,
      bestAttemptId: newBest ? attempt.id : currentProgress.bestAttemptId,
    );

    await _progressBox.put(key, updatedProgress.toJson());
  }

  @override
  Future<List<ScoreAttempt>> getAttempts(
    String textSequenceId, {
    int? limit,
  }) async {
    final data = _attemptsBox.get(textSequenceId);
    if (data == null) return [];

    final list = (data as List)
        .map((item) => ScoreAttempt.fromJson(
              Map<String, dynamic>.from(item as Map),
            ))
        .toList();

    if (limit != null && limit < list.length) {
      return list.sublist(0, limit);
    }
    return list;
  }
}
