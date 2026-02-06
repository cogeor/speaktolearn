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
  Future<Set<String>> getTrackedIds() {
    throw UnimplementedError();
  }

  @override
  Future<List<TextSequenceProgress>> getTrackedProgress() {
    throw UnimplementedError();
  }

  @override
  Future<void> toggleTracked(String textSequenceId) {
    throw UnimplementedError();
  }

  @override
  Future<void> setTracked(String textSequenceId, bool tracked) {
    throw UnimplementedError();
  }

  @override
  Future<void> saveAttempt(ScoreAttempt attempt) {
    throw UnimplementedError();
  }

  @override
  Future<List<ScoreAttempt>> getAttempts(
    String textSequenceId, {
    int? limit,
  }) {
    throw UnimplementedError();
  }
}
