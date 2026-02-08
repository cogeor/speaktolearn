import 'dart:async';

import 'package:speak_to_learn/core/audio/audio_source.dart';
import 'package:speak_to_learn/features/example_audio/domain/example_audio_repository.dart';
import 'package:speak_to_learn/features/progress/domain/progress_repository.dart';
import 'package:speak_to_learn/features/progress/domain/score_attempt.dart';
import 'package:speak_to_learn/features/progress/domain/text_sequence_progress.dart';
import 'package:speak_to_learn/features/recording/domain/recording.dart';
import 'package:speak_to_learn/features/recording/domain/recording_repository.dart';
import 'package:speak_to_learn/features/settings/domain/app_settings.dart';
import 'package:speak_to_learn/features/settings/domain/settings_repository.dart';
import 'package:speak_to_learn/features/text_sequences/domain/text_sequence.dart';
import 'package:speak_to_learn/features/text_sequences/domain/text_sequence_repository.dart';

/// Mock implementation of [TextSequenceRepository] for testing.
class MockTextSequenceRepository implements TextSequenceRepository {
  final List<TextSequence> _sequences;

  MockTextSequenceRepository([List<TextSequence>? sequences])
    : _sequences = sequences ?? [];

  @override
  Future<List<TextSequence>> getAll() async => _sequences;

  @override
  Future<TextSequence?> getById(String id) async {
    try {
      return _sequences.firstWhere((s) => s.id == id);
    } catch (_) {
      return null;
    }
  }

  @override
  Future<List<TextSequence>> getByTag(String tag) async {
    return _sequences.where((s) => s.tags?.contains(tag) ?? false).toList();
  }

  @override
  Future<List<TextSequence>> getByDifficulty(int difficulty) async {
    return _sequences.where((s) => s.difficulty == difficulty).toList();
  }

  @override
  Future<List<TextSequence>> getByLevel(int level) async {
    return _sequences.where((s) => s.hskLevel == level).toList();
  }

  @override
  Future<int> count() async => _sequences.length;
}

/// Mock implementation of [ProgressRepository] for testing.
class MockProgressRepository implements ProgressRepository {
  final Map<String, TextSequenceProgress> _progress = {};
  final Map<String, List<ScoreAttempt>> _attempts = {};

  @override
  Future<TextSequenceProgress> getProgress(String textSequenceId) async {
    return _progress[textSequenceId] ?? TextSequenceProgress.initial();
  }

  @override
  Future<Map<String, TextSequenceProgress>> getProgressMap(
    List<String> ids,
  ) async {
    final map = <String, TextSequenceProgress>{};
    for (final id in ids) {
      map[id] = _progress[id] ?? TextSequenceProgress.initial();
    }
    return map;
  }

  @override
  Future<Set<String>> getTrackedIds() async {
    return _progress.entries
        .where((e) => e.value.tracked)
        .map((e) => e.key)
        .toSet();
  }

  @override
  Future<List<TextSequenceProgress>> getTrackedProgress() async {
    return _progress.values.where((p) => p.tracked).toList();
  }

  @override
  Future<void> toggleTracked(String textSequenceId) async {
    final current = _progress[textSequenceId] ?? TextSequenceProgress.initial();
    _progress[textSequenceId] = current.copyWith(tracked: !current.tracked);
  }

  @override
  Future<void> setTracked(String textSequenceId, bool tracked) async {
    final current = _progress[textSequenceId] ?? TextSequenceProgress.initial();
    _progress[textSequenceId] = current.copyWith(tracked: tracked);
  }

  @override
  Future<void> saveAttempt(ScoreAttempt attempt) async {
    final attempts = _attempts[attempt.textSequenceId] ?? [];
    attempts.add(attempt);
    _attempts[attempt.textSequenceId] = attempts;

    final current =
        _progress[attempt.textSequenceId] ?? TextSequenceProgress.initial();
    final isBest =
        current.bestScore == null || attempt.score > current.bestScore!;
    _progress[attempt.textSequenceId] = current.copyWith(
      attemptCount: current.attemptCount + 1,
      lastAttemptAt: attempt.gradedAt,
      bestScore: isBest ? attempt.score : current.bestScore,
      bestAttemptId: isBest ? attempt.id : current.bestAttemptId,
      lastScore: attempt.score,
    );
  }

  @override
  Future<List<ScoreAttempt>> getAttempts(
    String textSequenceId, {
    int? limit,
  }) async {
    final attempts = _attempts[textSequenceId] ?? [];
    if (limit != null && attempts.length > limit) {
      return attempts.sublist(attempts.length - limit);
    }
    return attempts;
  }

  @override
  Future<List<ScoreAttempt>> getAllAttempts() async {
    final result = <ScoreAttempt>[];
    for (final attempts in _attempts.values) {
      result.addAll(attempts);
    }
    return result;
  }

  @override
  Future<void> generateFakeStats({
    required List<String> sequenceIds,
    int days = 60,
    int attemptsPerDay = 10,
  }) async {
    // No-op for testing
  }

  @override
  Future<void> clearAllStats() async {
    _progress.clear();
    _attempts.clear();
  }
}

/// Mock implementation of [RecordingRepository] for testing.
class MockRecordingRepository implements RecordingRepository {
  final Map<String, Recording> _recordings = {};

  @override
  Future<void> saveLatest(Recording recording) async {
    _recordings[recording.textSequenceId] = recording;
  }

  @override
  Future<Recording?> getLatest(String textSequenceId) async {
    return _recordings[textSequenceId];
  }

  @override
  Future<void> deleteLatest(String textSequenceId) async {
    _recordings.remove(textSequenceId);
  }

  @override
  Future<bool> hasRecording(String textSequenceId) async {
    return _recordings.containsKey(textSequenceId);
  }
}

/// Mock implementation of [SettingsRepository] for testing.
class MockSettingsRepository implements SettingsRepository {
  AppSettings _settings = AppSettings.defaults;
  final _controller = StreamController<AppSettings>.broadcast();

  @override
  Future<AppSettings> getSettings() async => _settings;

  @override
  Future<void> updateSettings(AppSettings settings) async {
    _settings = settings;
    _controller.add(_settings);
  }

  @override
  Future<void> resetToDefaults() async {
    _settings = AppSettings.defaults;
    _controller.add(_settings);
  }

  @override
  Stream<AppSettings> watchSettings() => _controller.stream;

  void dispose() {
    _controller.close();
  }
}

/// Mock implementation of [ExampleAudioRepository] for testing.
class MockExampleAudioRepository implements ExampleAudioRepository {
  final Map<String, AudioSource> _sources = {};

  void addSource(String uri, AudioSource source) {
    _sources[uri] = source;
  }

  @override
  Future<AudioSource> resolve(String uri) async {
    final source = _sources[uri];
    if (source != null) return source;

    // Return a file source for testing by default
    return FileAudioSource('/test/audio/$uri.m4a');
  }

  @override
  Future<AudioSource?> resolveVoice(
    TextSequence sequence,
    String voiceId,
  ) async {
    final voices = sequence.voices;
    if (voices == null || voices.isEmpty) return null;

    try {
      final voice = voices.firstWhere((v) => v.id == voiceId);
      return resolve(voice.uri);
    } catch (_) {
      return null;
    }
  }

  @override
  Future<void> prefetch(TextSequence sequence) async {
    // No-op for testing
  }

  @override
  Future<bool> isAvailableLocally(String uri) async => true;
}
