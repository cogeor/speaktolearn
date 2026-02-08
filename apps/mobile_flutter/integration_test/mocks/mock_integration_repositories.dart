// integration_test/mocks/mock_integration_repositories.dart
import 'dart:async';

import 'package:speak_to_learn/core/audio/audio_player.dart';
import 'package:speak_to_learn/core/audio/audio_source.dart';
import 'package:speak_to_learn/core/result.dart';
import 'package:speak_to_learn/features/example_audio/domain/example_audio_repository.dart';
import 'package:speak_to_learn/features/progress/domain/progress_repository.dart';
import 'package:speak_to_learn/features/progress/domain/score_attempt.dart';
import 'package:speak_to_learn/features/progress/domain/text_sequence_progress.dart';
import 'package:speak_to_learn/features/recording/domain/audio_recorder.dart';
import 'package:speak_to_learn/features/recording/domain/recording.dart';
import 'package:speak_to_learn/features/recording/domain/recording_repository.dart';
import 'package:speak_to_learn/features/settings/domain/app_settings.dart';
import 'package:speak_to_learn/features/settings/domain/settings_repository.dart';
import 'package:speak_to_learn/features/text_sequences/domain/text_sequence.dart';
import 'package:speak_to_learn/features/text_sequences/domain/text_sequence_repository.dart';

/// Mock TextSequenceRepository for integration tests.
class MockIntegrationTextSequenceRepository implements TextSequenceRepository {
  MockIntegrationTextSequenceRepository(this._sequences);

  final List<TextSequence> _sequences;

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
  Future<int> count() async => _sequences.length;
}

/// Mock ProgressRepository for integration tests.
class MockIntegrationProgressRepository implements ProgressRepository {
  MockIntegrationProgressRepository({Set<String>? trackedIds})
    : _progress = {
        for (final id in trackedIds ?? <String>{})
          id: const TextSequenceProgress(tracked: true),
      };

  final Map<String, TextSequenceProgress> _progress;
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
}

/// Mock RecordingRepository for integration tests.
class MockIntegrationRecordingRepository implements RecordingRepository {
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

/// Mock ExampleAudioRepository for integration tests.
class MockIntegrationExampleAudioRepository implements ExampleAudioRepository {
  @override
  Future<AudioSource> resolve(String uri) async {
    return FileAudioSource('/mock/audio.m4a');
  }

  @override
  Future<AudioSource?> resolveVoice(
    TextSequence sequence,
    String voiceId,
  ) async {
    return FileAudioSource('/mock/audio.m4a');
  }

  @override
  Future<void> prefetch(TextSequence sequence) async {}

  @override
  Future<bool> isAvailableLocally(String uri) async => true;
}

/// Mock SettingsRepository for integration tests.
class MockIntegrationSettingsRepository implements SettingsRepository {
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
}

/// Mock AudioRecorder for integration tests.
class MockIntegrationAudioRecorder implements AudioRecorder {
  bool _isRecording = false;
  int _recordingCounter = 0;

  @override
  bool get isRecording => _isRecording;

  @override
  Future<Result<String, RecordingError>> start() async {
    _isRecording = true;
    _recordingCounter++;
    return Success('/mock/recordings/recording_$_recordingCounter.m4a');
  }

  @override
  Future<Result<String, RecordingError>> stop() async {
    final path = '/mock/recordings/recording_$_recordingCounter.m4a';
    _isRecording = false;
    return Success(path);
  }

  @override
  Future<void> cancel() async {
    _isRecording = false;
  }

  @override
  Future<void> dispose() async {
    await cancel();
  }
}

/// Mock AudioPlayer for integration tests.
class MockIntegrationAudioPlayer implements AudioPlayer {
  final _stateController = StreamController<PlaybackState>.broadcast();
  final _positionController = StreamController<Duration>.broadcast();
  PlaybackState _state = PlaybackState.idle;

  @override
  Stream<PlaybackState> get stateStream => _stateController.stream;

  @override
  Stream<Duration> get positionStream => _positionController.stream;

  @override
  PlaybackState get currentState => _state;

  @override
  Duration? get duration => const Duration(seconds: 2);

  @override
  Future<void> load(AudioSource source) async {
    _state = PlaybackState.paused;
    _stateController.add(_state);
  }

  @override
  Future<void> play() async {
    _state = PlaybackState.playing;
    _stateController.add(_state);
    // Simulate playback completion after a short delay
    Future.delayed(const Duration(milliseconds: 100), () {
      _state = PlaybackState.completed;
      _stateController.add(_state);
    });
  }

  @override
  Future<void> pause() async {
    _state = PlaybackState.paused;
    _stateController.add(_state);
  }

  @override
  Future<void> stop() async {
    _state = PlaybackState.idle;
    _stateController.add(_state);
  }

  @override
  Future<void> seek(Duration position) async {}

  @override
  Future<void> dispose() async {
    await _stateController.close();
    await _positionController.close();
  }
}
