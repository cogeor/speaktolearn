import 'dart:async';

import 'package:speak_to_learn/core/audio/audio_player.dart';
import 'package:speak_to_learn/core/audio/audio_source.dart';
import 'package:speak_to_learn/core/result.dart';
import 'package:speak_to_learn/features/recording/domain/audio_recorder.dart';

/// Fake implementation of [AudioRecorder] for testing.
class FakeAudioRecorder implements AudioRecorder {
  bool _isRecording = false;
  String? _recordingPath;
  int _recordingCounter = 0;

  /// Controls whether the next start() call should fail.
  RecordingError? nextStartError;

  /// Controls whether the next stop() call should fail.
  RecordingError? nextStopError;

  @override
  bool get isRecording => _isRecording;

  @override
  Future<Result<String, RecordingError>> start() async {
    if (nextStartError != null) {
      final error = nextStartError!;
      nextStartError = null;
      return Failure(error);
    }

    if (_isRecording) {
      return const Failure(RecordingError.alreadyRecording);
    }

    _recordingCounter++;
    _recordingPath = '/test/recordings/recording_$_recordingCounter.m4a';
    _isRecording = true;
    return Success(_recordingPath!);
  }

  @override
  Future<Result<String, RecordingError>> stop() async {
    if (nextStopError != null) {
      final error = nextStopError!;
      nextStopError = null;
      return Failure(error);
    }

    if (!_isRecording) {
      return const Failure(RecordingError.notRecording);
    }

    final path = _recordingPath!;
    _isRecording = false;
    _recordingPath = null;
    return Success(path);
  }

  @override
  Future<void> cancel() async {
    _isRecording = false;
    _recordingPath = null;
  }

  @override
  Future<void> dispose() async {
    await cancel();
  }
}

/// Fake implementation of [AudioPlayer] for testing.
class FakeAudioPlayer implements AudioPlayer {
  final _stateController = StreamController<PlaybackState>.broadcast();
  final _positionController = StreamController<Duration>.broadcast();

  PlaybackState _state = PlaybackState.idle;
  Duration _currentPosition = Duration.zero;
  Duration? _duration;
  AudioSource? _loadedSource;

  /// The current playback position, for test assertions.
  Duration get position => _currentPosition;

  /// Controls whether the next load() call should fail.
  bool nextLoadFails = false;

  @override
  Stream<PlaybackState> get stateStream => _stateController.stream;

  @override
  Stream<Duration> get positionStream => _positionController.stream;

  @override
  PlaybackState get currentState => _state;

  @override
  Duration? get duration => _duration;

  /// The currently loaded audio source, for test assertions.
  AudioSource? get loadedSource => _loadedSource;

  void _setState(PlaybackState state) {
    _state = state;
    _stateController.add(state);
  }

  void _setPosition(Duration position) {
    _currentPosition = position;
    _positionController.add(position);
  }

  @override
  Future<void> load(AudioSource source) async {
    _setState(PlaybackState.loading);
    _loadedSource = source;

    if (nextLoadFails) {
      nextLoadFails = false;
      _setState(PlaybackState.error);
      return;
    }

    _duration = const Duration(seconds: 3);
    _setPosition(Duration.zero);
    _setState(PlaybackState.paused);
  }

  @override
  Future<void> play() async {
    if (_loadedSource == null) return;
    _setState(PlaybackState.playing);
  }

  @override
  Future<void> pause() async {
    _setState(PlaybackState.paused);
  }

  @override
  Future<void> stop() async {
    _setPosition(Duration.zero);
    _setState(PlaybackState.idle);
  }

  @override
  Future<void> seek(Duration position) async {
    if (_duration != null && position > _duration!) {
      _setPosition(_duration!);
    } else if (position < Duration.zero) {
      _setPosition(Duration.zero);
    } else {
      _setPosition(position);
    }
  }

  @override
  Future<void> dispose() async {
    await _stateController.close();
    await _positionController.close();
  }

  /// Simulates playback completing for testing.
  void simulateComplete() {
    if (_duration != null) {
      _setPosition(_duration!);
    }
    _setState(PlaybackState.completed);
  }
}

/// Fake implementation of a speech recognizer for testing.
class FakeSpeechRecognizer {
  /// The text to return from recognition.
  String recognizedText = '';

  /// Whether recognition should fail.
  bool shouldFail = false;

  /// Simulates speech recognition on audio.
  Future<String?> recognize(String audioPath) async {
    if (shouldFail) return null;
    return recognizedText;
  }
}
