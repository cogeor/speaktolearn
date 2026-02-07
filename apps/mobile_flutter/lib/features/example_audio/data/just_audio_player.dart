import 'dart:async';

import 'package:just_audio/just_audio.dart' as ja;

import '../../../core/audio/audio_player.dart';
import '../../../core/audio/audio_source.dart';

/// AudioPlayer implementation using just_audio package.
class JustAudioPlayer implements AudioPlayer {
  JustAudioPlayer() : _player = ja.AudioPlayer() {
    _listenToPlayerState();
  }

  final ja.AudioPlayer _player;
  final _stateController = StreamController<PlaybackState>.broadcast();
  PlaybackState _currentState = PlaybackState.idle;

  void _listenToPlayerState() {
    _player.playerStateStream.listen((playerState) {
      final PlaybackState newState;
      if (playerState.processingState == ja.ProcessingState.loading ||
          playerState.processingState == ja.ProcessingState.buffering) {
        newState = PlaybackState.loading;
      } else if (playerState.processingState == ja.ProcessingState.completed) {
        newState = PlaybackState.completed;
      } else if (playerState.playing) {
        newState = PlaybackState.playing;
      } else {
        newState = PlaybackState.paused;
      }
      _updateState(newState);
    });
  }

  @override
  Stream<PlaybackState> get stateStream => _stateController.stream;

  @override
  Stream<Duration> get positionStream => _player.positionStream;

  @override
  PlaybackState get currentState => _currentState;

  @override
  Duration? get duration => _player.duration;

  void _updateState(PlaybackState state) {
    if (_currentState != state) {
      _currentState = state;
      _stateController.add(state);
    }
  }

  @override
  Future<void> load(AudioSource source) async {
    try {
      final jaSource = switch (source) {
        AssetAudioSource(:final assetPath) => ja.AudioSource.asset(assetPath),
        FileAudioSource(:final filePath) => ja.AudioSource.file(filePath),
        UrlAudioSource(:final url) => ja.AudioSource.uri(Uri.parse(url)),
      };
      await _player.setAudioSource(jaSource);
    } catch (e) {
      _updateState(PlaybackState.error);
    }
  }

  @override
  Future<void> play() async {
    await _player.play();
  }

  @override
  Future<void> pause() async {
    await _player.pause();
    _updateState(PlaybackState.paused);
  }

  @override
  Future<void> stop() async {
    await _player.stop();
    await _player.seek(Duration.zero);
    _updateState(PlaybackState.idle);
  }

  @override
  Future<void> seek(Duration position) async {
    await _player.seek(position);
  }

  @override
  Future<void> dispose() async {
    await _stateController.close();
    await _player.dispose();
  }
}
