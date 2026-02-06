import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:freezed_annotation/freezed_annotation.dart';

import '../../../core/audio/audio_player.dart';
import '../domain/example_audio_repository.dart';
import '../../text_sequences/domain/text_sequence.dart';

part 'example_audio_controller.freezed.dart';

/// State for the example audio playback controller.
@freezed
class ExampleAudioState with _$ExampleAudioState {
  const factory ExampleAudioState({
    @Default(false) bool isPlaying,
    String? currentSequenceId,
    String? currentVoiceId,
  }) = _ExampleAudioState;
}

/// Controller for managing example audio playback.
///
/// Handles playing voice examples for text sequences, including
/// resolving audio sources and managing playback state.
class ExampleAudioController extends StateNotifier<ExampleAudioState> {
  ExampleAudioController({
    required AudioPlayer player,
    required ExampleAudioRepository repository,
  })  : _player = player,
        _repository = repository,
        super(const ExampleAudioState()) {
    _listenToPlaybackState();
  }

  final AudioPlayer _player;
  final ExampleAudioRepository _repository;

  void _listenToPlaybackState() {
    _player.stateStream.listen((playbackState) {
      final isPlaying = playbackState == PlaybackState.playing;
      if (state.isPlaying != isPlaying) {
        state = state.copyWith(isPlaying: isPlaying);
      }
      // Clear current ids when playback completes or stops
      if (playbackState == PlaybackState.completed ||
          playbackState == PlaybackState.idle) {
        state = state.copyWith(
          currentSequenceId: null,
          currentVoiceId: null,
        );
      }
    });
  }

  /// Plays the example audio for a specific voice in a sequence.
  ///
  /// Resolves the audio source from the repository and starts playback.
  /// Updates state with the current sequence and voice IDs.
  Future<void> play(String sequenceId, String voiceId) async {
    // Create a temporary TextSequence to resolve the voice
    // In practice, the sequence should be passed or looked up
    final sequence = TextSequence(
      id: sequenceId,
      text: '',
      language: '',
    );

    final audioSource = await _repository.resolveVoice(sequence, voiceId);
    if (audioSource == null) {
      return;
    }

    state = state.copyWith(
      currentSequenceId: sequenceId,
      currentVoiceId: voiceId,
    );

    await _player.load(audioSource);
    await _player.play();
  }

  /// Stops the current audio playback.
  Future<void> stop() async {
    await _player.stop();
    state = state.copyWith(
      isPlaying: false,
      currentSequenceId: null,
      currentVoiceId: null,
    );
  }

  /// Prefetches all example audio for a sequence.
  ///
  /// This ensures audio files are available locally for quick playback.
  Future<void> prefetchSequence(TextSequence sequence) async {
    await _repository.prefetch(sequence);
  }
}
