import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:uuid/uuid.dart';

import '../../../core/audio/audio_player.dart';
import '../../../core/audio/audio_source.dart';
import '../../progress/domain/progress_repository.dart';
import '../../progress/domain/score_attempt.dart';
import '../../scoring/domain/grade.dart';
import '../../scoring/domain/pronunciation_scorer.dart';
import '../../text_sequences/domain/text_sequence.dart';
import '../domain/audio_recorder.dart';
import '../domain/recording.dart';
import '../domain/recording_repository.dart';
import 'recording_state.dart';

/// Controller for managing audio recording.
///
/// Handles starting and stopping recordings, managing state,
/// and coordinating with the repository for persistence.
class RecordingController extends StateNotifier<RecordingState> {
  RecordingController({
    required AudioRecorder recorder,
    required RecordingRepository repository,
    required PronunciationScorer scorer,
    required ProgressRepository progressRepository,
    required AudioPlayer audioPlayer,
  })  : _recorder = recorder,
        _repository = repository,
        _scorer = scorer,
        _progressRepository = progressRepository,
        _audioPlayer = audioPlayer,
        super(const RecordingState());

  final AudioRecorder _recorder;
  final RecordingRepository _repository;
  final PronunciationScorer _scorer;
  final ProgressRepository _progressRepository;
  final AudioPlayer _audioPlayer;

  /// Starts recording audio for the given text sequence.
  ///
  /// Sets [RecordingState.isRecording] to true on success.
  /// On failure, sets [RecordingState.error] with the error message.
  Future<void> startRecording(String textSequenceId) async {
    state = state.copyWith(isRecording: true, error: null);

    final result = await _recorder.start();

    result.when(
      success: (_) {
        // Recording started successfully, state already set
      },
      failure: (error) {
        state = state.copyWith(
          isRecording: false,
          error: _errorToMessage(error),
        );
      },
    );
  }

  /// Stops the current recording.
  Future<void> stopRecording() async {
    final result = await _recorder.stop();

    result.when(
      success: (_) {
        state = state.copyWith(isRecording: false);
      },
      failure: (error) {
        state = state.copyWith(
          isRecording: false,
          error: _errorToMessage(error),
        );
      },
    );
  }

  /// Stops recording and scores the pronunciation against the given text sequence.
  ///
  /// Sets [RecordingState.isScoring] to true while scoring is in progress.
  /// On success, saves the attempt and returns the [Grade].
  /// On failure, sets [RecordingState.error] and returns null.
  Future<Grade?> stopAndScore(TextSequence textSequence) async {
    state = state.copyWith(isScoring: true, error: null);

    final stopResult = await _recorder.stop();

    return stopResult.when(
      success: (filePath) async {
        try {
          final recording = Recording(
            id: const Uuid().v4(),
            textSequenceId: textSequence.id,
            createdAt: DateTime.now(),
            filePath: filePath,
          );

          final grade = await _scorer.score(textSequence, recording);

          final attempt = ScoreAttempt(
            id: const Uuid().v4(),
            textSequenceId: textSequence.id,
            gradedAt: DateTime.now(),
            score: grade.overall,
            method: grade.method,
            recognizedText: grade.recognizedText,
            details: grade.details,
          );

          await _progressRepository.saveAttempt(attempt);

          state = state.copyWith(
            isRecording: false,
            isScoring: false,
            hasLatestRecording: true,
          );

          return grade;
        } catch (e) {
          state = state.copyWith(
            isRecording: false,
            isScoring: false,
            error: 'Scoring failed: ${e.toString()}',
          );
          return null;
        }
      },
      failure: (error) {
        state = state.copyWith(
          isRecording: false,
          isScoring: false,
          error: _errorToMessage(error),
        );
        return null;
      },
    );
  }

  /// Checks if a latest recording exists for the given text sequence.
  ///
  /// Updates [RecordingState.hasLatestRecording] based on the result.
  Future<void> checkLatestRecording(String textSequenceId) async {
    final recording = await _repository.getLatest(textSequenceId);
    state = state.copyWith(hasLatestRecording: recording != null);
  }

  /// Replays the latest recording for the given text sequence.
  ///
  /// Sets [RecordingState.isPlaying] to true while playing.
  /// Does nothing if no recording exists for the sequence.
  Future<void> replayLatest(String textSequenceId) async {
    final recording = await _repository.getLatest(textSequenceId);
    if (recording == null) return;

    state = state.copyWith(isPlaying: true);
    await _audioPlayer.load(FileAudioSource(recording.filePath));
    await _audioPlayer.play();

    // Wait for playback to complete
    await _audioPlayer.stateStream.firstWhere(
      (playbackState) =>
          playbackState == PlaybackState.completed ||
          playbackState == PlaybackState.idle ||
          playbackState == PlaybackState.error,
    );

    state = state.copyWith(isPlaying: false);
  }

  /// Cancels any active recording or playback.
  ///
  /// Stops the recorder if recording is in progress.
  /// Stops the audio player if playback is in progress.
  /// Resets state to initial values.
  Future<void> cancel() async {
    if (state.isRecording) {
      await _recorder.stop();
    }
    if (state.isPlaying) {
      await _audioPlayer.stop();
    }
    state = const RecordingState();
  }

  String _errorToMessage(RecordingError error) {
    return switch (error) {
      RecordingError.permissionDenied => 'Microphone permission denied',
      RecordingError.noMicrophone => 'No microphone available',
      RecordingError.notSupported => 'Recording not supported on this device',
      RecordingError.recordingFailed => 'Recording failed',
      RecordingError.alreadyRecording => 'Already recording',
      RecordingError.notRecording => 'No recording in progress',
    };
  }
}
