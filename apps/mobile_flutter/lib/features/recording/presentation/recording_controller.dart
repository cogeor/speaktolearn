import 'dart:async';

import 'package:audio_waveforms/audio_waveforms.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
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
import '../domain/recording_duration_calculator.dart';
import '../domain/recording_repository.dart';
import 'recording_state.dart';

/// Controller for managing audio recording.
///
/// Handles starting and stopping recordings, managing state,
/// and coordinating with the repository for persistence.
///
/// ## Haptic Feedback
///
/// This controller provides tactile feedback for recording operations:
/// - **Medium impact**: Recording started successfully - confirms action initiated
/// - **Light impact**: Recording stopped successfully - subtle confirmation
/// - **Heavy impact**: Error occurred - alerts user to problem
/// - **Medium impact (on scoring)**: Good score (>=80) - positive reinforcement
///
/// Haptics respect device settings via Flutter's HapticFeedback class.
class RecordingController extends StateNotifier<RecordingState> {
  RecordingController({
    required AudioRecorder recorder,
    required RecordingRepository repository,
    required PronunciationScorer scorer,
    required ProgressRepository progressRepository,
    required AudioPlayer audioPlayer,
  }) : _recorder = recorder,
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

  Timer? _autoStopTimer;
  Timer? _countdownTimer;
  int _remainingSeconds = 0;
  TextSequence? _currentTextSequence;

  /// Controller for waveform visualization during recording.
  late final RecorderController waveformController = RecorderController()
    ..androidEncoder = AndroidEncoder.aac
    ..androidOutputFormat = AndroidOutputFormat.mpeg4
    ..iosEncoder = IosEncoder.kAudioFormatMPEG4AAC
    ..sampleRate = 44100;

  /// Starts recording audio for the given text sequence.
  ///
  /// Sets [RecordingState.isRecording] to true on success.
  /// Starts an auto-stop timer based on the expected duration.
  /// On failure, sets [RecordingState.error] with the error message.
  Future<void> startRecording(TextSequence textSequence) async {
    // Calculate expected duration from text
    final duration = calculateRecordingDuration(
      textSequence.text,
      textSequence.language,
    );
    _remainingSeconds = duration.inSeconds;
    _currentTextSequence = textSequence;

    state = state.copyWith(
      isRecording: true,
      error: null,
      remainingSeconds: _remainingSeconds,
      totalDurationSeconds: _remainingSeconds,
    );

    // NOTE: waveformController.record() disabled - conflicts with _recorder
    // causing silent audio. Waveform visualization temporarily unavailable.
    // TODO: Use single recorder for both visualization and file capture
    // await waveformController.record();

    final result = await _recorder.start();

    result.when(
      success: (_) {
        // Haptic feedback on successful start
        HapticFeedback.mediumImpact();

        // Start auto-stop timer
        _autoStopTimer = Timer(duration, () {
          if (state.isRecording && _currentTextSequence != null) {
            stopAndScore(_currentTextSequence!);
          }
        });

        // Start countdown timer for UI updates (every second)
        _countdownTimer = Timer.periodic(const Duration(seconds: 1), (_) {
          if (_remainingSeconds > 0) {
            _remainingSeconds--;
            state = state.copyWith(remainingSeconds: _remainingSeconds);
          }
        });
      },
      failure: (error) {
        // Haptic feedback on failure
        HapticFeedback.heavyImpact();
        _cancelTimers();
        state = state.copyWith(
          isRecording: false,
          error: _errorToMessage(error),
          remainingSeconds: null,
          totalDurationSeconds: null,
        );
      },
    );
  }

  /// Stops the current recording without scoring.
  Future<void> stopRecording() async {
    _cancelTimers();

    // Stop waveform visualization (disabled - see startRecording)
    // await waveformController.stop();

    final result = await _recorder.stop();

    result.when(
      success: (_) {
        // Haptic feedback on successful stop
        HapticFeedback.lightImpact();
        state = state.copyWith(
          isRecording: false,
          remainingSeconds: null,
          totalDurationSeconds: null,
        );
      },
      failure: (error) {
        // Haptic feedback on failure
        HapticFeedback.heavyImpact();
        state = state.copyWith(
          isRecording: false,
          error: _errorToMessage(error),
          remainingSeconds: null,
          totalDurationSeconds: null,
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
    _cancelTimers();

    state = state.copyWith(
      isScoring: true,
      error: null,
      remainingSeconds: null,
      totalDurationSeconds: null,
    );

    // Stop waveform visualization (disabled - see startRecording)
    // await waveformController.stop();

    final stopResult = await _recorder.stop();

    return stopResult.when(
      success: (filePath) async {
        // Haptic feedback on successful stop
        HapticFeedback.lightImpact();

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
            details: {
              ...?grade.details,
              'accuracy': grade.accuracy,
              'completeness': grade.completeness,
            },
          );

          await _progressRepository.saveAttempt(attempt);

          // Save recording to enable replay functionality
          await _repository.saveLatest(recording);

          // Haptic feedback based on score quality
          if (grade.overall >= 80) {
            HapticFeedback.mediumImpact(); // Good score celebration
          }

          state = state.copyWith(
            isRecording: false,
            isScoring: false,
            hasLatestRecording: true,
            latestGrade: grade,
          );

          return grade;
        } catch (e) {
          // Haptic feedback on scoring failure
          HapticFeedback.heavyImpact();
          state = state.copyWith(
            isRecording: false,
            isScoring: false,
            error: 'Scoring failed: ${e.toString()}',
          );
          return null;
        }
      },
      failure: (error) {
        // Haptic feedback on recording failure
        HapticFeedback.heavyImpact();
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
    debugPrint('🔊 replayLatest called for $textSequenceId');
    final recording = await _repository.getLatest(textSequenceId);
    debugPrint('🔊 Got recording: ${recording?.filePath}');
    if (recording == null) {
      debugPrint('🔊 No recording found, returning');
      return;
    }

    state = state.copyWith(isPlaying: true);
    debugPrint('🔊 Loading audio from ${recording.filePath}');
    await _audioPlayer.load(FileAudioSource(recording.filePath));

    // Get duration and wait for that time + buffer
    final duration = _audioPlayer.duration ?? const Duration(seconds: 5);
    debugPrint('🔊 Audio duration: ${duration.inMilliseconds}ms');

    await _audioPlayer.play();
    debugPrint('🔊 Playing audio');

    // Wait for the duration of the audio plus a small buffer
    await Future.delayed(duration + const Duration(milliseconds: 500));
    debugPrint('🔊 Playback complete (duration elapsed)');

    // Stop to ensure cleanup
    await _audioPlayer.stop();

    state = state.copyWith(isPlaying: false);
  }

  /// Cancels any active recording or playback.
  ///
  /// Stops the recorder if recording is in progress.
  /// Stops the audio player if playback is in progress.
  /// Cancels any active timers.
  /// Resets state to initial values.
  Future<void> cancel() async {
    _cancelTimers();

    // Stop waveform if recording (disabled - see startRecording)
    if (state.isRecording) {
      // await waveformController.stop();
      await _recorder.stop();
    }
    if (state.isPlaying) {
      await _audioPlayer.stop();
    }
    state = const RecordingState();
  }

  @override
  void dispose() {
    _cancelTimers();
    waveformController.dispose();
    _audioPlayer.dispose();
    super.dispose();
  }

  /// Cancels all active timers.
  void _cancelTimers() {
    _autoStopTimer?.cancel();
    _autoStopTimer = null;
    _countdownTimer?.cancel();
    _countdownTimer = null;
    _currentTextSequence = null;
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
