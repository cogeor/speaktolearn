import 'dart:async';

import 'package:audio_waveforms/audio_waveforms.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:uuid/uuid.dart';

import '../../../core/audio/audio_player.dart' show AudioPlayer, PlaybackState;
import '../../../core/audio/audio_source.dart';
import '../../text_sequences/domain/text_sequence.dart';
import '../../scoring/domain/pronunciation_scorer.dart';
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
///
/// Haptics respect device settings via Flutter's HapticFeedback class.
class RecordingController extends StateNotifier<RecordingState> {
  RecordingController({
    required AudioRecorder recorder,
    required RecordingRepository repository,
    required AudioPlayer audioPlayer,
    required PronunciationScorer scorer,
  }) : _recorder = recorder,
       _repository = repository,
       _audioPlayer = audioPlayer,
       _scorer = scorer,
       super(const RecordingState());

  final AudioRecorder _recorder;
  final RecordingRepository _repository;
  final AudioPlayer _audioPlayer;
  final PronunciationScorer _scorer;

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
  /// Transitions to [RecordingPhase.recording] on success.
  /// Starts an auto-stop timer based on the expected duration.
  /// On failure, transitions to [RecordingPhase.error] with error message.
  Future<void> startRecording(TextSequence textSequence) async {
    // Calculate expected duration from text
    final duration = calculateRecordingDuration(
      textSequence.text,
      textSequence.language,
    );
    _remainingSeconds = duration.inSeconds;
    _currentTextSequence = textSequence;

    state = state.copyWith(
      phase: RecordingPhase.recording,
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
            stopAndSave(_currentTextSequence!);
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
          phase: RecordingPhase.error,
          error: _errorToMessage(error),
          remainingSeconds: null,
          totalDurationSeconds: null,
        );
      },
    );
  }

  /// Stops the current recording without scoring.
  ///
  /// Transitions to [RecordingPhase.idle] on success.
  /// Transitions to [RecordingPhase.error] on failure.
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
          phase: RecordingPhase.idle,
          remainingSeconds: null,
          totalDurationSeconds: null,
        );
      },
      failure: (error) {
        // Haptic feedback on failure
        HapticFeedback.heavyImpact();
        state = state.copyWith(
          phase: RecordingPhase.error,
          error: _errorToMessage(error),
          remainingSeconds: null,
          totalDurationSeconds: null,
        );
      },
    );
  }

  /// Stops recording and saves the audio file.
  ///
  /// State transitions: recording -> saving -> complete
  /// On success, saves the recording and transitions to complete phase.
  /// On failure, transitions to [RecordingPhase.error].
  Future<void> stopAndSave(TextSequence textSequence) async {
    _cancelTimers();

    // Transition to saving phase
    state = state.copyWith(
      phase: RecordingPhase.saving,
      error: null,
      remainingSeconds: null,
      totalDurationSeconds: null,
    );

    // Stop waveform visualization (disabled - see startRecording)
    // await waveformController.stop();

    final stopResult = await _recorder.stop();

    stopResult.when(
      success: (filePath) async {
        // Haptic feedback on successful stop
        HapticFeedback.lightImpact();

        await _saveRecording(filePath, textSequence);
      },
      failure: (error) {
        // Haptic feedback on recording failure
        HapticFeedback.heavyImpact();
        state = state.copyWith(
          phase: RecordingPhase.error,
          error: _errorToMessage(error),
        );
      },
    );
  }

  /// Saves the recording file and transitions to complete phase.
  ///
  /// Creates a [Recording] object and saves it to the repository.
  /// Scores the pronunciation using the ML scorer and updates state with the grade.
  Future<void> _saveRecording(
    String filePath,
    TextSequence textSequence,
  ) async {
    try {
      final recording = Recording(
        id: const Uuid().v4(),
        textSequenceId: textSequence.id,
        createdAt: DateTime.now(),
        filePath: filePath,
      );

      // Save recording to enable replay functionality
      await _repository.saveLatest(recording);

      // Score the pronunciation using ML scorer
      final grade = await _scorer.score(textSequence, recording);

      // Transition to complete phase after file is saved and scored
      state = state.copyWith(
        phase: RecordingPhase.complete,
        hasLatestRecording: true,
        hasPlayedBack: false,
        latestGrade: grade,
      );
    } catch (e) {
      HapticFeedback.heavyImpact();
      state = state.copyWith(
        phase: RecordingPhase.error,
        error: 'Failed to save recording: ${e.toString()}',
      );
    }
  }

  /// Checks if a latest recording exists for the given text sequence.
  ///
  /// Updates [RecordingState.hasLatestRecording] based on the result.
  /// Clears [latestGrade] since we're switching to a different sentence.
  Future<void> checkLatestRecording(String textSequenceId) async {
    final recording = await _repository.getLatest(textSequenceId);
    state = state.copyWith(
      hasLatestRecording: recording != null,
      latestGrade: null,
    );
  }

  /// Replays the latest recording for the given text sequence.
  ///
  /// Sets [RecordingState.isPlaying] to true while playing.
  /// Sets [RecordingState.hasPlayedBack] to true after playback completes.
  /// Does nothing if no recording exists for the sequence.
  Future<void> replayLatest(String textSequenceId) async {
    debugPrint('replayLatest called for $textSequenceId');
    final recording = await _repository.getLatest(textSequenceId);
    debugPrint('Got recording: ${recording?.filePath}');
    if (recording == null) {
      debugPrint('No recording found, returning');
      return;
    }

    state = state.copyWith(isPlaying: true);
    debugPrint('Loading audio from ${recording.filePath}');
    await _audioPlayer.load(FileAudioSource(recording.filePath));

    // Subscribe to state stream BEFORE calling play to avoid missing events
    final completionFuture = _audioPlayer.stateStream
        .firstWhere(
          (s) => s == PlaybackState.completed || s == PlaybackState.idle,
        )
        .timeout(
          const Duration(seconds: 30),
          onTimeout: () => PlaybackState.completed,
        );

    debugPrint('Playing audio');
    await _audioPlayer.play();

    // Wait for playback to complete
    await completionFuture;
    debugPrint('Playback complete');

    // Mark playback complete - enables rating buttons
    state = state.copyWith(isPlaying: false, hasPlayedBack: true);
  }

  /// Resets playback state for the next sentence.
  ///
  /// Called after rating a sentence to reset the state for the next one.
  /// Sets [hasPlayedBack] and [hasLatestRecording] to false.
  /// Clears the latest grade.
  void resetPlaybackState() {
    state = state.copyWith(
      hasPlayedBack: false,
      hasLatestRecording: false,
      latestGrade: null,
    );
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
