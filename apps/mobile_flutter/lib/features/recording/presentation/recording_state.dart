import 'package:freezed_annotation/freezed_annotation.dart';

import '../../scoring/domain/grade.dart';

part 'recording_state.freezed.dart';

/// Explicit phases for the recording state machine.
///
/// State flow: idle -> recording -> saving -> scoring -> complete
/// Error at any phase transitions to error, which can be dismissed back to idle.
enum RecordingPhase {
  /// Ready to record
  idle,

  /// Actively recording audio
  recording,

  /// Recording stopped, saving file
  saving,

  /// File saved, scoring pronunciation
  scoring,

  /// Scoring complete, showing results
  complete,

  /// An error occurred
  error,
}

@freezed
class RecordingState with _$RecordingState {
  const RecordingState._();

  const factory RecordingState({
    /// The current phase in the recording state machine.
    @Default(RecordingPhase.idle) RecordingPhase phase,
    @Default(false) bool isPlaying,
    @Default(false) bool hasLatestRecording,
    String? error,

    /// The latest grade from the most recent scoring attempt.
    Grade? latestGrade,

    /// Remaining seconds in the auto-stop countdown. Null when not recording.
    int? remainingSeconds,

    /// Total duration in seconds for progress calculation. Null when not recording.
    int? totalDurationSeconds,
  }) = _RecordingState;

  /// Whether the recorder is actively recording audio.
  /// Derived from [phase] for backward compatibility.
  bool get isRecording => phase == RecordingPhase.recording;

  /// Whether scoring is in progress (saving or scoring phases).
  /// Derived from [phase] for backward compatibility.
  bool get isScoring =>
      phase == RecordingPhase.saving || phase == RecordingPhase.scoring;
}
