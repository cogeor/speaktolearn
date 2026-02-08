import 'package:freezed_annotation/freezed_annotation.dart';

import '../../scoring/domain/grade.dart';

part 'recording_state.freezed.dart';

@freezed
class RecordingState with _$RecordingState {
  const factory RecordingState({
    @Default(false) bool isRecording,
    @Default(false) bool isScoring,
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
}
