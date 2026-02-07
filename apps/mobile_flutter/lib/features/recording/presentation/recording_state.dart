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
  }) = _RecordingState;
}
