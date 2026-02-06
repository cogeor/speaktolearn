import 'package:freezed_annotation/freezed_annotation.dart';

part 'recording_state.freezed.dart';

@freezed
class RecordingState with _$RecordingState {
  const factory RecordingState({
    @Default(false) bool isRecording,
    @Default(false) bool isScoring,
    @Default(false) bool isPlaying,
    @Default(false) bool hasLatestRecording,
    String? error,
  }) = _RecordingState;
}
