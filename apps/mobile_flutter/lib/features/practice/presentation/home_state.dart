import 'package:freezed_annotation/freezed_annotation.dart';

import '../../recording/presentation/recording_state.dart';
import '../../text_sequences/domain/text_sequence.dart';
import '../../progress/domain/text_sequence_progress.dart';

part 'home_state.freezed.dart';

/// Recording status for FAB display.
/// Derived from [RecordingState] to maintain single source of truth.
enum RecordingStatus {
  idle, // Ready to record - show mic icon
  recording, // Currently recording - show stop icon, red pulsing
  processing, // Scoring in progress - show spinner
}

/// Extension to derive [RecordingStatus] from [RecordingState].
extension RecordingStatusExtension on RecordingState {
  RecordingStatus get recordingStatus {
    if (isScoring) return RecordingStatus.processing;
    if (isRecording) return RecordingStatus.recording;
    return RecordingStatus.idle;
  }
}

@freezed
class HomeState with _$HomeState {
  const factory HomeState({
    TextSequence? current,
    TextSequenceProgress? currentProgress,
    @Default(true) bool isLoading,
    @Default(false) bool isEmptyTracked,
  }) = _HomeState;
}
