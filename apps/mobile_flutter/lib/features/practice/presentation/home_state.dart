import 'package:freezed_annotation/freezed_annotation.dart';

import '../../text_sequences/domain/text_sequence.dart';
import '../../progress/domain/text_sequence_progress.dart';

part 'home_state.freezed.dart';

/// Recording status for home screen FAB display.
enum RecordingStatus {
  idle,       // Ready to record - show mic icon
  recording,  // Currently recording - show stop icon, red pulsing
  processing, // Scoring in progress - show spinner
}

@freezed
class HomeState with _$HomeState {
  const factory HomeState({
    TextSequence? current,
    TextSequenceProgress? currentProgress,
    @Default(true) bool isLoading,
    @Default(false) bool isEmptyTracked,
    @Default(RecordingStatus.idle) RecordingStatus recordingStatus,
    int? latestScore,
  }) = _HomeState;
}
