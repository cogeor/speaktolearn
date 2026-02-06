import 'package:freezed_annotation/freezed_annotation.dart';

import '../../text_sequences/domain/text_sequence.dart';
import '../../progress/domain/text_sequence_progress.dart';

part 'home_state.freezed.dart';

@freezed
class HomeState with _$HomeState {
  const factory HomeState({
    TextSequence? current,
    TextSequenceProgress? currentProgress,
    @Default(true) bool isLoading,
    @Default(false) bool isEmptyTracked,
  }) = _HomeState;
}
