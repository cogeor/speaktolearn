import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../progress/domain/progress_repository.dart';
import '../../selection/domain/get_next_tracked.dart';
import '../../text_sequences/domain/text_sequence_repository.dart';
import 'home_state.dart';

/// Controller for the home screen that manages the current practice sequence.
class HomeController extends StateNotifier<HomeState> {
  HomeController({
    required TextSequenceRepository textSequenceRepository,
    required ProgressRepository progressRepository,
    required GetNextTrackedSequence getNextTrackedSequence,
  })  : _textSequenceRepository = textSequenceRepository,
        _progressRepository = progressRepository,
        _getNextTrackedSequence = getNextTrackedSequence,
        super(const HomeState()) {
    _init();
  }

  final TextSequenceRepository _textSequenceRepository;
  final ProgressRepository _progressRepository;
  final GetNextTrackedSequence _getNextTrackedSequence;

  Future<void> _init() async {
    state = state.copyWith(isLoading: true);

    final sequence = await _getNextTrackedSequence();

    if (sequence == null) {
      state = state.copyWith(
        isLoading: false,
        isEmptyTracked: true,
      );
      return;
    }

    final progress = await _progressRepository.getProgress(sequence.id);

    state = state.copyWith(
      current: sequence,
      currentProgress: progress,
      isLoading: false,
      isEmptyTracked: false,
    );
  }

  /// Advances to the next tracked sequence, excluding the current one.
  Future<void> next() async {
    final sequence = await _getNextTrackedSequence(currentId: state.current?.id);

    if (sequence == null) {
      state = state.copyWith(isEmptyTracked: true);
      return;
    }

    final progress = await _progressRepository.getProgress(sequence.id);

    state = state.copyWith(
      current: sequence,
      currentProgress: progress,
      isEmptyTracked: false,
    );
  }

  /// Sets the current sequence by its ID.
  ///
  /// Used when selecting a sequence from the list screen.
  Future<void> setCurrentSequence(String id) async {
    final sequence = await _textSequenceRepository.getById(id);

    if (sequence == null) {
      return;
    }

    final progress = await _progressRepository.getProgress(sequence.id);

    state = state.copyWith(
      current: sequence,
      currentProgress: progress,
      isEmptyTracked: false,
    );
  }

  /// Toggles the tracked status of the current sequence.
  Future<void> toggleTracked() async {
    if (state.current == null) {
      return;
    }

    await _progressRepository.toggleTracked(state.current!.id);
    await refreshProgress();
  }

  /// Refreshes the progress for the current sequence.
  ///
  /// Used after scoring to update the best score display.
  Future<void> refreshProgress() async {
    if (state.current == null) {
      return;
    }

    final progress = await _progressRepository.getProgress(state.current!.id);

    state = state.copyWith(
      currentProgress: progress,
      recordingStatus: RecordingStatus.idle,
    );
  }

  /// Updates the recording status for FAB display.
  void setRecordingStatus(RecordingStatus status) {
    state = state.copyWith(recordingStatus: status);
  }

  /// Sets the latest score after a recording attempt.
  void setLatestScore(int? score) {
    state = state.copyWith(latestScore: score);
  }
}
