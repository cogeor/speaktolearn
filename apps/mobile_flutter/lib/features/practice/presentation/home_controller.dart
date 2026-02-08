import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../progress/domain/progress_repository.dart';
import '../../selection/domain/get_next_by_level.dart';
import '../../settings/domain/app_settings.dart';
import '../../text_sequences/domain/text_sequence_repository.dart';
import 'home_state.dart';

/// Controller for the home screen that manages the current practice sequence.
class HomeController extends StateNotifier<HomeState> {
  HomeController({
    required TextSequenceRepository textSequenceRepository,
    required ProgressRepository progressRepository,
    required GetNextByLevel getNextByLevel,
    required AsyncValue<AppSettings> settings,
  }) : _textSequenceRepository = textSequenceRepository,
       _progressRepository = progressRepository,
       _getNextByLevel = getNextByLevel,
       _settings = settings,
       super(const HomeState()) {
    _init();
  }

  final TextSequenceRepository _textSequenceRepository;
  final ProgressRepository _progressRepository;
  final GetNextByLevel _getNextByLevel;
  final AsyncValue<AppSettings> _settings;

  int get _currentLevel => _settings.valueOrNull?.currentLevel ?? 1;

  Future<void> _init() async {
    state = state.copyWith(isLoading: true);

    final sequence = await _getNextByLevel(level: _currentLevel);

    if (sequence == null) {
      state = state.copyWith(isLoading: false, isEmptyTracked: true);
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

  /// Advances to a random sequence from the current level, excluding the current one.
  Future<void> next() async {
    final sequence = await _getNextByLevel(
      level: _currentLevel,
      currentId: state.current?.id,
    );

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

  /// Refreshes the progress for the current sequence.
  ///
  /// Used after scoring to update the best score display.
  Future<void> refreshProgress() async {
    if (state.current == null) {
      return;
    }

    final progress = await _progressRepository.getProgress(state.current!.id);

    state = state.copyWith(currentProgress: progress);
  }
}
