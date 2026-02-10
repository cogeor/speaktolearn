import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:uuid/uuid.dart';

import '../../progress/domain/progress_repository.dart';
import '../../progress/domain/rating_attempt.dart';
import '../../progress/domain/sentence_rating.dart';
import '../../selection/domain/get_next_by_level.dart';
import '../../selection/domain/sentence_queue_manager.dart';
import '../../settings/domain/app_settings.dart';
import '../../text_sequences/domain/text_sequence.dart';
import '../../text_sequences/domain/text_sequence_repository.dart';
import 'home_state.dart';

/// Controller for the home screen that manages the current practice sequence.
class HomeController extends StateNotifier<HomeState> {
  HomeController({
    required TextSequenceRepository textSequenceRepository,
    required ProgressRepository progressRepository,
    required GetNextByLevel getNextByLevel,
    required AsyncValue<AppSettings> settings,
    SentenceQueueManager? queueManager,
  }) : _textSequenceRepository = textSequenceRepository,
       _progressRepository = progressRepository,
       _getNextByLevel = getNextByLevel,
       _settings = settings,
       _queueManager = queueManager,
       super(const HomeState()) {
    _init();
  }

  final TextSequenceRepository _textSequenceRepository;
  final ProgressRepository _progressRepository;
  final GetNextByLevel _getNextByLevel;
  final SentenceQueueManager? _queueManager;
  final AsyncValue<AppSettings> _settings;

  int get _currentLevel => _settings.valueOrNull?.currentLevel ?? 1;

  Future<void> _init() async {
    if (!mounted) return;
    state = state.copyWith(isLoading: true);

    final sequence = await _getNextSequence();

    if (!mounted) return;
    if (sequence == null) {
      state = state.copyWith(isLoading: false);
      return;
    }

    final progress = await _progressRepository.getProgress(sequence.id);

    if (!mounted) return;
    state = state.copyWith(
      current: sequence,
      currentProgress: progress,
      isLoading: false,
    );
  }

  /// Gets the next sequence using the queue manager if available,
  /// otherwise falls back to random selection.
  Future<TextSequence?> _getNextSequence({String? excludeId}) async {
    if (_queueManager != null) {
      return _queueManager.getNext(level: _currentLevel, excludeId: excludeId);
    }
    return _getNextByLevel(level: _currentLevel, currentId: excludeId);
  }

  /// Advances to the next sequence from the current level, excluding the current one.
  ///
  /// Uses queue-based prioritization if a queue manager is provided,
  /// otherwise falls back to random selection.
  Future<void> next() async {
    final sequence = await _getNextSequence(excludeId: state.current?.id);

    if (!mounted || sequence == null) return;

    final progress = await _progressRepository.getProgress(sequence.id);

    if (!mounted) return;
    state = state.copyWith(current: sequence, currentProgress: progress);
  }

  /// Clears the sentence queue (if using queue manager).
  ///
  /// Should be called when the user changes levels to force a refill
  /// with sentences from the new level.
  void clearQueue() {
    _queueManager?.clear();
  }

  /// Sets the current sequence by its ID.
  ///
  /// Used when selecting a sequence from the list screen.
  Future<void> setCurrentSequence(String id) async {
    final sequence = await _textSequenceRepository.getById(id);

    if (!mounted || sequence == null) return;

    final progress = await _progressRepository.getProgress(sequence.id);

    if (!mounted) return;
    state = state.copyWith(current: sequence, currentProgress: progress);
  }

  /// Refreshes the progress for the current sequence.
  ///
  /// Used after scoring to update the best score display.
  Future<void> refreshProgress() async {
    if (!mounted || state.current == null) return;

    final progress = await _progressRepository.getProgress(state.current!.id);

    if (!mounted) return;
    state = state.copyWith(currentProgress: progress);
  }

  /// Records a self-reported rating for the current sentence and advances to next.
  Future<void> rateAndNext(SentenceRating rating) async {
    if (state.current == null) return;

    final attempt = RatingAttempt(
      id: const Uuid().v4(),
      textSequenceId: state.current!.id,
      gradedAt: DateTime.now(),
      rating: rating,
    );

    await _progressRepository.saveAttempt(attempt);
    await next();
  }
}
