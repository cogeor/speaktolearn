import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../practice/presentation/home_controller.dart';
import '../../practice/presentation/home_state.dart';
import '../../progress/domain/progress_repository.dart';
import '../domain/text_sequence_repository.dart';
import 'sequence_list_item.dart';

/// Provider for text sequence repository.
///
/// Placeholder that throws until wired in di.dart.
final textSequenceRepositoryProvider = Provider<TextSequenceRepository>((ref) {
  throw UnimplementedError('textSequenceRepositoryProvider must be overridden');
});

/// Provider for progress repository.
///
/// Placeholder that throws until wired in di.dart.
final progressRepositoryProvider = Provider<ProgressRepository>((ref) {
  throw UnimplementedError('progressRepositoryProvider must be overridden');
});

/// Provider for home controller.
///
/// Placeholder that throws until wired in home_screen.dart.
final homeControllerProvider =
    StateNotifierProvider<HomeController, HomeState>((ref) {
  throw UnimplementedError('homeControllerProvider must be overridden');
});

/// Provider for the sequence list controller.
final sequenceListControllerProvider = AutoDisposeAsyncNotifierProvider<
    SequenceListController, List<SequenceListItem>>(
  SequenceListController.new,
);

/// Controller that manages the list of text sequences for browsing.
///
/// Loads all sequences with their progress and provides methods for
/// toggling tracked status and selecting a sequence for practice.
class SequenceListController
    extends AutoDisposeAsyncNotifier<List<SequenceListItem>> {
  @override
  Future<List<SequenceListItem>> build() async {
    final textSequenceRepo = ref.watch(textSequenceRepositoryProvider);
    final progressRepo = ref.watch(progressRepositoryProvider);

    // Get all sequences
    final sequences = await textSequenceRepo.getAll();

    // Get progress for all sequences
    final ids = sequences.map((s) => s.id).toList();
    final progressMap = await progressRepo.getProgressMap(ids);

    // Map to SequenceListItem with progress data
    final items = sequences.map((sequence) {
      final progress = progressMap[sequence.id];
      return SequenceListItem(
        id: sequence.id,
        text: sequence.text,
        isTracked: progress?.tracked ?? false,
        bestScore: progress?.bestScore,
      );
    }).toList();

    // Sort: tracked first, then alphabetically by text
    items.sort((a, b) {
      if (a.isTracked != b.isTracked) {
        return a.isTracked ? -1 : 1;
      }
      return a.text.compareTo(b.text);
    });

    return items;
  }

  /// Toggles the tracked status of a sequence.
  Future<void> toggleTracked(String id) async {
    final progressRepo = ref.read(progressRepositoryProvider);
    await progressRepo.toggleTracked(id);

    // Refresh the list
    ref.invalidateSelf();
  }

  /// Selects a sequence for practice on the home screen.
  void select(String id) {
    final homeController = ref.read(homeControllerProvider.notifier);
    homeController.setCurrentSequence(id);
  }
}
