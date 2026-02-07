import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../app/di.dart';
import '../../practice/presentation/home_screen.dart';
import 'sequence_list_item.dart';

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
        hskLevel: sequence.hskLevel,
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
