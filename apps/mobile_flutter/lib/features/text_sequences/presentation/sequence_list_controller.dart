import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../app/di.dart';
import '../../practice/presentation/home_screen.dart';
import 'sequence_list_item.dart';

/// Provider for HSK level filter state.
/// Empty set means show all levels.
final hskFilterProvider = StateProvider<Set<int>>((ref) => {});

/// Provider for the sequence list controller.
final sequenceListControllerProvider =
    AutoDisposeAsyncNotifierProvider<
      SequenceListController,
      List<SequenceListItem>
    >(SequenceListController.new);

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
    final hskFilter = ref.watch(hskFilterProvider);

    // Get all sequences
    final sequences = await textSequenceRepo.getAll();

    // Get progress for all sequences
    final ids = sequences.map((s) => s.id).toList();
    final progressMap = await progressRepo.getProgressMap(ids);

    // Map to SequenceListItem with progress data
    var items = sequences.map((sequence) {
      final progress = progressMap[sequence.id];
      return SequenceListItem(
        id: sequence.id,
        text: sequence.text,
        lastRating: progress?.lastRating,
        hskLevel: sequence.hskLevel,
      );
    }).toList();

    // Apply HSK filter if set
    if (hskFilter.isNotEmpty) {
      items = items.where((item) {
        // Include items with matching level
        return item.hskLevel != null && hskFilter.contains(item.hskLevel);
      }).toList();
    }

    // Sort by HSK level, then alphabetically
    items.sort((a, b) {
      // Sort by HSK level (null at end)
      final aLevel = a.hskLevel ?? 999;
      final bLevel = b.hskLevel ?? 999;
      if (aLevel != bLevel) {
        return aLevel.compareTo(bLevel);
      }
      return a.text.compareTo(b.text);
    });

    return items;
  }

  /// Selects a sequence for practice on the home screen.
  void select(String id) {
    final homeController = ref.read(homeControllerProvider.notifier);
    homeController.setCurrentSequence(id);
  }

  /// Toggles an HSK level in the filter.
  ///
  /// The list auto-rebuilds because build() watches hskFilterProvider.
  void toggleHskFilter(int level) {
    final current = ref.read(hskFilterProvider);
    if (current.contains(level)) {
      ref.read(hskFilterProvider.notifier).state = current.difference({level});
    } else {
      ref.read(hskFilterProvider.notifier).state = current.union({level});
    }
  }

  /// Clears all HSK filters (show all).
  ///
  /// The list auto-rebuilds because build() watches hskFilterProvider.
  void clearHskFilter() {
    ref.read(hskFilterProvider.notifier).state = {};
  }
}
