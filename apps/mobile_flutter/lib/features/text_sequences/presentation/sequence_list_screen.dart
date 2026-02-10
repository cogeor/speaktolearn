import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../../app/di.dart';
import 'sequence_list_controller.dart';
import 'sequence_list_tile.dart';
import 'widgets/hsk_filter_chips.dart';

/// Screen that displays all available text sequences for browsing and selection.
class SequenceListScreen extends ConsumerWidget {
  /// Creates a SequenceListScreen.
  const SequenceListScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(sequenceListControllerProvider);
    final controller = ref.read(sequenceListControllerProvider.notifier);
    final selectedLevels = ref.watch(hskFilterProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Sentences'),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => context.go('/'),
        ),
      ),
      body: Column(
        children: [
          // Filter chips
          HskFilterChips(
            selectedLevels: selectedLevels,
            onToggle: controller.toggleHskFilter,
            onClear: controller.clearHskFilter,
          ),
          // Result count indicator
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
            child: state.when(
              loading: () => const SizedBox.shrink(),
              error: (_, __) => const SizedBox.shrink(),
              data: (items) => Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  '${items.length} sequence${items.length == 1 ? '' : 's'}',
                  style: TextStyle(
                    color: Theme.of(context).colorScheme.secondary,
                    fontSize: 12,
                  ),
                ),
              ),
            ),
          ),
          // Divider
          const Divider(height: 1),
          // List
          Expanded(
            child: state.when(
              loading: () => const Center(child: CircularProgressIndicator()),
              error: (error, _) => Center(child: Text(error.toString())),
              data: (items) => items.isEmpty
                  ? const Center(child: Text('No sequences match the filter'))
                  : ListView.separated(
                      itemCount: items.length,
                      separatorBuilder: (context, index) => const Divider(),
                      itemBuilder: (context, index) {
                        final item = items[index];
                        return SequenceListTile(
                          item: item,
                          onTap: () async {
                            await ref.read(audioPlayerProvider).stop();
                            if (!context.mounted) return;
                            controller.select(item.id);
                            context.go('/');
                          },
                        );
                      },
                    ),
            ),
          ),
        ],
      ),
    );
  }
}
