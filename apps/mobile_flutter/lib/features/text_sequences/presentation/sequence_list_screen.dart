import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import 'sequence_list_controller.dart';
import 'sequence_list_tile.dart';

/// Screen that displays all available text sequences for browsing and selection.
class SequenceListScreen extends ConsumerWidget {
  /// Creates a SequenceListScreen.
  const SequenceListScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(sequenceListControllerProvider);
    final controller = ref.read(sequenceListControllerProvider.notifier);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Sequences'),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => context.go('/'),
        ),
      ),
      body: state.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (error, _) => Center(child: Text(error.toString())),
        data: (items) => ListView.separated(
          itemCount: items.length,
          separatorBuilder: (context, index) => const Divider(),
          itemBuilder: (context, index) {
            final item = items[index];
            return SequenceListTile(
              item: item,
              onTap: () {
                controller.select(item.id);
                context.go('/');
              },
              onToggleTrack: () => controller.toggleTracked(item.id),
            );
          },
        ),
      ),
    );
  }
}
