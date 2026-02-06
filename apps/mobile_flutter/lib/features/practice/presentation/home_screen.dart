import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../../app/theme.dart';
import 'home_controller.dart';
import 'home_state.dart';

/// Provider for the home screen controller.
///
/// TODO: Wire up with actual dependencies.
final homeControllerProvider =
    StateNotifierProvider<HomeController, HomeState>((ref) {
  throw UnimplementedError('homeControllerProvider is not yet wired');
});

/// Home screen that displays the current practice sequence.
class HomeScreen extends ConsumerWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(homeControllerProvider);
    final controller = ref.read(homeControllerProvider.notifier);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Home'),
        leading: IconButton(
          icon: const Icon(Icons.list),
          onPressed: () => context.go('/list'),
        ),
        actions: [
          IconButton(
            icon: Icon(
              state.currentProgress?.tracked == true
                  ? Icons.bookmark
                  : Icons.bookmark_border,
            ),
            onPressed: controller.toggleTracked,
          ),
        ],
      ),
      body: state.isLoading
          ? const Center(child: CircularProgressIndicator())
          : state.isEmptyTracked
              ? const _EmptyState()
              : _HomeContent(state: state, controller: controller),
    );
  }
}

class _EmptyState extends StatelessWidget {
  const _EmptyState();

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            'No tracked sequences',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: 16),
          ElevatedButton(
            onPressed: () => context.go('/list'),
            child: const Text('Open list'),
          ),
        ],
      ),
    );
  }
}

class _HomeContent extends StatelessWidget {
  const _HomeContent({
    required this.state,
    required this.controller,
  });

  final HomeState state;
  final HomeController controller;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Expanded(
          child: Center(
            child: GestureDetector(
              onTap: () {
                // TODO: Open practice sheet
                debugPrint('Opening practice sheet');
              },
              child: Text(
                state.current!.text,
                style: Theme.of(context).textTheme.displayLarge,
              ),
            ),
          ),
        ),
        Padding(
          padding: const EdgeInsets.all(16),
          child: Row(
            children: [
              if (state.currentProgress?.bestScore != null)
                Text(
                  'Best: ${state.currentProgress!.bestScore}',
                  style: TextStyle(
                    color: state.currentProgress!.bestScore!.scoreColor,
                  ),
                ),
              const Spacer(),
              ElevatedButton(
                onPressed: controller.next,
                child: const Text('Next'),
              ),
            ],
          ),
        ),
      ],
    );
  }
}
