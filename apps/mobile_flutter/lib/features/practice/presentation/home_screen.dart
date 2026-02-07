import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../../app/di.dart';
import '../../../app/theme.dart';
import '../../text_sequences/domain/text_sequence.dart';
import '../../progress/domain/text_sequence_progress.dart';
import 'home_controller.dart';
import 'home_state.dart';
import 'practice_sheet.dart';

/// Provider for the home screen controller.
final homeControllerProvider =
    StateNotifierProvider<HomeController, HomeState>((ref) {
  return HomeController(
    textSequenceRepository: ref.watch(textSequenceRepositoryProvider),
    progressRepository: ref.watch(progressRepositoryProvider),
    getNextTrackedSequence: ref.watch(getNextTrackedSequenceProvider),
  );
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

class _HomeContent extends StatefulWidget {
  const _HomeContent({
    required this.state,
    required this.controller,
  });

  final HomeState state;
  final HomeController controller;

  @override
  State<_HomeContent> createState() => _HomeContentState();
}

class _HomeContentState extends State<_HomeContent> {
  bool _showPinyin = true;

  void _showPracticeSheet(
    BuildContext context,
    TextSequence sequence,
    TextSequenceProgress? progress,
  ) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => PracticeSheet(
        sequence: sequence,
        progress: progress,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final sequence = widget.state.current!;
    final hasPinyin = sequence.romanization != null &&
                      sequence.romanization!.isNotEmpty;

    return Column(
      children: [
        Expanded(
          child: Center(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                GestureDetector(
                  onTap: () {
                    _showPracticeSheet(
                      context,
                      sequence,
                      widget.state.currentProgress,
                    );
                  },
                  child: Text(
                    sequence.text,
                    style: Theme.of(context).textTheme.displayLarge,
                  ),
                ),
                if (hasPinyin) ...[
                  const SizedBox(height: 12),
                  GestureDetector(
                    onTap: () => setState(() => _showPinyin = !_showPinyin),
                    child: Text(
                      _showPinyin
                          ? sequence.romanization!
                          : '(tap to show pinyin)',
                      style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        color: _showPinyin
                            ? Theme.of(context).colorScheme.secondary
                            : AppTheme.subtle,
                        fontStyle: _showPinyin
                            ? FontStyle.normal
                            : FontStyle.italic,
                      ),
                    ),
                  ),
                ],
              ],
            ),
          ),
        ),
        Padding(
          padding: const EdgeInsets.all(16),
          child: Row(
            children: [
              if (widget.state.currentProgress?.bestScore != null)
                Text(
                  'Best: ${widget.state.currentProgress!.bestScore}',
                  style: TextStyle(
                    color: widget.state.currentProgress!.bestScore!.scoreColor,
                  ),
                ),
              const Spacer(),
              ElevatedButton(
                onPressed: widget.controller.next,
                child: const Text('Next'),
              ),
            ],
          ),
        ),
      ],
    );
  }
}
