import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../../app/di.dart';
import '../../../app/theme.dart';
import '../../settings/presentation/settings_controller.dart';
import '../../text_sequences/domain/text_sequence.dart';
import '../../progress/domain/text_sequence_progress.dart';
import 'home_controller.dart';
import 'home_state.dart';
import 'practice_sheet.dart';
import 'widgets/record_fab.dart';
import 'widgets/score_bar.dart';

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
    // Watch recording state - single source of truth for recording status
    final recordingState = ref.watch(recordingControllerProvider);
    final recordingController = ref.read(recordingControllerProvider.notifier);

    // Derive recording status from RecordingState
    final recordingStatus = recordingState.recordingStatus;

    // Handle FAB press based on current recording status
    void handleFabPress() async {
      if (state.current == null) return;

      switch (recordingStatus) {
        case RecordingStatus.idle:
          // Start recording
          await recordingController.startRecording(state.current!);
          // Check if recording actually started (no error)
          final newRecordingState = ref.read(recordingControllerProvider);
          if (newRecordingState.error != null && context.mounted) {
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(content: Text(newRecordingState.error!)),
            );
          }
          break;

        case RecordingStatus.recording:
          // Stop and score
          await recordingController.stopAndScore(state.current!);
          await controller.refreshProgress();
          break;

        case RecordingStatus.processing:
          // Do nothing - button is disabled during processing
          break;
      }
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('Home'),
        leading: IconButton(
          icon: const Icon(Icons.list),
          onPressed: () => context.go('/list'),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.bar_chart),
            onPressed: () => context.go('/stats'),
          ),
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: () => context.go('/settings'),
          ),
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
      floatingActionButton: state.current != null && !state.isLoading && !state.isEmptyTracked
          ? RecordFAB(
              status: recordingStatus,
              onPressed: handleFabPress,
            )
          : null,
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
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

class _HomeContent extends ConsumerStatefulWidget {
  const _HomeContent({
    required this.state,
    required this.controller,
  });

  final HomeState state;
  final HomeController controller;

  @override
  ConsumerState<_HomeContent> createState() => _HomeContentState();
}

class _HomeContentState extends ConsumerState<_HomeContent> {
  bool _showPinyin = true;
  bool _initializedFromSettings = false;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (!_initializedFromSettings) {
      final settingsAsync = ref.read(settingsControllerProvider);
      if (settingsAsync case AsyncData(value: final settings)) {
        setState(() {
          _showPinyin = settings.showRomanization;
          _initializedFromSettings = true;
        });
      }
    }
  }

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

  void _playAudio(TextSequence sequence) {
    final voices = sequence.voices;
    if (voices == null || voices.isEmpty) return;

    final controller = ref.read(exampleAudioControllerProvider.notifier);
    controller.play(sequence, voices.first.id);
  }

  @override
  Widget build(BuildContext context) {
    final sequence = widget.state.current!;
    final hasPinyin = sequence.romanization != null &&
                      sequence.romanization!.isNotEmpty;
    final hasAudio = sequence.voices != null && sequence.voices!.isNotEmpty;
    final audioState = ref.watch(exampleAudioControllerProvider);

    return Column(
      children: [
        Expanded(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 24),
            child: Center(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 420),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.stretch,
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
                        textAlign: TextAlign.center,
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
                          textAlign: TextAlign.center,
                        ),
                      ),
                    ],
                    const SizedBox(height: 24),
                    if (hasAudio)
                      Align(
                        alignment: Alignment.center,
                        child: IconButton(
                          onPressed: audioState.isPlaying
                              ? null
                              : () => _playAudio(sequence),
                          icon: Icon(
                            audioState.isPlaying
                                ? Icons.volume_up
                                : Icons.play_arrow,
                          ),
                          iconSize: 32,
                        ),
                      ),
                  ],
                ),
              ),
            ),
          ),
        ),
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 80), // Extra bottom padding for FAB
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              if (widget.state.currentProgress?.bestScore != null) ...[
                Row(
                  children: [
                    Text(
                      'Best: ${widget.state.currentProgress!.bestScore}',
                      style: TextStyle(
                        color: widget.state.currentProgress!.bestScore!.scoreColor,
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: ScoreBar(
                        score: widget.state.currentProgress!.bestScore,
                        height: 8,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
              ],
              Center(
                child: SizedBox(
                  width: 180,
                  height: 52,
                  child: ElevatedButton(
                    onPressed: widget.controller.next,
                    style: ElevatedButton.styleFrom(
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(26),
                      ),
                    ),
                    child: const Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Text('Next', style: TextStyle(fontSize: 18)),
                        SizedBox(width: 8),
                        Icon(Icons.arrow_forward, size: 22),
                      ],
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}
