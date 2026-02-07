import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../app/theme.dart';
import '../../text_sequences/domain/text_sequence.dart';
import '../../progress/domain/text_sequence_progress.dart';
import '../../example_audio/presentation/example_audio_controller.dart';
import '../../recording/presentation/recording_controller.dart';
import '../../recording/presentation/recording_state.dart';

/// Placeholder provider for example audio controller.
/// Will be properly implemented when DI is set up.
final exampleAudioControllerProvider =
    StateNotifierProvider<ExampleAudioController, ExampleAudioState>((ref) {
  throw UnimplementedError(
    'exampleAudioControllerProvider must be overridden in ProviderScope',
  );
});

/// Placeholder provider for recording controller.
/// Will be properly implemented when DI is set up.
final recordingControllerProvider =
    StateNotifierProvider<RecordingController, RecordingState>((ref) {
  throw UnimplementedError(
    'recordingControllerProvider must be overridden in ProviderScope',
  );
});

/// Bottom sheet for practicing a text sequence.
class PracticeSheet extends ConsumerStatefulWidget {
  const PracticeSheet({
    super.key,
    required this.sequence,
    this.progress,
  });

  final TextSequence sequence;
  final TextSequenceProgress? progress;

  @override
  ConsumerState<PracticeSheet> createState() => _PracticeSheetState();
}

class _PracticeSheetState extends ConsumerState<PracticeSheet> {
  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // Drag handle
            Container(
              width: 40,
              height: 4,
              decoration: BoxDecoration(
                color: Colors.grey[400],
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            const SizedBox(height: 24),
            // Text display
            Text(
              widget.sequence.text,
              style: Theme.of(context).textTheme.bodyLarge,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 16),
            _ExampleAudioButtons(sequence: widget.sequence),
            const SizedBox(height: 16),
            _RecordButton(sequence: widget.sequence),
            const SizedBox(height: 16),
            _ReplayButton(sequence: widget.sequence),
            const SizedBox(height: 16),
            _ScoreDisplay(
              progress: widget.progress,
              sequence: widget.sequence,
            ),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }
}

/// Buttons for playing example audio voices.
class _ExampleAudioButtons extends ConsumerWidget {
  const _ExampleAudioButtons({required this.sequence});

  final TextSequence sequence;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final voices = sequence.voices;
    if (voices == null || voices.isEmpty) {
      return const SizedBox.shrink();
    }

    final state = ref.watch(exampleAudioControllerProvider);
    final controller = ref.read(exampleAudioControllerProvider.notifier);

    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: voices.map((voice) {
        return Padding(
          padding: const EdgeInsets.symmetric(horizontal: 4),
          child: OutlinedButton(
            onPressed: state.isPlaying
                ? null
                : () => controller.play(sequence.id, voice.id),
            child: Text(voice.id),
          ),
        );
      }).toList(),
    );
  }
}

/// Button for recording and scoring pronunciation.
class _RecordButton extends ConsumerWidget {
  const _RecordButton({required this.sequence});

  final TextSequence sequence;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(recordingControllerProvider);
    final controller = ref.read(recordingControllerProvider.notifier);

    Widget child;
    if (state.isScoring) {
      child = const SizedBox(
        width: 24,
        height: 24,
        child: CircularProgressIndicator(strokeWidth: 2),
      );
    } else if (state.isRecording) {
      child = const Icon(Icons.stop, color: Colors.red);
    } else {
      child = const Icon(Icons.mic);
    }

    return FloatingActionButton(
      onPressed: state.isScoring
          ? null
          : () {
              if (state.isRecording) {
                controller.stopAndScore(sequence);
              } else {
                controller.startRecording(sequence.id);
              }
            },
      child: child,
    );
  }
}

/// Button for replaying the latest recording.
class _ReplayButton extends ConsumerWidget {
  const _ReplayButton({required this.sequence});

  final TextSequence sequence;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(recordingControllerProvider);
    final controller = ref.read(recordingControllerProvider.notifier);

    if (!state.hasLatestRecording) {
      return const SizedBox.shrink();
    }

    return IconButton(
      icon: const Icon(Icons.replay),
      onPressed: state.isPlaying
          ? null
          : () => controller.replayLatest(sequence.id),
    );
  }
}

/// Displays latest and best scores.
class _ScoreDisplay extends ConsumerWidget {
  const _ScoreDisplay({
    required this.progress,
    required this.sequence,
  });

  final TextSequenceProgress? progress;
  final TextSequence sequence;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        _ScoreLabel(
          label: 'Latest',
          score: progress?.bestScore,
        ),
        const SizedBox(width: 24),
        _ScoreLabel(
          label: 'Best',
          score: progress?.bestScore,
        ),
      ],
    );
  }
}

/// Helper widget to display a score label.
class _ScoreLabel extends StatelessWidget {
  const _ScoreLabel({
    required this.label,
    required this.score,
  });

  final String label;
  final int? score;

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          label,
          style: Theme.of(context).textTheme.bodySmall,
        ),
        Text(
          score?.toString() ?? '-',
          style: TextStyle(
            fontSize: 24,
            fontWeight: FontWeight.bold,
            color: score?.scoreColor ?? AppTheme.subtle,
          ),
        ),
      ],
    );
  }
}
