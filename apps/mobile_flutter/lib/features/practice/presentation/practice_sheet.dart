import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../app/di.dart';
import '../../../app/theme.dart';
import '../../text_sequences/domain/text_sequence.dart';
import '../../progress/domain/text_sequence_progress.dart';
import '../../example_audio/presentation/example_audio_controller.dart';
import '../../recording/presentation/widgets/recording_waveform.dart';
import 'widgets/score_bar.dart';

/// Provider for example audio controller.
final exampleAudioControllerProvider =
    StateNotifierProvider<ExampleAudioController, ExampleAudioState>((ref) {
  return ExampleAudioController(
    player: ref.watch(audioPlayerProvider),
    repository: ref.watch(exampleAudioRepositoryProvider),
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
            const _RecordingWaveformDisplay(),
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
                : () => controller.play(sequence, voice.id),
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

/// Displays latest and best scores with detailed breakdown.
class _ScoreDisplay extends ConsumerWidget {
  const _ScoreDisplay({
    required this.progress,
    required this.sequence,
  });

  final TextSequenceProgress? progress;
  final TextSequence sequence;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final recordingState = ref.watch(recordingControllerProvider);
    final latestGrade = recordingState.latestGrade;

    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        // Main score row
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            _ScoreLabel(
              label: 'Latest',
              score: progress?.lastScore,
            ),
            const SizedBox(width: 24),
            _ScoreLabel(
              label: 'Best',
              score: progress?.bestScore,
            ),
          ],
        ),
        // Detailed breakdown (if available from latest grade)
        if (latestGrade != null &&
            (latestGrade.accuracy != null || latestGrade.completeness != null)) ...[
          const SizedBox(height: 16),
          _DetailedScoreRow(
            accuracy: latestGrade.accuracy,
            completeness: latestGrade.completeness,
          ),
        ],
        // Recognized text comparison
        if (latestGrade?.recognizedText != null) ...[
          const SizedBox(height: 16),
          _RecognizedTextComparison(
            expected: sequence.text,
            recognized: latestGrade!.recognizedText!,
          ),
        ],
      ],
    );
  }
}

/// Row showing accuracy and completeness scores.
class _DetailedScoreRow extends StatelessWidget {
  const _DetailedScoreRow({
    this.accuracy,
    this.completeness,
  });

  final int? accuracy;
  final int? completeness;

  @override
  Widget build(BuildContext context) {
    if (accuracy == null && completeness == null) {
      return const SizedBox.shrink();
    }

    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        if (accuracy != null) ...[
          _MiniScoreLabel(label: 'Accuracy', score: accuracy!),
          const SizedBox(width: 16),
        ],
        if (completeness != null)
          _MiniScoreLabel(label: 'Completeness', score: completeness!),
      ],
    );
  }
}

/// Small score label for secondary metrics.
class _MiniScoreLabel extends StatelessWidget {
  const _MiniScoreLabel({
    required this.label,
    required this.score,
  });

  final String label;
  final int score;

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          label,
          style: Theme.of(context).textTheme.labelSmall,
        ),
        Text(
          '$score%',
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w500,
            color: score.scoreColor,
          ),
        ),
      ],
    );
  }
}

/// Displays expected vs recognized text for comparison.
class _RecognizedTextComparison extends StatelessWidget {
  const _RecognizedTextComparison({
    required this.expected,
    required this.recognized,
  });

  final String expected;
  final String recognized;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'You said:',
            style: Theme.of(context).textTheme.labelSmall,
          ),
          const SizedBox(height: 4),
          Text(
            recognized.isNotEmpty ? recognized : '(nothing detected)',
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
              fontStyle: recognized.isEmpty ? FontStyle.italic : FontStyle.normal,
            ),
          ),
        ],
      ),
    );
  }
}

/// Helper widget to display a score label with visual bar.
class _ScoreLabel extends StatelessWidget {
  const _ScoreLabel({
    required this.label,
    required this.score,
  });

  final String label;
  final int? score;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 80,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            label,
            style: Theme.of(context).textTheme.bodySmall,
          ),
          const SizedBox(height: 4),
          Text(
            score?.toString() ?? '-',
            style: TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.bold,
              color: score?.scoreColor ?? AppTheme.subtle,
            ),
          ),
          const SizedBox(height: 8),
          ScoreBar(
            score: score,
            height: 6,
          ),
        ],
      ),
    );
  }
}

/// Displays waveform visualization during recording.
class _RecordingWaveformDisplay extends ConsumerWidget {
  const _RecordingWaveformDisplay();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(recordingControllerProvider);
    final controller = ref.read(recordingControllerProvider.notifier);

    if (!state.isRecording) {
      return const SizedBox.shrink();
    }

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 24),
      child: RecordingWaveform(
        controller: controller.waveformController,
      ),
    );
  }
}
