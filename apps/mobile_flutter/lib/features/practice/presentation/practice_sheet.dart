import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../app/di.dart';
import '../../../app/theme.dart';
import '../../text_sequences/domain/text_sequence.dart';
import '../../progress/domain/sentence_rating.dart';
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
  const PracticeSheet({super.key, required this.sequence, this.progress});

  final TextSequence sequence;
  final TextSequenceProgress? progress;

  @override
  ConsumerState<PracticeSheet> createState() => _PracticeSheetState();
}

class _PracticeSheetState extends ConsumerState<PracticeSheet> {
  @override
  void initState() {
    super.initState();
    // Check if a recording exists for this sequence on sheet open
    WidgetsBinding.instance.addPostFrameCallback((_) {
      ref
          .read(recordingControllerProvider.notifier)
          .checkLatestRecording(widget.sequence.id);
    });
  }

  @override
  void dispose() {
    // Cancel any active recording or playback when sheet is dismissed
    ref.read(recordingControllerProvider.notifier).cancel();
    super.dispose();
  }

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
            _ScoreDisplay(progress: widget.progress, sequence: widget.sequence),
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
    if (state.isSaving) {
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
      onPressed: state.isSaving
          ? null
          : () {
              if (state.isRecording) {
                controller.stopAndSave(sequence);
              } else {
                controller.startRecording(sequence);
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

/// Displays attempt count and last rating.
/// Note: Detailed breakdown removed as scoring moved to self-report.
class _ScoreDisplay extends StatelessWidget {
  const _ScoreDisplay({required this.progress, required this.sequence});

  final TextSequenceProgress? progress;
  final TextSequence sequence;

  @override
  Widget build(BuildContext context) {
    final lastRating = progress?.lastRating;
    final ratingText = lastRating?.label ?? '-';

    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        _ScoreLabel(label: 'Attempts', score: progress?.attemptCount),
        const SizedBox(width: 24),
        SizedBox(
          width: 80,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text('Latest', style: Theme.of(context).textTheme.bodySmall),
              const SizedBox(height: 4),
              Text(
                ratingText,
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: lastRating?.color ?? AppTheme.subtle,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

/// Helper widget to display a score label with visual bar.
class _ScoreLabel extends StatelessWidget {
  const _ScoreLabel({required this.label, required this.score});

  final String label;
  final int? score;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 80,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(label, style: Theme.of(context).textTheme.bodySmall),
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
          ScoreBar(score: score, height: 6),
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
      child: RecordingWaveform(controller: controller.waveformController),
    );
  }
}
