import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../../app/di.dart';
import '../../../app/theme.dart';
import '../../example_audio/presentation/example_audio_controller.dart';
import '../../recording/presentation/recording_controller.dart';
import '../../recording/presentation/recording_state.dart';
import '../../settings/domain/app_settings.dart';
import '../../settings/presentation/settings_controller.dart';
import '../../text_sequences/domain/text_sequence.dart';
import '../../progress/domain/text_sequence_progress.dart';
import '../../stats/presentation/stats_controller.dart';
import 'home_controller.dart';
import 'home_state.dart';
import '../../text_sequences/presentation/sequence_list_controller.dart';
import 'practice_sheet.dart';
import 'widgets/activity_summary.dart';
import 'widgets/level_picker.dart';
import 'widgets/rating_nav_buttons.dart';

/// Provider for the home screen controller.
final homeControllerProvider = StateNotifierProvider<HomeController, HomeState>(
  (ref) {
    return HomeController(
      textSequenceRepository: ref.watch(textSequenceRepositoryProvider),
      progressRepository: ref.watch(progressRepositoryProvider),
      getNextByLevel: ref.watch(getNextByLevelProvider),
      settings: ref.watch(settingsControllerProvider),
    );
  },
);

/// Home screen that displays the current practice sequence.
class HomeScreen extends ConsumerWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(homeControllerProvider);
    final controller = ref.read(homeControllerProvider.notifier);

    return Scaffold(
      appBar: AppBar(
        leading: IconButton(
          icon: const Icon(Icons.list),
          onPressed: () => context.go('/list'),
        ),
        title: LevelPicker(
          onLevelChanged: (_) => ref.invalidate(homeControllerProvider),
        ),
        centerTitle: true,
        actions: [
          IconButton(
            icon: const Icon(Icons.bar_chart),
            onPressed: () => context.go('/stats'),
          ),
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: () => context.go('/settings'),
          ),
        ],
      ),
      body: state.isLoading
          ? const Center(child: CircularProgressIndicator())
          : state.current == null
          ? const _NoSentenceState()
          : _HomeContent(state: state, controller: controller),
    );
  }
}

/// Displayed when no sentence is available for the current level.
class _NoSentenceState extends StatelessWidget {
  const _NoSentenceState();

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        children: [
          const ActivitySummary(),
          const Spacer(),
          Icon(
            Icons.search_off,
            size: 64,
            color: Theme.of(context).colorScheme.outline,
          ),
          const SizedBox(height: 16),
          Text(
            'No sentences available for this level',
            style: Theme.of(context).textTheme.titleMedium?.copyWith(
              color: Theme.of(context).colorScheme.onSurfaceVariant,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 8),
          Text(
            'Try selecting a different HSK level above',
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
              color: Theme.of(context).colorScheme.outline,
            ),
            textAlign: TextAlign.center,
          ),
          const Spacer(),
        ],
      ),
    );
  }
}

class _HomeContent extends ConsumerStatefulWidget {
  const _HomeContent({required this.state, required this.controller});

  final HomeState state;
  final HomeController controller;

  @override
  ConsumerState<_HomeContent> createState() => _HomeContentState();
}

class _HomeContentState extends ConsumerState<_HomeContent> {
  // Local override for pinyin visibility (null = use settings, true/false = user override)
  bool? _pinyinOverride;
  String? _lastSequenceId;

  @override
  void didUpdateWidget(covariant _HomeContent oldWidget) {
    super.didUpdateWidget(oldWidget);
    // Reset local override when sequence changes
    if (widget.state.current?.id != _lastSequenceId) {
      _pinyinOverride = null;
      _lastSequenceId = widget.state.current?.id;
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
      builder: (context) =>
          PracticeSheet(sequence: sequence, progress: progress),
    );
  }

  void _playAudio(TextSequence sequence) {
    final voices = sequence.voices;
    if (voices == null || voices.isEmpty) return;

    final controller = ref.read(exampleAudioControllerProvider.notifier);
    final voiceId = _selectVoice(voices);
    controller.play(sequence, voiceId);
  }

  /// Selects voice based on user's VoicePreference setting.
  String _selectVoice(List<ExampleVoice> voices) {
    final settingsAsync = ref.read(settingsControllerProvider);
    final preference =
        settingsAsync.valueOrNull?.voicePreference ??
        VoicePreference.noPreference;

    if (preference == VoicePreference.noPreference) {
      return voices.first.id;
    }

    // Try to find a voice matching the preference
    final targetGender = preference == VoicePreference.male ? 'male' : 'female';
    final match = voices.firstWhere(
      (v) => v.id.toLowerCase().contains(targetGender),
      orElse: () => voices.first,
    );
    return match.id;
  }

  @override
  Widget build(BuildContext context) {
    final sequence = widget.state.current!;
    final hasPinyin =
        sequence.romanization != null && sequence.romanization!.isNotEmpty;
    final hasAudio = sequence.voices != null && sequence.voices!.isNotEmpty;
    final audioState = ref.watch(exampleAudioControllerProvider);

    // Watch settings for reactive pinyin visibility
    final settingsAsync = ref.watch(settingsControllerProvider);
    final showRomanizationSetting =
        settingsAsync.valueOrNull?.showRomanization ?? true;
    final showPinyin = _pinyinOverride ?? showRomanizationSetting;

    final recordingState = ref.watch(recordingControllerProvider);
    final canRate = recordingState.hasPlayedBack;

    return Column(
      children: [
        // Top 75% - sentence, pinyin, audio controls
        Expanded(
          flex: 3,
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 24),
            child: Column(
              children: [
                const Spacer(flex: 1),
                // Sentence and pinyin
                ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 420),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      // Sentence text
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
                      // Pinyin
                      if (hasPinyin) ...[
                        const SizedBox(height: 12),
                        GestureDetector(
                          onTap: () =>
                              setState(() => _pinyinOverride = !showPinyin),
                          child: Text(
                            showPinyin
                                ? sequence.romanization!
                                : '(tap to show pinyin)',
                            style: Theme.of(context).textTheme.titleMedium
                                ?.copyWith(
                                  color: showPinyin
                                      ? Theme.of(context).colorScheme.secondary
                                      : AppTheme.subtle,
                                  fontStyle: showPinyin
                                      ? FontStyle.normal
                                      : FontStyle.italic,
                                ),
                            textAlign: TextAlign.center,
                          ),
                        ),
                      ],
                    ],
                  ),
                ),
                const Spacer(flex: 1),
                // Audio controls (centered, between text and rating buttons)
                _AudioControlsRow(
                  sequence: sequence,
                  hasAudio: hasAudio,
                  audioState: audioState,
                  onPlayAudio: () => _playAudio(sequence),
                ),
                const Spacer(flex: 1),
              ],
            ),
          ),
        ),
        // Bottom 25% - rating buttons
        Expanded(
          flex: 1,
          child: Column(
            mainAxisAlignment: MainAxisAlignment.start,
            children: [
              RatingNavButtons(
                enabled: canRate,
                onRate: (rating) async {
                  await ref
                      .read(exampleAudioControllerProvider.notifier)
                      .stop();
                  await widget.controller.rateAndNext(rating);
                  ref
                      .read(recordingControllerProvider.notifier)
                      .resetPlaybackState();
                  // Invalidate stats so they refresh when stats screen is opened
                  ref.invalidate(statsControllerProvider);
                  // Invalidate sentence list so ratings are reflected
                  ref.invalidate(sequenceListControllerProvider);
                },
              ),
              if (!canRate)
                const Padding(
                  padding: EdgeInsets.only(top: 8),
                  child: Text(
                    'Record and replay to grade',
                    style: TextStyle(color: Colors.white, fontSize: 14),
                  ),
                ),
            ],
          ),
        ),
      ],
    );
  }
}

/// Column of audio controls: Play example (top), Record (middle), Replay (bottom).
class _AudioControlsRow extends ConsumerWidget {
  const _AudioControlsRow({
    required this.sequence,
    required this.hasAudio,
    required this.audioState,
    required this.onPlayAudio,
  });

  final TextSequence sequence;
  final bool hasAudio;
  final ExampleAudioState audioState;
  final VoidCallback onPlayAudio;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final recordingState = ref.watch(recordingControllerProvider);
    final recordingController = ref.read(recordingControllerProvider.notifier);
    final recordingStatus = recordingState.recordingStatus;

    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        // Play example audio button (top)
        if (hasAudio) ...[
          IconButton(
            onPressed: audioState.isPlaying ? null : onPlayAudio,
            icon: Icon(
              audioState.isPlaying ? Icons.volume_up : Icons.play_arrow,
            ),
            iconSize: 32,
            tooltip: 'Play example',
          ),
          const SizedBox(height: 16),
        ],

        // Record button (central, larger)
        _buildRecordButton(context, ref, recordingStatus, recordingController),

        const SizedBox(height: 16),

        // Replay button (always visible, greyed when no recording)
        _buildReplayButton(context, ref, recordingState, recordingController),
      ],
    );
  }

  Widget _buildRecordButton(
    BuildContext context,
    WidgetRef ref,
    RecordingStatus status,
    RecordingController controller,
  ) {
    final isProcessing = status == RecordingStatus.processing;
    final isRecording = status == RecordingStatus.recording;

    return FloatingActionButton(
      heroTag: 'record_fab',
      onPressed: isProcessing
          ? null
          : () async {
              debugPrint('🎤 Record button pressed. isRecording=$isRecording');
              try {
                if (isRecording) {
                  debugPrint('Stopping and saving...');
                  await controller.stopAndSave(sequence);
                  debugPrint('Save complete');
                } else {
                  debugPrint('🎤 Starting recording...');
                  await controller.startRecording(sequence);
                  final newState = ref.read(recordingControllerProvider);
                  debugPrint('🎤 Recording started. Error: ${newState.error}');
                  if (newState.error != null && context.mounted) {
                    ScaffoldMessenger.of(
                      context,
                    ).showSnackBar(SnackBar(content: Text(newState.error!)));
                  }
                }
              } catch (e, stackTrace) {
                debugPrint('🎤 ERROR: $e');
                debugPrint('🎤 Stack: $stackTrace');
                if (context.mounted) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(content: Text('Recording error: $e')),
                  );
                }
              }
            },
      child: _buildRecordIcon(status),
    );
  }

  Widget _buildRecordIcon(RecordingStatus status) {
    switch (status) {
      case RecordingStatus.idle:
        return const Icon(Icons.mic);
      case RecordingStatus.recording:
        return const Icon(Icons.stop, color: Colors.red);
      case RecordingStatus.processing:
        return const SizedBox(
          width: 24,
          height: 24,
          child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
        );
    }
  }

  Widget _buildReplayButton(
    BuildContext context,
    WidgetRef ref,
    RecordingState recordingState,
    RecordingController controller,
  ) {
    final hasRecording = recordingState.hasLatestRecording;
    final isPlaying = recordingState.isPlaying;

    return IconButton(
      onPressed: hasRecording && !isPlaying
          ? () => controller.replayLatest(sequence.id)
          : null,
      icon: const Icon(Icons.loop),
      iconSize: 32,
      tooltip: hasRecording ? 'Replay recording' : 'No recording yet',
    );
  }
}
