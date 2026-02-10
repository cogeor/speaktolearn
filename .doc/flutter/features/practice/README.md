# features/practice/ Module

## Purpose

Orchestrates the main practice flow on the Home screen. This is the primary UI module that ties together all other features: selection, recording, scoring, and playback.

## Folder Structure

```
practice/
└── presentation/
    ├── home_screen.dart         # Main trainer screen
    ├── home_controller.dart     # State management
    └── practice_sheet.dart      # Bottom sheet for practice controls
```

**Note**: No `domain/` or `data/` layers. This module only has presentation because it orchestrates other modules' domain logic.

---

## Presentation Layer

### `home_screen.dart`

**Purpose**: The main trainer screen showing one text sequence.

**Layout**:
```
┌────────────────────────────────────────┐
│ [List]                         [Track] │  <- AppBar with icons
├────────────────────────────────────────┤
│                                        │
│                                        │
│              我想喝水。                 │  <- Large centered text
│                                        │    (tap to open sheet)
│                                        │
├────────────────────────────────────────┤
│              [ Next ]                  │  <- Primary button
└────────────────────────────────────────┘
```

**Implementation**:

```dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'home_controller.dart';
import 'practice_sheet.dart';

class HomeScreen extends ConsumerWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(homeControllerProvider);
    final controller = ref.read(homeControllerProvider.notifier);
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        leading: IconButton(
          icon: const Icon(Icons.list),
          tooltip: 'Browse sequences',
          onPressed: () => context.push('/list'),
        ),
        actions: [
          if (state.current != null)
            IconButton(
              icon: Icon(
                state.currentProgress?.tracked == true
                    ? Icons.star
                    : Icons.star_border,
                color: state.currentProgress?.tracked == true
                    ? Colors.amber
                    : null,
              ),
              tooltip: state.currentProgress?.tracked == true
                  ? 'Untrack'
                  : 'Track',
              onPressed: controller.toggleTracked,
            ),
        ],
      ),
      body: _buildBody(context, state, controller, theme),
    );
  }

  Widget _buildBody(
    BuildContext context,
    HomeState state,
    HomeController controller,
    ThemeData theme,
  ) {
    if (state.isLoading) {
      return const Center(child: CircularProgressIndicator());
    }

    if (state.isEmptyTracked) {
      return _EmptyState(
        onOpenList: () => context.push('/list'),
      );
    }

    if (state.current == null) {
      return const Center(child: Text('No sequence selected'));
    }

    return Column(
      children: [
        // Main content area - tappable text
        Expanded(
          child: GestureDetector(
            onTap: () => _showPracticeSheet(context, state),
            child: Center(
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Text(
                  state.current!.text,
                  style: theme.textTheme.displayLarge,
                  textAlign: TextAlign.center,
                ),
              ),
            ),
          ),
        ),

        // Score display (if available)
        if (state.currentProgress?.bestScore != null)
          Padding(
            padding: const EdgeInsets.only(bottom: 16),
            child: Text(
              'Best: ${state.currentProgress!.bestScore!.toStringAsFixed(0)}',
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurface.withOpacity(0.6),
              ),
            ),
          ),

        // Next button
        Padding(
          padding: const EdgeInsets.all(24),
          child: SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              onPressed: controller.next,
              child: const Text('Next'),
            ),
          ),
        ),
      ],
    );
  }

  void _showPracticeSheet(BuildContext context, HomeState state) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      builder: (_) => PracticeSheet(
        textSequence: state.current!,
        progress: state.currentProgress,
      ),
    );
  }
}

class _EmptyState extends StatelessWidget {
  final VoidCallback onOpenList;

  const _EmptyState({required this.onOpenList});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            'No tracked sequences',
            style: theme.textTheme.titleMedium?.copyWith(
              color: theme.colorScheme.onSurface.withOpacity(0.6),
            ),
          ),
          const SizedBox(height: 16),
          OutlinedButton(
            onPressed: onOpenList,
            child: const Text('Open list'),
          ),
        ],
      ),
    );
  }
}
```

---

### `home_controller.dart`

**Purpose**: State management for the home screen.

**Implementation**:

```dart
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:freezed_annotation/freezed_annotation.dart';
import '../text_sequences/domain/text_sequence.dart';
import '../text_sequences/domain/text_sequence_repository.dart';
import '../progress/domain/text_sequence_progress.dart';
import '../progress/domain/progress_repository.dart';
import '../selection/domain/get_next_tracked.dart';

part 'home_controller.freezed.dart';

@freezed
class HomeState with _$HomeState {
  const factory HomeState({
    /// Currently displayed text sequence.
    TextSequence? current,

    /// Progress for the current sequence.
    TextSequenceProgress? currentProgress,

    /// True while loading initial data.
    @Default(true) bool isLoading,

    /// True if user has no tracked sequences.
    @Default(false) bool isEmptyTracked,
  }) = _HomeState;
}

final homeControllerProvider =
    StateNotifierProvider<HomeController, HomeState>(
  (ref) => HomeController(
    textSequences: ref.watch(textSequenceRepositoryProvider),
    progress: ref.watch(progressRepositoryProvider),
    getNextTracked: ref.watch(getNextTrackedSequenceProvider),
  ),
);

class HomeController extends StateNotifier<HomeState> {
  final TextSequenceRepository _textSequences;
  final ProgressRepository _progress;
  final GetNextTrackedSequence _getNextTracked;

  HomeController({
    required TextSequenceRepository textSequences,
    required ProgressRepository progress,
    required GetNextTrackedSequence getNextTracked,
  })  : _textSequences = textSequences,
        _progress = progress,
        _getNextTracked = getNextTracked,
        super(const HomeState()) {
    _init();
  }

  Future<void> _init() async {
    state = state.copyWith(isLoading: true);

    try {
      final next = await _getNextTracked();

      if (next == null) {
        state = state.copyWith(
          isLoading: false,
          isEmptyTracked: true,
        );
        return;
      }

      final progress = await _progress.getProgress(next.id);

      state = state.copyWith(
        current: next,
        currentProgress: progress,
        isLoading: false,
        isEmptyTracked: false,
      );
    } catch (e) {
      state = state.copyWith(isLoading: false);
      // Handle error - could expose error state
    }
  }

  /// Loads the next tracked sequence.
  Future<void> next() async {
    final currentId = state.current?.id;
    final next = await _getNextTracked(currentId: currentId);

    if (next == null) {
      state = state.copyWith(isEmptyTracked: true);
      return;
    }

    final progress = await _progress.getProgress(next.id);

    state = state.copyWith(
      current: next,
      currentProgress: progress,
      isEmptyTracked: false,
    );
  }

  /// Sets a specific sequence as current (from list selection).
  Future<void> setCurrentSequence(String id) async {
    final sequence = await _textSequences.getById(id);
    if (sequence == null) return;

    final progress = await _progress.getProgress(id);

    state = state.copyWith(
      current: sequence,
      currentProgress: progress,
      isEmptyTracked: false,
    );
  }

  /// Toggles tracked status for current sequence.
  Future<void> toggleTracked() async {
    final currentId = state.current?.id;
    if (currentId == null) return;

    await _progress.toggleTracked(currentId);

    // Refresh current progress
    final progress = await _progress.getProgress(currentId);
    state = state.copyWith(currentProgress: progress);

    // If we just untracked the only tracked item, load next or show empty
    if (progress?.tracked == false) {
      final trackedIds = await _progress.getTrackedIds();
      if (trackedIds.isEmpty) {
        state = state.copyWith(isEmptyTracked: true);
      }
    }
  }

  /// Called after a scoring attempt to refresh progress.
  Future<void> refreshProgress() async {
    final currentId = state.current?.id;
    if (currentId == null) return;

    final progress = await _progress.getProgress(currentId);
    state = state.copyWith(currentProgress: progress);
  }
}
```

---

### `practice_sheet.dart`

**Purpose**: Bottom sheet with practice controls (play, record, score).

**Layout**:
```
┌────────────────────────────────────────┐
│              我想喝水。                 │  <- Text (smaller)
├────────────────────────────────────────┤
│         [ Male ]  [ Female ]           │  <- Example audio buttons
├────────────────────────────────────────┤
│              [ Record ]                │  <- Record button
│           [ Replay Latest ]            │  <- Replay (if exists)
├────────────────────────────────────────┤
│    Latest: 72      Best: 85            │  <- Scores
└────────────────────────────────────────┘
```

**Implementation**:

```dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../text_sequences/domain/text_sequence.dart';
import '../progress/domain/text_sequence_progress.dart';
import '../example_audio/presentation/example_audio_controller.dart';
import '../recording/presentation/recording_controller.dart';
import '../scoring/presentation/scoring_controller.dart';

class PracticeSheet extends ConsumerStatefulWidget {
  final TextSequence textSequence;
  final TextSequenceProgress? progress;

  const PracticeSheet({
    super.key,
    required this.textSequence,
    this.progress,
  });

  @override
  ConsumerState<PracticeSheet> createState() => _PracticeSheetState();
}

class _PracticeSheetState extends ConsumerState<PracticeSheet> {
  double? _latestScore;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final exampleAudio = ref.watch(exampleAudioControllerProvider);
    final recording = ref.watch(recordingControllerProvider);

    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // Drag handle
            Container(
              width: 40,
              height: 4,
              decoration: BoxDecoration(
                color: theme.colorScheme.onSurface.withOpacity(0.3),
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            const SizedBox(height: 24),

            // Text (smaller than main screen)
            Text(
              widget.textSequence.text,
              style: theme.textTheme.titleLarge,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 24),

            // Example audio buttons
            _ExampleAudioButtons(
              voices: widget.textSequence.exampleAudio,
              onPlay: (voiceId) => ref
                  .read(exampleAudioControllerProvider.notifier)
                  .play(widget.textSequence.id, voiceId),
              isPlaying: exampleAudio.isPlaying,
              playingVoiceId: exampleAudio.currentVoiceId,
            ),
            const SizedBox(height: 24),

            // Record button
            _RecordButton(
              isRecording: recording.isRecording,
              isScoring: recording.isScoring,
              onTap: () => _handleRecordTap(recording),
            ),
            const SizedBox(height: 16),

            // Replay button (if has recording)
            if (recording.hasLatestRecording)
              OutlinedButton.icon(
                onPressed: recording.isPlaying
                    ? null
                    : () => ref
                        .read(recordingControllerProvider.notifier)
                        .replayLatest(widget.textSequence.id),
                icon: Icon(recording.isPlaying ? Icons.stop : Icons.replay),
                label: const Text('Replay'),
              ),
            const SizedBox(height: 24),

            // Scores
            _ScoreDisplay(
              latestScore: _latestScore,
              bestScore: widget.progress?.bestScore,
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _handleRecordTap(RecordingState recording) async {
    final controller = ref.read(recordingControllerProvider.notifier);

    if (recording.isRecording) {
      // Stop and score
      final result = await controller.stopAndScore(widget.textSequence);
      if (result != null) {
        setState(() => _latestScore = result.score);
        // Refresh home controller to update best score
        ref.read(homeControllerProvider.notifier).refreshProgress();
      }
    } else {
      // Start recording
      await controller.startRecording(widget.textSequence.id);
    }
  }
}

class _ExampleAudioButtons extends StatelessWidget {
  final List<ExampleVoice> voices;
  final void Function(String voiceId) onPlay;
  final bool isPlaying;
  final String? playingVoiceId;

  const _ExampleAudioButtons({
    required this.voices,
    required this.onPlay,
    required this.isPlaying,
    this.playingVoiceId,
  });

  @override
  Widget build(BuildContext context) {
    if (voices.isEmpty) {
      return const Text('No example audio available');
    }

    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: voices.map((voice) {
        final isThisPlaying = isPlaying && playingVoiceId == voice.id;
        final label = voice.label['en'] ?? voice.id;

        return Padding(
          padding: const EdgeInsets.symmetric(horizontal: 8),
          child: OutlinedButton.icon(
            onPressed: isPlaying ? null : () => onPlay(voice.id),
            icon: Icon(isThisPlaying ? Icons.stop : Icons.play_arrow),
            label: Text(label),
          ),
        );
      }).toList(),
    );
  }
}

class _RecordButton extends StatelessWidget {
  final bool isRecording;
  final bool isScoring;
  final VoidCallback onTap;

  const _RecordButton({
    required this.isRecording,
    required this.isScoring,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    if (isScoring) {
      return const CircularProgressIndicator();
    }

    return ElevatedButton.icon(
      onPressed: onTap,
      style: ElevatedButton.styleFrom(
        backgroundColor: isRecording ? Colors.red : null,
        minimumSize: const Size(160, 56),
      ),
      icon: Icon(isRecording ? Icons.stop : Icons.mic),
      label: Text(isRecording ? 'Stop' : 'Record'),
    );
  }
}

class _ScoreDisplay extends StatelessWidget {
  final double? latestScore;
  final double? bestScore;

  const _ScoreDisplay({
    this.latestScore,
    this.bestScore,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        if (latestScore != null) ...[
          _ScoreLabel(
            label: 'Latest',
            score: latestScore!,
            theme: theme,
          ),
          const SizedBox(width: 32),
        ],
        if (bestScore != null)
          _ScoreLabel(
            label: 'Best',
            score: bestScore!,
            theme: theme,
          ),
      ],
    );
  }
}

class _ScoreLabel extends StatelessWidget {
  final String label;
  final double score;
  final ThemeData theme;

  const _ScoreLabel({
    required this.label,
    required this.score,
    required this.theme,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Text(
          label,
          style: theme.textTheme.bodySmall?.copyWith(
            color: theme.colorScheme.onSurface.withOpacity(0.6),
          ),
        ),
        Text(
          score.toStringAsFixed(0),
          style: theme.textTheme.headlineMedium?.copyWith(
            color: _scoreColor(score),
          ),
        ),
      ],
    );
  }

  Color _scoreColor(double score) {
    if (score >= 80) return Colors.green;
    if (score >= 50) return Colors.amber;
    return Colors.red;
  }
}
```

---

## Integration Tests

### Home Screen Tests

```dart
void main() {
  group('HomeScreen', () {
    testWidgets('shows empty state when no tracked sequences', (tester) async {
      await tester.pumpWidget(
        ProviderScope(
          overrides: [
            homeControllerProvider.overrideWith(
              () => MockHomeController(HomeState(isEmptyTracked: true, isLoading: false)),
            ),
          ],
          child: MaterialApp(home: HomeScreen()),
        ),
      );

      expect(find.text('No tracked sequences'), findsOneWidget);
      expect(find.text('Open list'), findsOneWidget);
    });

    testWidgets('shows current sequence text', (tester) async {
      final sequence = TextSequence(id: '1', text: '你好', language: 'zh');
      await tester.pumpWidget(
        ProviderScope(
          overrides: [
            homeControllerProvider.overrideWith(
              () => MockHomeController(HomeState(
                current: sequence,
                isLoading: false,
              )),
            ),
          ],
          child: MaterialApp(home: HomeScreen()),
        ),
      );

      expect(find.text('你好'), findsOneWidget);
    });

    testWidgets('shows best score when available', (tester) async {
      final sequence = TextSequence(id: '1', text: '你好', language: 'zh');
      final progress = TextSequenceProgress(
        textSequenceId: '1',
        bestScore: 85.0,
        updatedAt: DateTime.now(),
      );

      await tester.pumpWidget(
        ProviderScope(
          overrides: [
            homeControllerProvider.overrideWith(
              () => MockHomeController(HomeState(
                current: sequence,
                currentProgress: progress,
                isLoading: false,
              )),
            ),
          ],
          child: MaterialApp(home: HomeScreen()),
        ),
      );

      expect(find.text('Best: 85'), findsOneWidget);
    });

    testWidgets('next button advances to next sequence', (tester) async {
      final controller = MockHomeController(HomeState(
        current: TextSequence(id: '1', text: '你好', language: 'zh'),
        isLoading: false,
      ));

      await tester.pumpWidget(
        ProviderScope(
          overrides: [
            homeControllerProvider.overrideWith(() => controller),
          ],
          child: MaterialApp(home: HomeScreen()),
        ),
      );

      await tester.tap(find.text('Next'));
      expect(controller.nextCalled, isTrue);
    });

    testWidgets('track toggle updates state', (tester) async {
      final controller = MockHomeController(HomeState(
        current: TextSequence(id: '1', text: '你好', language: 'zh'),
        currentProgress: TextSequenceProgress(
          textSequenceId: '1',
          tracked: false,
          updatedAt: DateTime.now(),
        ),
        isLoading: false,
      ));

      await tester.pumpWidget(
        ProviderScope(
          overrides: [
            homeControllerProvider.overrideWith(() => controller),
          ],
          child: MaterialApp(home: HomeScreen()),
        ),
      );

      await tester.tap(find.byIcon(Icons.star_border));
      expect(controller.toggleTrackedCalled, isTrue);
    });

    testWidgets('tapping text opens practice sheet', (tester) async {
      await tester.pumpWidget(
        ProviderScope(
          overrides: [
            homeControllerProvider.overrideWith(
              () => MockHomeController(HomeState(
                current: TextSequence(id: '1', text: '你好', language: 'zh'),
                isLoading: false,
              )),
            ),
          ],
          child: MaterialApp(home: HomeScreen()),
        ),
      );

      await tester.tap(find.text('你好'));
      await tester.pumpAndSettle();

      expect(find.byType(PracticeSheet), findsOneWidget);
    });
  });
}
```

### Practice Flow Integration Test

```dart
void main() {
  group('Practice flow', () {
    testWidgets('complete recording and scoring flow', (tester) async {
      // This test requires mocked recording and scoring services
      // that simulate the full flow

      await tester.pumpWidget(
        ProviderScope(
          overrides: testOverridesWithMockedAudio,
          child: const SpeakToLearnApp(),
        ),
      );
      await tester.pumpAndSettle();

      // Tap sequence to open practice sheet
      await tester.tap(find.text('我想喝水。'));
      await tester.pumpAndSettle();

      // Verify practice sheet is open
      expect(find.byType(PracticeSheet), findsOneWidget);

      // Start recording
      await tester.tap(find.text('Record'));
      await tester.pump();

      // Verify recording state
      expect(find.text('Stop'), findsOneWidget);

      // Stop recording (triggers scoring)
      await tester.tap(find.text('Stop'));
      await tester.pumpAndSettle();

      // Verify score appears
      expect(find.textContaining('Latest'), findsOneWidget);
    });
  });
}
```

---

## Notes

### Why No Domain Layer?

Practice orchestrates other modules but doesn't have unique domain logic:
- Selection logic → `selection/`
- Recording logic → `recording/`
- Scoring logic → `scoring/`
- Progress storage → `progress/`

The controller is the orchestration point.

### Sheet vs. Screen

Using a bottom sheet instead of a separate screen:
1. **Minimal UI**: Only 2 screens as per spec
2. **Context preservation**: User sees the sentence while practicing
3. **Quick dismissal**: Swipe down to close
4. **Focus**: No navigation animation interruption

### State Management

The practice sheet maintains local state (`_latestScore`) for immediate feedback:
- Riverpod state for async operations (recording, playback)
- Local state for session-specific display
- Progress refresh triggers home controller update
