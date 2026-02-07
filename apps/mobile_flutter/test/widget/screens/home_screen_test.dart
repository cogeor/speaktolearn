import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:speak_to_learn/app/di.dart';
import 'package:speak_to_learn/app/theme.dart';
import 'package:speak_to_learn/features/practice/presentation/home_controller.dart';
import 'package:speak_to_learn/features/practice/presentation/home_screen.dart';
import 'package:speak_to_learn/features/practice/presentation/home_state.dart';
import 'package:speak_to_learn/features/progress/domain/text_sequence_progress.dart';
import 'package:speak_to_learn/features/text_sequences/domain/text_sequence.dart';

import '../../mocks/mock_audio.dart';
import '../../mocks/mock_repositories.dart';

/// Fake HomeController for testing.
class FakeHomeController extends StateNotifier<HomeState>
    implements HomeController {
  FakeHomeController([HomeState? initialState])
      : super(initialState ?? const HomeState());

  bool nextCalled = false;
  bool toggleTrackedCalled = false;
  String? setCurrentSequenceId;

  @override
  Future<void> next() async {
    nextCalled = true;
  }

  @override
  Future<void> toggleTracked() async {
    toggleTrackedCalled = true;
  }

  @override
  Future<void> setCurrentSequence(String id) async {
    setCurrentSequenceId = id;
  }

  @override
  Future<void> refreshProgress() async {}

  void updateState(HomeState newState) {
    state = newState;
  }
}


void main() {
  const testSequence = TextSequence(
    id: 'test-001',
    text: '你好',
    language: 'zh',
  );

  const testProgress = TextSequenceProgress(
    tracked: true,
    bestScore: 85,
  );

  Widget buildTestWidget({
    required HomeState initialState,
    FakeHomeController? controller,
  }) {
    final fakeController = controller ?? FakeHomeController(initialState);

    return ProviderScope(
      overrides: [
        homeControllerProvider.overrideWith((_) => fakeController),
        // Override all base dependencies to avoid Hive
        textSequenceRepositoryProvider.overrideWithValue(MockTextSequenceRepository()),
        progressRepositoryProvider.overrideWithValue(MockProgressRepository()),
        recordingRepositoryProvider.overrideWithValue(MockRecordingRepository()),
        settingsRepositoryProvider.overrideWithValue(MockSettingsRepository()),
        exampleAudioRepositoryProvider.overrideWithValue(MockExampleAudioRepository()),
        audioRecorderProvider.overrideWithValue(FakeAudioRecorder()),
        audioPlayerProvider.overrideWithValue(FakeAudioPlayer()),
      ],
      child: MaterialApp(
        theme: AppTheme.darkTheme,
        home: const HomeScreen(),
      ),
    );
  }

  group('HomeScreen empty state', () {
    testWidgets('shows loading indicator when loading', (tester) async {
      await tester.pumpWidget(
        buildTestWidget(
          initialState: const HomeState(isLoading: true),
        ),
      );

      expect(find.byType(CircularProgressIndicator), findsOneWidget);
    });

    testWidgets('shows empty state when no tracked sequences', (tester) async {
      await tester.pumpWidget(
        buildTestWidget(
          initialState: const HomeState(isLoading: false, isEmptyTracked: true),
        ),
      );

      expect(find.text('No tracked sequences'), findsOneWidget);
      expect(find.text('Open list'), findsOneWidget);
    });
  });

  group('HomeScreen content', () {
    testWidgets('shows current sequence text', (tester) async {
      await tester.pumpWidget(
        buildTestWidget(
          initialState: const HomeState(
            isLoading: false,
            isEmptyTracked: false,
            current: testSequence,
          ),
        ),
      );

      expect(find.text('你好'), findsOneWidget);
    });

    testWidgets('shows best score when available', (tester) async {
      await tester.pumpWidget(
        buildTestWidget(
          initialState: const HomeState(
            isLoading: false,
            isEmptyTracked: false,
            current: testSequence,
            currentProgress: testProgress,
          ),
        ),
      );

      expect(find.text('Best: 85'), findsOneWidget);
    });

    testWidgets('shows Next button', (tester) async {
      await tester.pumpWidget(
        buildTestWidget(
          initialState: const HomeState(
            isLoading: false,
            isEmptyTracked: false,
            current: testSequence,
          ),
        ),
      );

      expect(find.text('Next'), findsOneWidget);
    });

    testWidgets('next button calls controller.next', (tester) async {
      final controller = FakeHomeController(
        const HomeState(
          isLoading: false,
          isEmptyTracked: false,
          current: testSequence,
        ),
      );

      await tester.pumpWidget(
        buildTestWidget(
          initialState: const HomeState(
            isLoading: false,
            isEmptyTracked: false,
            current: testSequence,
          ),
          controller: controller,
        ),
      );

      await tester.tap(find.text('Next'));
      await tester.pump();

      expect(controller.nextCalled, isTrue);
    });

    testWidgets('track icon toggles on tap', (tester) async {
      final controller = FakeHomeController(
        const HomeState(
          isLoading: false,
          isEmptyTracked: false,
          current: testSequence,
          currentProgress: testProgress,
        ),
      );

      await tester.pumpWidget(
        buildTestWidget(
          initialState: const HomeState(
            isLoading: false,
            isEmptyTracked: false,
            current: testSequence,
            currentProgress: testProgress,
          ),
          controller: controller,
        ),
      );

      // Find the bookmark icon button in the AppBar
      final bookmarkButton = find.byIcon(Icons.bookmark);
      expect(bookmarkButton, findsOneWidget);

      await tester.tap(bookmarkButton);
      await tester.pump();

      expect(controller.toggleTrackedCalled, isTrue);
    });

    testWidgets('shows untracked icon when not tracked', (tester) async {
      await tester.pumpWidget(
        buildTestWidget(
          initialState: const HomeState(
            isLoading: false,
            isEmptyTracked: false,
            current: testSequence,
            currentProgress: TextSequenceProgress(tracked: false),
          ),
        ),
      );

      // Should show outline bookmark when not tracked
      expect(find.byIcon(Icons.bookmark_border), findsOneWidget);
    });
  });

  group('HomeScreen app bar', () {
    testWidgets('shows list icon button', (tester) async {
      await tester.pumpWidget(
        buildTestWidget(
          initialState: const HomeState(isLoading: false, isEmptyTracked: true),
        ),
      );

      expect(find.byIcon(Icons.list), findsOneWidget);
    });

    testWidgets('has Home title', (tester) async {
      await tester.pumpWidget(
        buildTestWidget(
          initialState: const HomeState(isLoading: false, isEmptyTracked: true),
        ),
      );

      expect(find.text('Home'), findsOneWidget);
    });
  });
}
