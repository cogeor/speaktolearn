import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:speak_to_learn/app/di.dart';
import 'package:speak_to_learn/app/theme.dart';
import 'package:speak_to_learn/features/practice/presentation/home_controller.dart';
import 'package:speak_to_learn/features/practice/presentation/home_screen.dart';
import 'package:speak_to_learn/features/practice/presentation/home_state.dart';
import 'package:speak_to_learn/features/progress/domain/sentence_rating.dart';
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
  String? setCurrentSequenceId;
  SentenceRating? lastRating;

  @override
  Future<void> next() async {
    nextCalled = true;
  }

  @override
  Future<void> setCurrentSequence(String id) async {
    setCurrentSequenceId = id;
  }

  @override
  Future<void> refreshProgress() async {}

  @override
  void clearQueue() {}

  @override
  Future<void> rateAndNext(SentenceRating rating) async {
    lastRating = rating;
    nextCalled = true;
  }

  void updateState(HomeState newState) {
    state = newState;
  }
}

void main() {
  const testSequence = TextSequence(id: 'test-001', text: '你好', language: 'zh');

  const testProgress = TextSequenceProgress(tracked: true, lastRating: SentenceRating.good);

  Widget buildTestWidget({
    required HomeState initialState,
    FakeHomeController? controller,
  }) {
    final fakeController = controller ?? FakeHomeController(initialState);

    return ProviderScope(
      overrides: [
        homeControllerProvider.overrideWith((_) => fakeController),
        // Override all base dependencies to avoid Hive
        textSequenceRepositoryProvider.overrideWithValue(
          MockTextSequenceRepository(),
        ),
        progressRepositoryProvider.overrideWithValue(MockProgressRepository()),
        recordingRepositoryProvider.overrideWithValue(
          MockRecordingRepository(),
        ),
        settingsRepositoryProvider.overrideWithValue(MockSettingsRepository()),
        exampleAudioRepositoryProvider.overrideWithValue(
          MockExampleAudioRepository(),
        ),
        audioRecorderProvider.overrideWithValue(FakeAudioRecorder()),
        audioPlayerProvider.overrideWithValue(FakeAudioPlayer()),
      ],
      child: MaterialApp(theme: AppTheme.darkTheme, home: const HomeScreen()),
    );
  }

  group('HomeScreen empty state', () {
    testWidgets('shows loading indicator when loading', (tester) async {
      await tester.pumpWidget(
        buildTestWidget(initialState: const HomeState(isLoading: true)),
      );

      expect(find.byType(CircularProgressIndicator), findsOneWidget);
    });
  });

  group('HomeScreen content', () {
    testWidgets('shows current sequence text', (tester) async {
      await tester.pumpWidget(
        buildTestWidget(
          initialState: const HomeState(
            isLoading: false,
            current: testSequence,
          ),
        ),
      );

      expect(find.text('你好'), findsOneWidget);
    });

    testWidgets('renders with progress data', (tester) async {
      await tester.pumpWidget(
        buildTestWidget(
          initialState: const HomeState(
            isLoading: false,
            current: testSequence,
            currentProgress: testProgress,
          ),
        ),
      );

      // The main sequence text should be visible
      expect(find.text('你好'), findsOneWidget);
    });

    testWidgets('shows record button', (tester) async {
      await tester.pumpWidget(
        buildTestWidget(
          initialState: const HomeState(
            isLoading: false,
            current: testSequence,
          ),
        ),
      );

      // Record button (mic icon) should be visible
      expect(find.byIcon(Icons.mic), findsOneWidget);
    });
  });

  group('HomeScreen app bar', () {
    testWidgets('shows list icon button', (tester) async {
      await tester.pumpWidget(
        buildTestWidget(
          initialState: const HomeState(
            isLoading: false,
            current: testSequence,
          ),
        ),
      );

      expect(find.byIcon(Icons.list), findsOneWidget);
    });
  });
}
