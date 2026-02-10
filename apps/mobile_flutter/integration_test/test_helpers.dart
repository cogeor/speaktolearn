// integration_test/test_helpers.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:speak_to_learn/app/di.dart';
import 'package:speak_to_learn/app/router.dart';
import 'package:speak_to_learn/app/theme.dart';
import 'package:speak_to_learn/features/practice/presentation/practice_sheet.dart';
import 'package:speak_to_learn/features/text_sequences/domain/text_sequence.dart';

import 'mocks/mock_integration_repositories.dart';

/// Test data for integration tests.
class TestData {
  static const sequences = [
    TextSequence(
      id: 'test-001',
      text: '你好',
      romanization: 'ni hao',
      language: 'zh',
      tags: ['greeting'],
      difficulty: 1,
      voices: [
        ExampleVoice(id: 'male', uri: 'assets://examples/male/test-001.m4a'),
        ExampleVoice(
          id: 'female',
          uri: 'assets://examples/female/test-001.m4a',
        ),
      ],
    ),
    TextSequence(
      id: 'test-002',
      text: '谢谢',
      romanization: 'xie xie',
      language: 'zh',
      tags: ['courtesy', 'basic'],
      difficulty: 1,
      voices: [
        ExampleVoice(id: 'male', uri: 'assets://examples/male/test-002.m4a'),
      ],
    ),
    TextSequence(
      id: 'test-003',
      text: '再见',
      romanization: 'zai jian',
      language: 'zh',
      tags: ['farewell', 'basic'],
      difficulty: 1,
    ),
  ];
}

/// Common widget finders for integration tests.
class IntegrationFinders {
  // App bar elements
  static Finder get listButton => find.byIcon(Icons.list);
  static Finder get backButton => find.byIcon(Icons.arrow_back);
  static Finder get bookmarkFilledIcon => find.byIcon(Icons.bookmark);
  static Finder get bookmarkOutlineIcon => find.byIcon(Icons.bookmark_border);

  // Home screen elements
  static Finder get homeTitle => find.text('Home');
  static Finder get nextButton => find.text('Next');
  static Finder get openListButton => find.text('Open list');
  static Finder get emptyStateText => find.text('No tracked sequences');

  // List screen elements
  static Finder get listTitle => find.text('Sequences');
  static Finder get starIcon => find.byIcon(Icons.star);
  static Finder get starBorderIcon => find.byIcon(Icons.star_border);

  // List item tracking icons
  static Finder get trackedStarIcon => find.byIcon(Icons.star);
  static Finder get untrackedStarIcon => find.byIcon(Icons.star_border);

  // Recording elements
  static Finder get fabButton => find.byType(FloatingActionButton);
  static Finder get micIcon => find.byIcon(Icons.mic);
  static Finder get stopIcon => find.byIcon(Icons.stop);
  static Finder get progressIndicator => find.byType(CircularProgressIndicator);

  // Practice sheet elements
  static Finder get practiceSheet => find.byType(PracticeSheet);
  static Finder get recordFab => find.byType(FloatingActionButton);
  static Finder get replayButton => find.byIcon(Icons.replay);

  // Score elements
  static Finder get latestScoreLabel => find.text('Latest');
  static Finder get bestScoreLabel => find.text('Best');
  static Finder scoreValue(int score) => find.text(score.toString());

  // Helper to find sequence text
  static Finder sequenceText(String text) => find.text(text);

  /// Find a ListTile containing the given text.
  static Finder listTileWithText(String text) {
    return find.ancestor(of: find.text(text), matching: find.byType(ListTile));
  }

  /// Find the star button within a list tile for a specific sequence.
  static Finder starButtonForSequence(String text) {
    return find.descendant(
      of: listTileWithText(text),
      matching: find.byType(IconButton),
    );
  }
}

/// Creates provider overrides for integration tests.
///
/// Uses in-memory mock implementations that don't require Hive.
List<Override> createIntegrationTestOverrides({
  List<TextSequence>? sequences,
  Set<String>? trackedIds,
}) {
  final mockTextSequenceRepo = MockIntegrationTextSequenceRepository(
    sequences ?? TestData.sequences,
  );
  final mockProgressRepo = MockIntegrationProgressRepository(
    trackedIds: trackedIds,
  );
  final mockRecordingRepo = MockIntegrationRecordingRepository();
  final mockExampleAudioRepo = MockIntegrationExampleAudioRepository();
  final mockSettingsRepo = MockIntegrationSettingsRepository();
  final mockRecorder = MockIntegrationAudioRecorder();
  final mockPlayer = MockIntegrationAudioPlayer();

  return [
    textSequenceRepositoryProvider.overrideWithValue(mockTextSequenceRepo),
    progressRepositoryProvider.overrideWithValue(mockProgressRepo),
    recordingRepositoryProvider.overrideWithValue(mockRecordingRepo),
    exampleAudioRepositoryProvider.overrideWithValue(mockExampleAudioRepo),
    settingsRepositoryProvider.overrideWithValue(mockSettingsRepo),
    audioRecorderProvider.overrideWithValue(mockRecorder),
    audioPlayerProvider.overrideWithValue(mockPlayer),
  ];
}

/// Integration test app widget with router support.
class IntegrationTestApp extends ConsumerWidget {
  const IntegrationTestApp({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final router = ref.watch(routerProvider);

    return MaterialApp.router(
      theme: AppTheme.darkTheme,
      routerConfig: router,
      debugShowCheckedModeBanner: false,
    );
  }
}

/// Extension for common integration test operations.
extension IntegrationTestHelpers on WidgetTester {
  /// Pumps the integration test app with overrides and settles.
  Future<void> pumpIntegrationApp({
    List<TextSequence>? sequences,
    Set<String>? trackedIds,
  }) async {
    await pumpWidget(
      ProviderScope(
        overrides: createIntegrationTestOverrides(
          sequences: sequences,
          trackedIds: trackedIds,
        ),
        child: const IntegrationTestApp(),
      ),
    );
    await pumpAndSettle();
  }

  /// Navigates to the list screen via the list button.
  Future<void> navigateToListScreen() async {
    await tap(IntegrationFinders.listButton);
    await pumpAndSettle();
  }

  /// Navigates back to home screen via back button.
  Future<void> navigateBackToHome() async {
    await tap(IntegrationFinders.backButton);
    await pumpAndSettle();
  }

  /// Taps a sequence in the list by its text.
  Future<void> tapSequence(String text) async {
    await tap(find.text(text));
    await pumpAndSettle();
  }

  /// Taps the star/track button for a sequence in the list.
  Future<void> tapTrackButton() async {
    final starBorder = IntegrationFinders.starBorderIcon;
    if (starBorder.evaluate().isNotEmpty) {
      await tap(starBorder.first);
    } else {
      await tap(IntegrationFinders.starIcon.first);
    }
    await pumpAndSettle();
  }

  /// Taps the star/track button for a specific sequence in the list.
  Future<void> tapTrackButtonForSequence(String text) async {
    final listTile = IntegrationFinders.listTileWithText(text);
    expect(listTile, findsOneWidget);

    // Find the star icon button within this tile
    final starButton = find.descendant(
      of: listTile,
      matching: find.byWidgetPredicate(
        (widget) =>
            widget is IconButton &&
            (widget.icon is Icon) &&
            ((widget.icon as Icon).icon == Icons.star ||
                (widget.icon as Icon).icon == Icons.star_border),
      ),
    );

    await tap(starButton);
    await pumpAndSettle();
  }

  /// Verifies a sequence is tracked (shows filled star).
  void verifySequenceTracked(String text) {
    final listTile = IntegrationFinders.listTileWithText(text);
    final filledStar = find.descendant(
      of: listTile,
      matching: find.byIcon(Icons.star),
    );
    expect(filledStar, findsOneWidget);
  }

  /// Verifies a sequence is not tracked (shows outline star).
  void verifySequenceNotTracked(String text) {
    final listTile = IntegrationFinders.listTileWithText(text);
    final outlineStar = find.descendant(
      of: listTile,
      matching: find.byIcon(Icons.star_border),
    );
    expect(outlineStar, findsOneWidget);
  }

  /// Taps the bookmark button in the app bar.
  Future<void> tapBookmarkButton() async {
    // Find whichever bookmark icon is present
    final filled = IntegrationFinders.bookmarkFilledIcon;
    final outline = IntegrationFinders.bookmarkOutlineIcon;

    if (filled.evaluate().isNotEmpty) {
      await tap(filled);
    } else {
      await tap(outline);
    }
    await pumpAndSettle();
  }

  /// Verifies the home screen shows the empty state.
  void verifyEmptyState() {
    expect(IntegrationFinders.emptyStateText, findsOneWidget);
    expect(IntegrationFinders.openListButton, findsOneWidget);
  }

  /// Verifies the home screen shows a sequence (not empty).
  void verifySequenceDisplayed(String text) {
    expect(IntegrationFinders.sequenceText(text), findsOneWidget);
    expect(IntegrationFinders.emptyStateText, findsNothing);
  }

  /// Opens the practice sheet by tapping on the main sequence text.
  Future<void> openPracticeSheet() async {
    // Find the GestureDetector wrapping the sequence text
    final sequenceGesture = find.byWidgetPredicate(
      (widget) =>
          widget is GestureDetector &&
          widget.child is Text &&
          (widget.child as Text).style?.fontSize != null,
    );
    if (sequenceGesture.evaluate().isNotEmpty) {
      await tap(sequenceGesture.first);
    } else {
      // Fallback: tap any GestureDetector containing large text
      final gestures = find.byType(GestureDetector);
      for (final element in gestures.evaluate()) {
        final widget = element.widget as GestureDetector;
        if (widget.child is Text) {
          await tap(find.byWidget(widget));
          break;
        }
      }
    }
    await pumpAndSettle();
  }

  /// Taps the recording FAB to start recording.
  Future<void> tapRecordButton() async {
    await tap(IntegrationFinders.recordFab);
    await pump();
  }

  /// Taps the FAB to stop recording and score.
  Future<void> tapStopAndScore() async {
    await tap(IntegrationFinders.recordFab);
    await pumpAndSettle();
  }

  /// Waits for scoring to complete.
  Future<void> waitForScoring({
    Duration timeout = const Duration(seconds: 5),
  }) async {
    final endTime = DateTime.now().add(timeout);
    while (DateTime.now().isBefore(endTime)) {
      await pump(const Duration(milliseconds: 100));
      // Check if scoring indicator is gone
      if (IntegrationFinders.progressIndicator.evaluate().isEmpty) {
        break;
      }
    }
    await pumpAndSettle();
  }

  /// Verifies the recording state indicator.
  Future<void> verifyRecordingState({required bool isRecording}) async {
    if (isRecording) {
      expect(IntegrationFinders.stopIcon, findsOneWidget);
      expect(IntegrationFinders.micIcon, findsNothing);
    } else {
      expect(IntegrationFinders.micIcon, findsOneWidget);
      expect(IntegrationFinders.stopIcon, findsNothing);
    }
  }
}
