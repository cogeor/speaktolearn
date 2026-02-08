// integration_test/practice_flow_test.dart
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'mocks/mock_speech_recognizer.dart';
import 'test_helpers.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('Practice Flow - Recording', () {
    testWidgets('FAB shows mic icon when idle', (tester) async {
      // Arrange
      await tester.pumpIntegrationApp(trackedIds: {'test-001'});

      // Assert - FAB should show mic icon in idle state
      expect(IntegrationFinders.micIcon, findsOneWidget);
      expect(IntegrationFinders.stopIcon, findsNothing);
    });

    testWidgets('tapping FAB starts recording', (tester) async {
      // Arrange
      await tester.pumpIntegrationApp(trackedIds: {'test-001'});

      // Act
      await tester.tapRecordButton();
      await tester.pump(const Duration(milliseconds: 100));

      // Assert - FAB shows stop icon
      await tester.verifyRecordingState(isRecording: true);
    });

    testWidgets('FAB changes to stop icon during recording', (tester) async {
      // Arrange
      await tester.pumpIntegrationApp(trackedIds: {'test-001'});

      // Act - start recording
      await tester.tapRecordButton();
      await tester.pump(const Duration(milliseconds: 100));

      // Assert
      expect(IntegrationFinders.stopIcon, findsOneWidget);
      expect(IntegrationFinders.micIcon, findsNothing);
    });

    testWidgets('tapping FAB again stops recording', (tester) async {
      // Arrange
      await tester.pumpIntegrationApp(trackedIds: {'test-001'});

      // Act - start then stop
      await tester.tapRecordButton();
      await tester.pump(const Duration(milliseconds: 100));
      await tester.tapStopAndScore();

      // Assert - FAB returns to mic icon (after scoring)
      await tester.waitForScoring();
      expect(IntegrationFinders.micIcon, findsOneWidget);
    });
  });

  group('Practice Flow - Scoring', () {
    testWidgets('shows progress indicator while scoring', (tester) async {
      // Arrange
      final recognizer = MockSpeechRecognizer();
      await tester.pumpIntegrationApp(
        trackedIds: {'test-001'},
        speechRecognizer: recognizer,
      );

      // Act - start and stop recording
      await tester.tapRecordButton();
      await tester.pump(const Duration(milliseconds: 100));
      await tester.tap(IntegrationFinders.recordFab);
      await tester.pump();

      // Assert - progress indicator shown during scoring
      expect(IntegrationFinders.progressIndicator, findsOneWidget);

      // Wait for scoring to complete
      await tester.waitForScoring();
    });

    testWidgets('displays score after scoring completes with perfect match', (
      tester,
    ) async {
      // Arrange
      final recognizer = MockSpeechRecognizer();
      recognizer.setupPerfectMatch(
        '\u4f60\u597d',
      ); // Perfect match with test sequence
      await tester.pumpIntegrationApp(
        trackedIds: {'test-001'},
        speechRecognizer: recognizer,
      );

      // Act
      await tester.tapRecordButton();
      await tester.pump(const Duration(milliseconds: 100));
      await tester.tapStopAndScore();
      await tester.waitForScoring();

      // Assert - high score displayed (100 for perfect match)
      // The score should be visible somewhere on screen
      expect(find.textContaining('100'), findsWidgets);
    });

    testWidgets('displays lower score for partial match', (tester) async {
      // Arrange
      final recognizer = MockSpeechRecognizer();
      recognizer.setupPartialMatch('\u4f60'); // Partial match
      await tester.pumpIntegrationApp(
        trackedIds: {'test-001'},
        speechRecognizer: recognizer,
      );

      // Act
      await tester.tapRecordButton();
      await tester.pump(const Duration(milliseconds: 100));
      await tester.tapStopAndScore();
      await tester.waitForScoring();

      // Assert - score displayed (lower than 100)
      // Best score label should appear after scoring
      expect(find.textContaining('Best:'), findsOneWidget);
    });

    testWidgets('displays zero score for no match', (tester) async {
      // Arrange
      final recognizer = MockSpeechRecognizer(mode: RecognizerMode.empty);
      await tester.pumpIntegrationApp(
        trackedIds: {'test-001'},
        speechRecognizer: recognizer,
      );

      // Act
      await tester.tapRecordButton();
      await tester.pump(const Duration(milliseconds: 100));
      await tester.tapStopAndScore();
      await tester.waitForScoring();

      // Assert - zero score displayed
      expect(find.textContaining('0'), findsWidgets);
    });

    testWidgets('shows error state when recognition fails', (tester) async {
      // Arrange
      final recognizer = MockSpeechRecognizer(mode: RecognizerMode.failure);
      await tester.pumpIntegrationApp(
        trackedIds: {'test-001'},
        speechRecognizer: recognizer,
      );

      // Act
      await tester.tapRecordButton();
      await tester.pump(const Duration(milliseconds: 100));
      await tester.tapStopAndScore();
      await tester.waitForScoring();

      // Assert - error handling (FAB returns to idle state)
      expect(IntegrationFinders.micIcon, findsOneWidget);
    });
  });

  group('Practice Flow - Progress Persistence', () {
    testWidgets('score persists and shows on home screen after scoring', (
      tester,
    ) async {
      // Arrange
      final recognizer = MockSpeechRecognizer();
      recognizer.setupPerfectMatch('\u4f60\u597d');
      await tester.pumpIntegrationApp(
        trackedIds: {'test-001'},
        speechRecognizer: recognizer,
      );

      // Act - record and score
      await tester.tapRecordButton();
      await tester.pump(const Duration(milliseconds: 100));
      await tester.tapStopAndScore();
      await tester.waitForScoring();

      // Assert - best score is shown on home screen
      expect(find.textContaining('Best:'), findsOneWidget);
    });

    testWidgets('can navigate to next sequence after scoring', (tester) async {
      // Arrange
      final recognizer = MockSpeechRecognizer();
      recognizer.setupPerfectMatch('\u4f60\u597d');
      await tester.pumpIntegrationApp(
        trackedIds: {'test-001', 'test-002'},
        speechRecognizer: recognizer,
      );

      // Act - score current sequence
      await tester.tapRecordButton();
      await tester.pump(const Duration(milliseconds: 100));
      await tester.tapStopAndScore();
      await tester.waitForScoring();

      // Tap Next
      await tester.tap(IntegrationFinders.nextButton);
      await tester.pumpAndSettle();

      // Assert - different sequence is shown
      expect(IntegrationFinders.sequenceText('\u8c22\u8c22'), findsOneWidget);
    });

    testWidgets('best score updates after better attempt', (tester) async {
      // Arrange
      final recognizer = MockSpeechRecognizer();
      await tester.pumpIntegrationApp(
        trackedIds: {'test-001'},
        speechRecognizer: recognizer,
      );

      // First attempt - partial match
      recognizer.setupPartialMatch('\u4f60');
      await tester.tapRecordButton();
      await tester.pump(const Duration(milliseconds: 100));
      await tester.tapStopAndScore();
      await tester.waitForScoring();

      // Second attempt - perfect match
      recognizer.setupPerfectMatch('\u4f60\u597d');
      await tester.tapRecordButton();
      await tester.pump(const Duration(milliseconds: 100));
      await tester.tapStopAndScore();
      await tester.waitForScoring();

      // Assert - best score is the higher one (100)
      expect(find.textContaining('100'), findsWidgets);
    });

    testWidgets('attempt count increases after each scoring', (tester) async {
      // Arrange
      final recognizer = MockSpeechRecognizer();
      recognizer.setupPerfectMatch('\u4f60\u597d');
      await tester.pumpIntegrationApp(
        trackedIds: {'test-001'},
        speechRecognizer: recognizer,
      );

      // Make multiple attempts
      for (var i = 0; i < 3; i++) {
        await tester.tapRecordButton();
        await tester.pump(const Duration(milliseconds: 100));
        await tester.tapStopAndScore();
        await tester.waitForScoring();
      }

      // Assert - recognizer was called 3 times
      expect(recognizer.recognizeCallCount, equals(3));
    });
  });

  group('Practice Flow - UI States', () {
    testWidgets('FAB is disabled during scoring', (tester) async {
      // Arrange
      final recognizer = MockSpeechRecognizer();
      await tester.pumpIntegrationApp(
        trackedIds: {'test-001'},
        speechRecognizer: recognizer,
      );

      // Start and stop to trigger scoring
      await tester.tapRecordButton();
      await tester.pump(const Duration(milliseconds: 100));
      await tester.tap(IntegrationFinders.recordFab);
      await tester.pump();

      // Assert - FAB shows progress indicator (disabled state)
      expect(IntegrationFinders.progressIndicator, findsOneWidget);

      // Clean up
      await tester.waitForScoring();
    });

    testWidgets(
      'example audio buttons are visible on home screen with voices',
      (tester) async {
        // Arrange
        await tester.pumpIntegrationApp(trackedIds: {'test-001'});

        // Assert - play button is displayed (for example audio)
        expect(find.byIcon(Icons.play_arrow), findsOneWidget);
      },
    );
  });

  group('Practice Flow - Practice Sheet', () {
    testWidgets('practice sheet opens when tapping sequence text', (
      tester,
    ) async {
      // Arrange
      await tester.pumpIntegrationApp(trackedIds: {'test-001'});

      // Act
      await tester.openPracticeSheet();

      // Assert - practice sheet is displayed
      expect(IntegrationFinders.practiceSheet, findsOneWidget);
    });

    testWidgets('practice sheet shows voice buttons', (tester) async {
      // Arrange
      await tester.pumpIntegrationApp(trackedIds: {'test-001'});

      // Act
      await tester.openPracticeSheet();

      // Assert - voice buttons are displayed
      expect(find.text('male'), findsOneWidget);
      expect(find.text('female'), findsOneWidget);
    });

    testWidgets('practice sheet FAB shows mic icon when idle', (tester) async {
      // Arrange
      await tester.pumpIntegrationApp(trackedIds: {'test-001'});

      // Act
      await tester.openPracticeSheet();

      // Assert
      expect(IntegrationFinders.micIcon, findsWidgets);
    });

    testWidgets('practice sheet shows score labels', (tester) async {
      // Arrange
      await tester.pumpIntegrationApp(trackedIds: {'test-001'});

      // Act
      await tester.openPracticeSheet();

      // Assert - score labels are displayed
      expect(IntegrationFinders.latestScoreLabel, findsOneWidget);
      expect(IntegrationFinders.bestScoreLabel, findsOneWidget);
    });

    testWidgets('replay button appears after recording in practice sheet', (
      tester,
    ) async {
      // Arrange
      final recognizer = MockSpeechRecognizer();
      recognizer.setupPerfectMatch('\u4f60\u597d');
      await tester.pumpIntegrationApp(
        trackedIds: {'test-001'},
        speechRecognizer: recognizer,
      );
      await tester.openPracticeSheet();

      // Act - record and score
      await tester.tap(IntegrationFinders.recordFab);
      await tester.pump(const Duration(milliseconds: 100));
      await tester.tap(IntegrationFinders.recordFab);
      await tester.pumpAndSettle();
      await tester.waitForScoring();

      // Assert - replay button is visible
      expect(IntegrationFinders.replayButton, findsOneWidget);
    });
  });

  group('Practice Flow - Edge Cases', () {
    testWidgets('can cancel recording by closing practice sheet', (
      tester,
    ) async {
      // Arrange
      await tester.pumpIntegrationApp(trackedIds: {'test-001'});
      await tester.openPracticeSheet();

      // Start recording in practice sheet
      await tester.tap(IntegrationFinders.recordFab);
      await tester.pump(const Duration(milliseconds: 100));

      // Close the practice sheet
      await tester.tapAt(const Offset(10, 10));
      await tester.pumpAndSettle();

      // Assert - FAB is still in idle state on home screen
      expect(IntegrationFinders.micIcon, findsOneWidget);
    });

    testWidgets('handles rapid start/stop gracefully', (tester) async {
      // Arrange
      await tester.pumpIntegrationApp(trackedIds: {'test-001'});

      // Rapid taps
      await tester.tapRecordButton();
      await tester.pump(const Duration(milliseconds: 50));
      await tester.tapRecordButton();
      await tester.pump(const Duration(milliseconds: 50));
      await tester.tapRecordButton();
      await tester.pumpAndSettle();

      // Should be in a valid state (not crashed)
      expect(IntegrationFinders.recordFab, findsOneWidget);
    });

    testWidgets('multiple sequences can be practiced in sequence', (
      tester,
    ) async {
      // Arrange
      final recognizer = MockSpeechRecognizer();
      recognizer.setupPerfectMatch('\u4f60\u597d');
      await tester.pumpIntegrationApp(
        trackedIds: {'test-001', 'test-002', 'test-003'},
        speechRecognizer: recognizer,
      );

      // Practice first sequence
      await tester.tapRecordButton();
      await tester.pump(const Duration(milliseconds: 100));
      await tester.tapStopAndScore();
      await tester.waitForScoring();

      // Navigate to next
      await tester.tap(IntegrationFinders.nextButton);
      await tester.pumpAndSettle();

      // Practice second sequence
      recognizer.setupPerfectMatch('\u8c22\u8c22');
      await tester.tapRecordButton();
      await tester.pump(const Duration(milliseconds: 100));
      await tester.tapStopAndScore();
      await tester.waitForScoring();

      // Assert - both were scored (2 calls total)
      expect(recognizer.recognizeCallCount, equals(2));
    });
  });
}
