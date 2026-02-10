// integration_test/practice_flow_test.dart
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

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

  // Scoring tests removed - system now uses self-reported ratings instead of
  // automated speech recognition scoring

  // Progress persistence tests removed - system now uses self-reported ratings
  // instead of automated speech recognition scoring

  group('Practice Flow - UI States', () {
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
      await tester.pumpIntegrationApp(trackedIds: {'test-001'});
      await tester.openPracticeSheet();

      // Act - record and stop
      await tester.tap(IntegrationFinders.recordFab);
      await tester.pump(const Duration(milliseconds: 100));
      await tester.tap(IntegrationFinders.recordFab);
      await tester.pumpAndSettle();

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
      await tester.pumpIntegrationApp(
        trackedIds: {'test-001', 'test-002', 'test-003'},
      );

      // Practice first sequence - record and stop
      await tester.tapRecordButton();
      await tester.pump(const Duration(milliseconds: 100));
      await tester.tapRecordButton();
      await tester.pumpAndSettle();

      // Navigate to next
      await tester.tap(IntegrationFinders.nextButton);
      await tester.pumpAndSettle();

      // Practice second sequence - record and stop
      await tester.tapRecordButton();
      await tester.pump(const Duration(milliseconds: 100));
      await tester.tapRecordButton();
      await tester.pumpAndSettle();

      // Assert - FAB is still functional
      expect(IntegrationFinders.recordFab, findsOneWidget);
    });
  });
}
