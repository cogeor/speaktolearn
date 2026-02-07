// integration_test/tracking_flow_test.dart
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'test_helpers.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('Tracking Flow - List Screen', () {
    testWidgets('list screen shows all sequences', (tester) async {
      // Arrange
      await tester.pumpIntegrationApp();

      // Act
      await tester.navigateToListScreen();

      // Assert - all sequences visible
      expect(IntegrationFinders.sequenceText('\u4f60\u597d'), findsOneWidget);
      expect(IntegrationFinders.sequenceText('\u8c22\u8c22'), findsOneWidget);
      expect(IntegrationFinders.sequenceText('\u518d\u89c1'), findsOneWidget);
    });

    testWidgets('untracked sequences show outline star', (tester) async {
      // Arrange - no sequences tracked
      await tester.pumpIntegrationApp(trackedIds: {});

      // Act
      await tester.navigateToListScreen();

      // Assert - all show outline star
      tester.verifySequenceNotTracked('\u4f60\u597d');
      tester.verifySequenceNotTracked('\u8c22\u8c22');
      tester.verifySequenceNotTracked('\u518d\u89c1');
    });

    testWidgets('tracked sequences show filled star', (tester) async {
      // Arrange - first sequence tracked
      await tester.pumpIntegrationApp(trackedIds: {'test-001'});

      // Act
      await tester.navigateToListScreen();

      // Assert
      tester.verifySequenceTracked('\u4f60\u597d');
      tester.verifySequenceNotTracked('\u8c22\u8c22');
      tester.verifySequenceNotTracked('\u518d\u89c1');
    });

    testWidgets('tapping star tracks untracked sequence', (tester) async {
      // Arrange - no sequences tracked
      await tester.pumpIntegrationApp(trackedIds: {});
      await tester.navigateToListScreen();

      // Act
      await tester.tapTrackButtonForSequence('\u8c22\u8c22');

      // Assert - now shows filled star
      tester.verifySequenceTracked('\u8c22\u8c22');
    });

    testWidgets('tapping star untracks tracked sequence', (tester) async {
      // Arrange - sequence is tracked
      await tester.pumpIntegrationApp(trackedIds: {'test-002'});
      await tester.navigateToListScreen();

      // Verify initially tracked
      tester.verifySequenceTracked('\u8c22\u8c22');

      // Act
      await tester.tapTrackButtonForSequence('\u8c22\u8c22');

      // Assert - now shows outline star
      tester.verifySequenceNotTracked('\u8c22\u8c22');
    });

    testWidgets('tracked sequences appear at top of list', (tester) async {
      // Arrange - first and third sequences tracked
      await tester.pumpIntegrationApp(trackedIds: {'test-001', 'test-003'});

      // Act
      await tester.navigateToListScreen();

      // Assert - tracked items should have stars
      final allStars = find.byIcon(Icons.star);
      expect(allStars, findsNWidgets(2)); // Two tracked
    });
  });

  group('Tracking Flow - Home Screen Integration', () {
    testWidgets('tracked sequence appears on home screen', (tester) async {
      // Arrange
      await tester.pumpIntegrationApp(trackedIds: {'test-001'});

      // Assert - sequence is displayed on home
      tester.verifySequenceDisplayed('\u4f60\u597d');
    });

    testWidgets('tracking from list makes sequence appear on home',
        (tester) async {
      // Arrange - start with no tracked sequences
      await tester.pumpIntegrationApp(trackedIds: {});

      // Verify empty state
      tester.verifyEmptyState();

      // Act - navigate to list and track a sequence
      await tester.navigateToListScreen();
      await tester.tapTrackButtonForSequence('\u8c22\u8c22');
      await tester.navigateBackToHome();

      // Assert - sequence now appears on home
      tester.verifySequenceDisplayed('\u8c22\u8c22');
    });

    testWidgets('home screen bookmark shows tracked state', (tester) async {
      // Arrange
      await tester.pumpIntegrationApp(trackedIds: {'test-001'});

      // Assert - bookmark is filled (tracked)
      expect(IntegrationFinders.bookmarkFilledIcon, findsOneWidget);
      expect(IntegrationFinders.bookmarkOutlineIcon, findsNothing);
    });

    testWidgets('untracking last sequence shows empty state', (tester) async {
      // Arrange - only one sequence tracked
      await tester.pumpIntegrationApp(trackedIds: {'test-001'});

      // Act - untrack via bookmark button
      await tester.tapBookmarkButton();

      // Assert - empty state shown
      tester.verifyEmptyState();
    });
  });

  group('Tracking Flow - Bookmark Button', () {
    testWidgets('tapping bookmark untracks current sequence', (tester) async {
      // Arrange
      await tester.pumpIntegrationApp(trackedIds: {'test-001'});

      // Verify initial state
      expect(IntegrationFinders.bookmarkFilledIcon, findsOneWidget);

      // Act
      await tester.tapBookmarkButton();

      // Assert - bookmark changes to outline
      expect(IntegrationFinders.bookmarkOutlineIcon, findsOneWidget);
      expect(IntegrationFinders.bookmarkFilledIcon, findsNothing);
    });

    testWidgets('tapping bookmark tracks current sequence', (tester) async {
      // Arrange - select untracked sequence
      await tester.pumpIntegrationApp(trackedIds: {'test-001'});
      await tester.navigateToListScreen();
      await tester.tapSequence('\u8c22\u8c22'); // Select untracked

      // Verify initial state - should show outline
      expect(IntegrationFinders.bookmarkOutlineIcon, findsOneWidget);

      // Act
      await tester.tapBookmarkButton();

      // Assert - bookmark changes to filled
      expect(IntegrationFinders.bookmarkFilledIcon, findsOneWidget);
    });

    testWidgets('untracking advances to next tracked sequence', (tester) async {
      // Arrange - multiple sequences tracked
      await tester.pumpIntegrationApp(trackedIds: {'test-001', 'test-002'});

      // Verify current is first
      tester.verifySequenceDisplayed('\u4f60\u597d');

      // Act - untrack current
      await tester.tapBookmarkButton();

      // Assert - shows next tracked sequence
      tester.verifySequenceDisplayed('\u8c22\u8c22');
    });
  });

  group('Tracking Flow - Persistence', () {
    testWidgets('tracking state persists after navigation', (tester) async {
      // Arrange
      await tester.pumpIntegrationApp(trackedIds: {});
      await tester.navigateToListScreen();

      // Act - track a sequence
      await tester.tapTrackButtonForSequence('\u518d\u89c1');

      // Navigate away and back
      await tester.navigateBackToHome();
      await tester.navigateToListScreen();

      // Assert - still tracked
      tester.verifySequenceTracked('\u518d\u89c1');
    });

    testWidgets('untracking state persists after navigation', (tester) async {
      // Arrange
      await tester.pumpIntegrationApp(trackedIds: {'test-001'});
      await tester.navigateToListScreen();

      // Act - untrack
      await tester.tapTrackButtonForSequence('\u4f60\u597d');

      // Navigate away and back
      await tester.navigateBackToHome();
      await tester.navigateToListScreen();

      // Assert - still untracked
      tester.verifySequenceNotTracked('\u4f60\u597d');
    });

    testWidgets('tracking from home updates list screen', (tester) async {
      // Arrange - select untracked sequence
      await tester.pumpIntegrationApp(trackedIds: {'test-001'});
      await tester.navigateToListScreen();
      await tester.tapSequence('\u8c22\u8c22');

      // Track via home bookmark
      await tester.tapBookmarkButton();

      // Navigate to list
      await tester.navigateToListScreen();

      // Assert - shows tracked in list
      tester.verifySequenceTracked('\u8c22\u8c22');
    });
  });

  group('Tracking Flow - Edge Cases', () {
    testWidgets('can track all sequences', (tester) async {
      // Arrange
      await tester.pumpIntegrationApp(trackedIds: {});
      await tester.navigateToListScreen();

      // Act - track all
      await tester.tapTrackButtonForSequence('\u4f60\u597d');
      await tester.tapTrackButtonForSequence('\u8c22\u8c22');
      await tester.tapTrackButtonForSequence('\u518d\u89c1');

      // Assert - all tracked
      tester.verifySequenceTracked('\u4f60\u597d');
      tester.verifySequenceTracked('\u8c22\u8c22');
      tester.verifySequenceTracked('\u518d\u89c1');
    });

    testWidgets('can untrack all sequences', (tester) async {
      // Arrange
      await tester.pumpIntegrationApp(
          trackedIds: {'test-001', 'test-002', 'test-003'});
      await tester.navigateToListScreen();

      // Act - untrack all
      await tester.tapTrackButtonForSequence('\u4f60\u597d');
      await tester.tapTrackButtonForSequence('\u8c22\u8c22');
      await tester.tapTrackButtonForSequence('\u518d\u89c1');

      // Assert - all untracked
      tester.verifySequenceNotTracked('\u4f60\u597d');
      tester.verifySequenceNotTracked('\u8c22\u8c22');
      tester.verifySequenceNotTracked('\u518d\u89c1');

      // Navigate home - should show empty state
      await tester.navigateBackToHome();
      tester.verifyEmptyState();
    });

    testWidgets('rapid track/untrack toggles correctly', (tester) async {
      // Arrange
      await tester.pumpIntegrationApp(trackedIds: {});
      await tester.navigateToListScreen();

      // Act - rapid toggles
      await tester.tapTrackButtonForSequence('\u4f60\u597d');
      await tester.tapTrackButtonForSequence('\u4f60\u597d');
      await tester.tapTrackButtonForSequence('\u4f60\u597d');

      // Assert - should end in tracked state (odd number of taps)
      tester.verifySequenceTracked('\u4f60\u597d');
    });

    testWidgets('selecting sequence from list sets it as current',
        (tester) async {
      // Arrange
      await tester.pumpIntegrationApp(trackedIds: {'test-001'});
      await tester.navigateToListScreen();

      // Act - select different sequence
      await tester.tapSequence('\u518d\u89c1');

      // Assert - home shows selected sequence
      tester.verifySequenceDisplayed('\u518d\u89c1');
    });

    testWidgets('Next button cycles through tracked sequences', (tester) async {
      // Arrange - multiple tracked
      await tester.pumpIntegrationApp(
          trackedIds: {'test-001', 'test-002', 'test-003'});

      // Keep track of seen sequences
      final seen = <String>{};

      // Tap Next multiple times to cycle
      for (var i = 0; i < 4; i++) {
        // Find current sequence text
        final textFinder = find.byType(Text);
        final widgets = textFinder.evaluate();
        for (final widget in widgets) {
          final text = (widget.widget as Text).data;
          if (text == '\u4f60\u597d' ||
              text == '\u8c22\u8c22' ||
              text == '\u518d\u89c1') {
            seen.add(text!);
            break;
          }
        }

        await tester.tap(IntegrationFinders.nextButton);
        await tester.pumpAndSettle();
      }

      // Assert - should have seen all sequences
      expect(seen.length, equals(3));
    });
  });

  group('Tracking Flow - List Sorting', () {
    testWidgets('tracked sequences sorted before untracked', (tester) async {
      // Arrange - only middle sequence tracked
      await tester.pumpIntegrationApp(trackedIds: {'test-002'});

      // Act
      await tester.navigateToListScreen();

      // Assert - tracked item (test-002) should be first
      final listTiles = find.byType(ListTile);
      final firstTile = listTiles.first;

      // The first tile should contain the tracked sequence
      expect(
        find.descendant(of: firstTile, matching: find.text('\u8c22\u8c22')),
        findsOneWidget,
      );
    });

    testWidgets('list updates order after tracking', (tester) async {
      // Arrange - no sequences tracked
      await tester.pumpIntegrationApp(trackedIds: {});
      await tester.navigateToListScreen();

      // Act - track a different sequence
      await tester.tapTrackButtonForSequence('\u518d\u89c1');

      // Allow list to re-sort
      await tester.pumpAndSettle();

      // Assert - tracked item should now be first
      final listTiles = find.byType(ListTile);
      final firstTile = listTiles.first;
      expect(
        find.descendant(of: firstTile, matching: find.text('\u518d\u89c1')),
        findsOneWidget,
      );
    });
  });
}
