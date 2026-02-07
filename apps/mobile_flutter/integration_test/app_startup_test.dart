// integration_test/app_startup_test.dart
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'test_helpers.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('App Startup', () {
    testWidgets('app launches without errors', (tester) async {
      // Arrange & Act
      await tester.pumpIntegrationApp();

      // Assert - app is running (no exceptions thrown)
      expect(find.byType(IntegrationTestApp), findsOneWidget);
    });

    testWidgets('home screen shows title in app bar', (tester) async {
      // Arrange
      await tester.pumpIntegrationApp();

      // Assert
      expect(IntegrationFinders.homeTitle, findsOneWidget);
    });

    testWidgets('home screen shows list button in app bar', (tester) async {
      // Arrange
      await tester.pumpIntegrationApp();

      // Assert
      expect(IntegrationFinders.listButton, findsOneWidget);
    });

    testWidgets('home screen shows empty state when no tracked sequences',
        (tester) async {
      // Arrange - no tracked sequences
      await tester.pumpIntegrationApp(trackedIds: {});

      // Assert
      expect(IntegrationFinders.emptyStateText, findsOneWidget);
      expect(IntegrationFinders.openListButton, findsOneWidget);
    });

    testWidgets('home screen shows current sequence when tracked',
        (tester) async {
      // Arrange - track the first sequence
      await tester.pumpIntegrationApp(trackedIds: {'test-001'});

      // Assert - sequence text is displayed
      expect(IntegrationFinders.sequenceText('你好'), findsOneWidget);
    });

    testWidgets('home screen shows bookmark button when sequence is displayed',
        (tester) async {
      // Arrange
      await tester.pumpIntegrationApp(trackedIds: {'test-001'});

      // Assert - bookmark filled icon (since it's tracked)
      expect(IntegrationFinders.bookmarkFilledIcon, findsOneWidget);
    });

    testWidgets('home screen shows Next button when sequence is displayed',
        (tester) async {
      // Arrange
      await tester.pumpIntegrationApp(trackedIds: {'test-001'});

      // Assert
      expect(IntegrationFinders.nextButton, findsOneWidget);
    });

    testWidgets('home screen shows pinyin when available', (tester) async {
      // Arrange
      await tester.pumpIntegrationApp(trackedIds: {'test-001'});

      // Assert - pinyin is shown by default
      expect(find.text('ni hao'), findsOneWidget);
    });
  });

  group('Navigation', () {
    testWidgets('tapping list button navigates to list screen', (tester) async {
      // Arrange
      await tester.pumpIntegrationApp();

      // Act
      await tester.navigateToListScreen();

      // Assert
      expect(IntegrationFinders.listTitle, findsOneWidget);
      expect(IntegrationFinders.homeTitle, findsNothing);
    });

    testWidgets('list screen shows all sequences', (tester) async {
      // Arrange
      await tester.pumpIntegrationApp();

      // Act
      await tester.navigateToListScreen();

      // Assert - all test sequences are visible
      expect(IntegrationFinders.sequenceText('你好'), findsOneWidget);
      expect(IntegrationFinders.sequenceText('谢谢'), findsOneWidget);
      expect(IntegrationFinders.sequenceText('再见'), findsOneWidget);
    });

    testWidgets('tapping back button returns to home screen', (tester) async {
      // Arrange
      await tester.pumpIntegrationApp(trackedIds: {'test-001'});
      await tester.navigateToListScreen();

      // Act
      await tester.navigateBackToHome();

      // Assert
      expect(IntegrationFinders.homeTitle, findsOneWidget);
      expect(IntegrationFinders.listTitle, findsNothing);
    });

    testWidgets('tapping Open list button navigates to list screen',
        (tester) async {
      // Arrange - empty state shows Open list button
      await tester.pumpIntegrationApp(trackedIds: {});

      // Act
      await tester.tap(IntegrationFinders.openListButton);
      await tester.pumpAndSettle();

      // Assert
      expect(IntegrationFinders.listTitle, findsOneWidget);
    });

    testWidgets('selecting sequence from list returns to home with that sequence',
        (tester) async {
      // Arrange
      await tester.pumpIntegrationApp(trackedIds: {'test-001'});
      await tester.navigateToListScreen();

      // Act - tap on a different sequence
      await tester.tapSequence('谢谢');

      // Assert - home screen shows the selected sequence
      expect(IntegrationFinders.homeTitle, findsOneWidget);
      expect(IntegrationFinders.sequenceText('谢谢'), findsOneWidget);
    });
  });

  group('App Bar Actions', () {
    testWidgets('bookmark button is visible when sequence is tracked',
        (tester) async {
      // Arrange
      await tester.pumpIntegrationApp(trackedIds: {'test-001'});

      // Assert
      expect(IntegrationFinders.bookmarkFilledIcon, findsOneWidget);
      expect(IntegrationFinders.bookmarkOutlineIcon, findsNothing);
    });

    testWidgets('bookmark outline is visible when sequence is not tracked',
        (tester) async {
      // Arrange - sequence is current but not tracked
      // (Need to select from list without tracking)
      await tester.pumpIntegrationApp(trackedIds: {'test-001'});

      // Navigate to list and select untracked sequence
      await tester.navigateToListScreen();
      await tester.tapSequence('谢谢');

      // Assert - outline icon shown for untracked sequence
      expect(IntegrationFinders.bookmarkOutlineIcon, findsOneWidget);
    });
  });
}
