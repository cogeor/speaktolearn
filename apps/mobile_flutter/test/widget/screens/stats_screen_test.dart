import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:speak_to_learn/app/di.dart';
import 'package:speak_to_learn/app/theme.dart';
import 'package:speak_to_learn/features/stats/domain/practice_stats.dart';
import 'package:speak_to_learn/features/stats/presentation/stats_controller.dart';
import 'package:speak_to_learn/features/stats/presentation/stats_screen.dart';

import '../../mocks/mock_audio.dart';
import '../../mocks/mock_repositories.dart';

void main() {
  Widget buildTestWidget(PracticeStats stats) {
    return ProviderScope(
      overrides: [
        // Override the async provider to return the provided stats directly
        statsControllerProvider.overrideWith(() => _TestStatsController(stats)),
        // Override base dependencies to avoid Hive initialization
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
      child: MaterialApp(theme: AppTheme.darkTheme, home: const StatsScreen()),
    );
  }

  group('StatsScreen empty state', () {
    testWidgets('shows empty state message when totalAttempts is 0', (
      tester,
    ) async {
      await tester.pumpWidget(
        buildTestWidget(const PracticeStats(totalAttempts: 0)),
      );
      await tester.pumpAndSettle();

      expect(find.text('No practice data yet'), findsOneWidget);
      expect(
        find.text(
          'Complete some practice sessions to see your statistics here.',
        ),
        findsOneWidget,
      );
    });

    testWidgets('shows empty icon when no data', (tester) async {
      await tester.pumpWidget(
        buildTestWidget(const PracticeStats(totalAttempts: 0)),
      );
      await tester.pumpAndSettle();

      expect(find.byIcon(Icons.bar_chart_outlined), findsOneWidget);
    });

    testWidgets('shows start practicing button in empty state', (tester) async {
      await tester.pumpWidget(
        buildTestWidget(const PracticeStats(totalAttempts: 0)),
      );
      await tester.pumpAndSettle();

      expect(find.text('Start Practicing'), findsOneWidget);
      expect(find.byIcon(Icons.play_arrow), findsOneWidget);
    });

    testWidgets('renders without crash when all fields are default', (
      tester,
    ) async {
      await tester.pumpWidget(buildTestWidget(const PracticeStats()));
      await tester.pumpAndSettle();

      // Should not throw and should show empty state
      expect(find.text('No practice data yet'), findsOneWidget);
    });
  });

  group('StatsScreen with data', () {
    testWidgets('shows stats grid when data exists', (tester) async {
      await tester.pumpWidget(
        buildTestWidget(
          const PracticeStats(
            totalAttempts: 50,
            sequencesPracticed: 12,
            hardCount: 5,
            almostCount: 8,
            goodCount: 22,
            easyCount: 15,
            currentStreak: 3,
            longestStreak: 7,
          ),
        ),
      );
      await tester.pumpAndSettle();

      expect(find.text('50'), findsOneWidget); // Total attempts
      expect(find.text('12'), findsOneWidget); // Sequences practiced
      expect(find.text('3 days'), findsOneWidget); // Current streak
    });

    testWidgets('shows rating counts when data exists', (tester) async {
      await tester.pumpWidget(
        buildTestWidget(
          const PracticeStats(
            totalAttempts: 5,
            sequencesPracticed: 2,
            hardCount: 1,
            almostCount: 1,
            goodCount: 2,
            easyCount: 1,
          ),
        ),
      );
      await tester.pumpAndSettle();

      // The screen should render without errors
      expect(find.text('5'), findsOneWidget); // Total attempts
    });

    testWidgets('does not show empty state when data exists', (tester) async {
      await tester.pumpWidget(
        buildTestWidget(
          const PracticeStats(totalAttempts: 10, sequencesPracticed: 5),
        ),
      );
      await tester.pumpAndSettle();

      // Should NOT show empty state message
      expect(find.text('No practice data yet'), findsNothing);
      // Should show the stats grid with data
      expect(find.text('10'), findsOneWidget); // Total attempts
    });
  });

  group('StatsScreen app bar', () {
    testWidgets('has Statistics title', (tester) async {
      await tester.pumpWidget(buildTestWidget(const PracticeStats()));
      await tester.pumpAndSettle();

      expect(find.text('Statistics'), findsOneWidget);
    });

    testWidgets('has back button', (tester) async {
      await tester.pumpWidget(buildTestWidget(const PracticeStats()));
      await tester.pumpAndSettle();

      expect(find.byIcon(Icons.arrow_back), findsOneWidget);
    });
  });

  group('PracticeStats.demo', () {
    test('generates non-empty demo data', () {
      final demo = PracticeStats.demo();

      expect(demo.totalAttempts, greaterThan(0));
      expect(demo.sequencesPracticed, greaterThan(0));
      expect(
        demo.hardCount + demo.almostCount + demo.goodCount + demo.easyCount,
        equals(demo.totalAttempts),
      );
      expect(demo.currentStreak, greaterThan(0));
      expect(demo.longestStreak, greaterThan(0));
      expect(demo.dailyAttempts.isNotEmpty, isTrue);
    });

    test('demo data has realistic values', () {
      final demo = PracticeStats.demo();

      expect(demo.hardCount, greaterThanOrEqualTo(0));
      expect(demo.almostCount, greaterThanOrEqualTo(0));
      expect(demo.goodCount, greaterThanOrEqualTo(0));
      expect(demo.easyCount, greaterThanOrEqualTo(0));
      expect(demo.currentStreak, lessThanOrEqualTo(demo.longestStreak + 1));
      expect(demo.totalAttempts, greaterThanOrEqualTo(demo.sequencesPracticed));
    });
  });

  group('PracticeStats.empty', () {
    test('has all zero/null values', () {
      const empty = PracticeStats.empty;

      expect(empty.totalAttempts, equals(0));
      expect(empty.sequencesPracticed, equals(0));
      expect(empty.hardCount, equals(0));
      expect(empty.almostCount, equals(0));
      expect(empty.goodCount, equals(0));
      expect(empty.easyCount, equals(0));
      expect(empty.currentStreak, equals(0));
      expect(empty.longestStreak, equals(0));
      expect(empty.lastPracticeDate, isNull);
      expect(empty.dailyAttempts, isEmpty);
    });
  });
}

/// Test implementation of StatsController that returns pre-set stats.
class _TestStatsController extends AsyncNotifier<PracticeStats>
    implements StatsController {
  final PracticeStats _stats;

  _TestStatsController(this._stats);

  @override
  Future<PracticeStats> build() async {
    return _stats;
  }
}
