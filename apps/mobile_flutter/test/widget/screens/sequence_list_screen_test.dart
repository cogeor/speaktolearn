import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:speak_to_learn/app/theme.dart';
import 'package:speak_to_learn/features/text_sequences/presentation/sequence_list_item.dart';
import 'package:speak_to_learn/features/text_sequences/presentation/sequence_list_tile.dart';

void main() {
  const testItems = [
    SequenceListItem(id: 'test-001', text: '你好', isTracked: true, bestScore: 85),
    SequenceListItem(id: 'test-002', text: '谢谢', isTracked: false, bestScore: null),
    SequenceListItem(id: 'test-003', text: '再见', isTracked: false, bestScore: 70),
  ];

  group('SequenceListTile', () {
    testWidgets('displays text correctly', (tester) async {
      const item = SequenceListItem(
        id: 'test-001',
        text: '你好',
        isTracked: false,
        bestScore: null,
      );

      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: Scaffold(
            body: SequenceListTile(
              item: item,
              onTap: () {},
              onToggleTrack: () {},
            ),
          ),
        ),
      );

      expect(find.text('你好'), findsOneWidget);
    });

    testWidgets('shows tracked star when tracked', (tester) async {
      const item = SequenceListItem(
        id: 'test-001',
        text: '你好',
        isTracked: true,
        bestScore: null,
      );

      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: Scaffold(
            body: SequenceListTile(
              item: item,
              onTap: () {},
              onToggleTrack: () {},
            ),
          ),
        ),
      );

      expect(find.byIcon(Icons.star), findsOneWidget);
    });

    testWidgets('shows untracked star when not tracked', (tester) async {
      const item = SequenceListItem(
        id: 'test-001',
        text: '你好',
        isTracked: false,
        bestScore: null,
      );

      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: Scaffold(
            body: SequenceListTile(
              item: item,
              onTap: () {},
              onToggleTrack: () {},
            ),
          ),
        ),
      );

      expect(find.byIcon(Icons.star_border), findsOneWidget);
    });

    testWidgets('shows best score when available', (tester) async {
      const item = SequenceListItem(
        id: 'test-001',
        text: '你好',
        isTracked: false,
        bestScore: 85,
      );

      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: Scaffold(
            body: SequenceListTile(
              item: item,
              onTap: () {},
              onToggleTrack: () {},
            ),
          ),
        ),
      );

      expect(find.text('Best: 85'), findsOneWidget);
    });

    testWidgets('does not show score when null', (tester) async {
      const item = SequenceListItem(
        id: 'test-001',
        text: '你好',
        isTracked: false,
        bestScore: null,
      );

      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: Scaffold(
            body: SequenceListTile(
              item: item,
              onTap: () {},
              onToggleTrack: () {},
            ),
          ),
        ),
      );

      expect(find.textContaining('Best:'), findsNothing);
    });

    testWidgets('calls onTap when tapped', (tester) async {
      bool tapped = false;
      const item = SequenceListItem(
        id: 'test-001',
        text: '你好',
        isTracked: false,
        bestScore: null,
      );

      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: Scaffold(
            body: SequenceListTile(
              item: item,
              onTap: () => tapped = true,
              onToggleTrack: () {},
            ),
          ),
        ),
      );

      await tester.tap(find.byType(ListTile));
      await tester.pump();

      expect(tapped, isTrue);
    });

    testWidgets('calls onToggleTrack when star is tapped', (tester) async {
      bool toggled = false;
      const item = SequenceListItem(
        id: 'test-001',
        text: '你好',
        isTracked: false,
        bestScore: null,
      );

      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: Scaffold(
            body: SequenceListTile(
              item: item,
              onTap: () {},
              onToggleTrack: () => toggled = true,
            ),
          ),
        ),
      );

      await tester.tap(find.byIcon(Icons.star_border));
      await tester.pump();

      expect(toggled, isTrue);
    });
  });

  group('SequenceListItem list rendering', () {
    testWidgets('renders multiple items correctly', (tester) async {
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: Scaffold(
            body: ListView.builder(
              itemCount: testItems.length,
              itemBuilder: (context, index) {
                return SequenceListTile(
                  item: testItems[index],
                  onTap: () {},
                  onToggleTrack: () {},
                );
              },
            ),
          ),
        ),
      );

      expect(find.text('你好'), findsOneWidget);
      expect(find.text('谢谢'), findsOneWidget);
      expect(find.text('再见'), findsOneWidget);
    });

    testWidgets('shows correct star icons for mixed tracked status', (tester) async {
      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: Scaffold(
            body: ListView.builder(
              itemCount: testItems.length,
              itemBuilder: (context, index) {
                return SequenceListTile(
                  item: testItems[index],
                  onTap: () {},
                  onToggleTrack: () {},
                );
              },
            ),
          ),
        ),
      );

      // First item is tracked
      expect(find.byIcon(Icons.star), findsOneWidget);
      // Other two items are not tracked
      expect(find.byIcon(Icons.star_border), findsNWidgets(2));
    });
  });
}
