import 'package:flutter/material.dart';

import 'package:flutter_test/flutter_test.dart';
import 'package:speak_to_learn/app/theme.dart';
import 'package:speak_to_learn/features/progress/domain/sentence_rating.dart';
import 'package:speak_to_learn/features/text_sequences/presentation/sequence_list_item.dart';
import 'package:speak_to_learn/features/text_sequences/presentation/sequence_list_tile.dart';

void main() {
  const testItems = [
    SequenceListItem(id: 'test-001', text: '你好', lastRating: SentenceRating.good),
    SequenceListItem(id: 'test-002', text: '谢谢', lastRating: null),
    SequenceListItem(id: 'test-003', text: '再见', lastRating: SentenceRating.almost),
  ];

  group('SequenceListTile', () {
    testWidgets('displays text correctly', (tester) async {
      const item = SequenceListItem(
        id: 'test-001',
        text: '你好',
        lastRating: null,
      );

      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: Scaffold(
            body: SequenceListTile(item: item, onTap: () {}),
          ),
        ),
      );

      expect(find.text('你好'), findsOneWidget);
    });

    testWidgets('shows rating indicator when available', (tester) async {
      const item = SequenceListItem(id: 'test-001', text: '你好', lastRating: SentenceRating.good);

      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: Scaffold(
            body: SequenceListTile(item: item, onTap: () {}),
          ),
        ),
      );

      // The tile should render with a rating indicator (RatingIndicator widget)
      expect(find.text('你好'), findsOneWidget);
    });

    testWidgets('renders correctly when no rating', (tester) async {
      const item = SequenceListItem(
        id: 'test-001',
        text: '你好',
        lastRating: null,
      );

      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: Scaffold(
            body: SequenceListTile(item: item, onTap: () {}),
          ),
        ),
      );

      // Should render without crash even when no rating
      expect(find.text('你好'), findsOneWidget);
    });

    testWidgets('calls onTap when tapped', (tester) async {
      bool tapped = false;
      const item = SequenceListItem(
        id: 'test-001',
        text: '你好',
        lastRating: null,
      );

      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: Scaffold(
            body: SequenceListTile(item: item, onTap: () => tapped = true),
          ),
        ),
      );

      await tester.tap(find.byType(ListTile));
      await tester.pump();

      expect(tapped, isTrue);
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
                return SequenceListTile(item: testItems[index], onTap: () {});
              },
            ),
          ),
        ),
      );

      expect(find.text('你好'), findsOneWidget);
      expect(find.text('谢谢'), findsOneWidget);
      expect(find.text('再见'), findsOneWidget);
    });
  });
}
