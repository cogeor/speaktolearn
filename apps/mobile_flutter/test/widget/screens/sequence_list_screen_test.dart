import 'package:flutter/material.dart';

import 'package:flutter_test/flutter_test.dart';
import 'package:speak_to_learn/app/theme.dart';
import 'package:speak_to_learn/features/text_sequences/presentation/sequence_list_item.dart';
import 'package:speak_to_learn/features/text_sequences/presentation/sequence_list_tile.dart';

void main() {
  const testItems = [
    SequenceListItem(
      id: 'test-001',
      text: '你好',
      bestScore: 85,
    ),
    SequenceListItem(
      id: 'test-002',
      text: '谢谢',
      bestScore: null,
    ),
    SequenceListItem(
      id: 'test-003',
      text: '再见',
      bestScore: 70,
    ),
  ];

  group('SequenceListTile', () {
    testWidgets('displays text correctly', (tester) async {
      const item = SequenceListItem(
        id: 'test-001',
        text: '你好',
        bestScore: null,
      );

      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: Scaffold(
            body: SequenceListTile(
              item: item,
              onTap: () {},
            ),
          ),
        ),
      );

      expect(find.text('你好'), findsOneWidget);
    });

    testWidgets('shows best score when available', (tester) async {
      const item = SequenceListItem(
        id: 'test-001',
        text: '你好',
        bestScore: 85,
      );

      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: Scaffold(
            body: SequenceListTile(
              item: item,
              onTap: () {},
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
        bestScore: null,
      );

      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: Scaffold(
            body: SequenceListTile(
              item: item,
              onTap: () {},
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
        bestScore: null,
      );

      await tester.pumpWidget(
        MaterialApp(
          theme: AppTheme.darkTheme,
          home: Scaffold(
            body: SequenceListTile(
              item: item,
              onTap: () => tapped = true,
            ),
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
                return SequenceListTile(
                  item: testItems[index],
                  onTap: () {},
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
  });
}
