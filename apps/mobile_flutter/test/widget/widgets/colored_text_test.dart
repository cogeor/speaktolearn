import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:speak_to_learn/app/theme.dart';
import 'package:speak_to_learn/features/practice/presentation/widgets/colored_text.dart';

void main() {
  Widget buildTestWidget({
    required String text,
    List<double>? scores,
    TextStyle? style,
  }) {
    return MaterialApp(
      theme: AppTheme.darkTheme,
      home: Scaffold(
        body: Center(
          child: ColoredText(
            text: text,
            scores: scores,
            style: style,
          ),
        ),
      ),
    );
  }

  group('ColoredText', () {
    group('null scores fallback', () {
      testWidgets('displays plain text when scores is null', (tester) async {
        await tester.pumpWidget(buildTestWidget(text: 'Hello'));

        expect(find.text('Hello'), findsOneWidget);
      });

      testWidgets('displays plain text when scores is empty', (tester) async {
        await tester.pumpWidget(buildTestWidget(text: 'Hello', scores: []));

        expect(find.text('Hello'), findsOneWidget);
      });
    });

    group('color mapping', () {
      testWidgets('maps score < 0.2 to ratingHard (red)', (tester) async {
        await tester.pumpWidget(buildTestWidget(
          text: 'A',
          scores: [0.1],
        ));

        final richText = tester.widget<RichText>(find.byType(RichText));
        final textSpan = richText.text as TextSpan;
        final spans = textSpan.children!.cast<TextSpan>();

        expect(spans[0].style?.color, AppTheme.ratingHard);
      });

      testWidgets('maps score 0.2-0.4 to ratingAlmost (yellow)', (tester) async {
        await tester.pumpWidget(buildTestWidget(
          text: 'A',
          scores: [0.3],
        ));

        final richText = tester.widget<RichText>(find.byType(RichText));
        final textSpan = richText.text as TextSpan;
        final spans = textSpan.children!.cast<TextSpan>();

        expect(spans[0].style?.color, AppTheme.ratingAlmost);
      });

      testWidgets('maps score 0.4-0.6 to ratingGood (green)', (tester) async {
        await tester.pumpWidget(buildTestWidget(
          text: 'A',
          scores: [0.5],
        ));

        final richText = tester.widget<RichText>(find.byType(RichText));
        final textSpan = richText.text as TextSpan;
        final spans = textSpan.children!.cast<TextSpan>();

        expect(spans[0].style?.color, AppTheme.ratingGood);
      });

      testWidgets('maps score >= 0.6 to ratingEasy (blue)', (tester) async {
        await tester.pumpWidget(buildTestWidget(
          text: 'A',
          scores: [0.8],
        ));

        final richText = tester.widget<RichText>(find.byType(RichText));
        final textSpan = richText.text as TextSpan;
        final spans = textSpan.children!.cast<TextSpan>();

        expect(spans[0].style?.color, AppTheme.ratingEasy);
      });

      testWidgets('threshold boundary at 0.2', (tester) async {
        await tester.pumpWidget(buildTestWidget(
          text: 'A',
          scores: [0.2], // Exactly 0.2 should be almost, not bad
        ));

        final richText = tester.widget<RichText>(find.byType(RichText));
        final textSpan = richText.text as TextSpan;
        final spans = textSpan.children!.cast<TextSpan>();

        expect(spans[0].style?.color, AppTheme.ratingAlmost);
      });

      testWidgets('threshold boundary at 0.4', (tester) async {
        await tester.pumpWidget(buildTestWidget(
          text: 'A',
          scores: [0.4], // Exactly 0.4 should be good, not almost
        ));

        final richText = tester.widget<RichText>(find.byType(RichText));
        final textSpan = richText.text as TextSpan;
        final spans = textSpan.children!.cast<TextSpan>();

        expect(spans[0].style?.color, AppTheme.ratingGood);
      });

      testWidgets('threshold boundary at 0.6', (tester) async {
        await tester.pumpWidget(buildTestWidget(
          text: 'A',
          scores: [0.6], // Exactly 0.6 should be easy, not good
        ));

        final richText = tester.widget<RichText>(find.byType(RichText));
        final textSpan = richText.text as TextSpan;
        final spans = textSpan.children!.cast<TextSpan>();

        expect(spans[0].style?.color, AppTheme.ratingEasy);
      });
    });

    group('character handling', () {
      testWidgets('creates one span per character', (tester) async {
        await tester.pumpWidget(buildTestWidget(
          text: 'ABC',
          scores: [0.1, 0.3, 0.8],
        ));

        final richText = tester.widget<RichText>(find.byType(RichText));
        final textSpan = richText.text as TextSpan;
        final spans = textSpan.children!.cast<TextSpan>();

        expect(spans.length, 3);
        expect(spans[0].text, 'A');
        expect(spans[1].text, 'B');
        expect(spans[2].text, 'C');
      });

      testWidgets('handles Chinese characters', (tester) async {
        await tester.pumpWidget(buildTestWidget(
          text: '\u4f60\u597d', // "ni hao" in Chinese (2 chars)
          scores: [0.8, 0.3],
        ));

        final richText = tester.widget<RichText>(find.byType(RichText));
        final textSpan = richText.text as TextSpan;
        final spans = textSpan.children!.cast<TextSpan>();

        expect(spans.length, 2);
        expect(spans[0].style?.color, AppTheme.ratingEasy);
        expect(spans[1].style?.color, AppTheme.ratingAlmost);
      });

      testWidgets('handles emoji as single grapheme', (tester) async {
        // Family emoji is a single grapheme cluster but multiple code points
        await tester.pumpWidget(buildTestWidget(
          text: '\u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}', // family emoji
          scores: [0.9],
        ));

        final richText = tester.widget<RichText>(find.byType(RichText));
        final textSpan = richText.text as TextSpan;
        final spans = textSpan.children!.cast<TextSpan>();

        expect(spans.length, 1);
        expect(spans[0].style?.color, AppTheme.ratingEasy);
      });
    });

    group('mismatched scores', () {
      testWidgets('handles fewer scores than characters', (tester) async {
        await tester.pumpWidget(buildTestWidget(
          text: 'ABC',
          scores: [0.8], // Only one score for 3 chars
        ));

        final richText = tester.widget<RichText>(find.byType(RichText));
        final textSpan = richText.text as TextSpan;
        final spans = textSpan.children!.cast<TextSpan>();

        expect(spans.length, 3);
        expect(spans[0].style?.color, AppTheme.ratingEasy); // Has score
        expect(spans[1].style?.color, AppTheme.foreground); // No score -> white
        expect(spans[2].style?.color, AppTheme.foreground); // No score -> white
      });

      testWidgets('handles more scores than characters', (tester) async {
        await tester.pumpWidget(buildTestWidget(
          text: 'AB',
          scores: [0.8, 0.3, 0.1, 0.5], // 4 scores for 2 chars
        ));

        final richText = tester.widget<RichText>(find.byType(RichText));
        final textSpan = richText.text as TextSpan;
        final spans = textSpan.children!.cast<TextSpan>();

        expect(spans.length, 2); // Only 2 spans for 2 chars
        expect(spans[0].style?.color, AppTheme.ratingEasy);
        expect(spans[1].style?.color, AppTheme.ratingAlmost);
      });
    });

    group('empty text', () {
      testWidgets('handles empty text gracefully', (tester) async {
        await tester.pumpWidget(buildTestWidget(text: ''));

        // Should not crash - either Text or RichText is fine
        expect(find.byType(ColoredText), findsOneWidget);
      });

      testWidgets('handles empty text with scores', (tester) async {
        await tester.pumpWidget(buildTestWidget(text: '', scores: [0.5, 0.6]));

        expect(find.byType(ColoredText), findsOneWidget);
      });
    });
  });
}
