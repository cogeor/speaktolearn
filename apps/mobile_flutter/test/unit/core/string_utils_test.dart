import 'package:flutter_test/flutter_test.dart';
import 'package:speak_to_learn/core/utils/string_utils.dart';

void main() {
  group('normalizeZhText', () {
    test('removes Chinese punctuation', () {
      expect(normalizeZhText('我想喝水。'), '我想喝水');
      expect(normalizeZhText('你好！'), '你好');
      expect(normalizeZhText('好的，谢谢'), '好的谢谢');
      expect(normalizeZhText('是吗？'), '是吗');
      expect(normalizeZhText('一、二、三'), '一二三');
      expect(normalizeZhText('你好；再见'), '你好再见');
      expect(normalizeZhText('他说：好的'), '他说好的');
      expect(normalizeZhText('"你好"'), '你好');
      expect(normalizeZhText("'再见'"), '再见');
      expect(normalizeZhText('（括号）'), '括号');
      expect(normalizeZhText('【方括号】'), '方括号');
      expect(normalizeZhText('《书名》'), '书名');
      expect(normalizeZhText('〈角括号〉'), '角括号');
      expect(normalizeZhText('等一下…'), '等一下');
      expect(normalizeZhText('你好—再见'), '你好再见');
      expect(normalizeZhText('中·间·点'), '中间点');
    });

    test('removes ASCII punctuation', () {
      expect(normalizeZhText('Hello, world!'), 'helloworld');
      expect(normalizeZhText('Hello.World'), 'helloworld');
      expect(normalizeZhText('Hello?'), 'hello');
      expect(normalizeZhText('Hello;World'), 'helloworld');
      expect(normalizeZhText('Hello:World'), 'helloworld');
      expect(normalizeZhText('"Hello"'), 'hello');
      expect(normalizeZhText("'Hello'"), 'hello');
      expect(normalizeZhText('(Hello)'), 'hello');
      expect(normalizeZhText('[Hello]'), 'hello');
      expect(normalizeZhText('<Hello>'), 'hello');
    });

    test('removes whitespace', () {
      expect(normalizeZhText('你 好'), '你好');
      expect(normalizeZhText('你  好'), '你好');
      expect(normalizeZhText('你\t好'), '你好');
      expect(normalizeZhText('你\n好'), '你好');
      expect(normalizeZhText(' 你好 '), '你好');
    });

    test('removes full-width space', () {
      expect(normalizeZhText('你\u3000好'), '你好'); // Full-width space
    });

    test('converts full-width ASCII to half-width', () {
      expect(normalizeZhText('Ａ'), 'a');
      expect(normalizeZhText('Ｚ'), 'z');
      expect(normalizeZhText('ａ'), 'a');
      expect(normalizeZhText('ｚ'), 'z');
      expect(normalizeZhText('１２３'), '123');
      expect(normalizeZhText('０'), '0');
      expect(normalizeZhText('９'), '9');
      expect(normalizeZhText('ＡＢＣＤ'), 'abcd');
      expect(normalizeZhText('ａｂｃｄ'), 'abcd');
    });

    test('converts to lowercase', () {
      expect(normalizeZhText('HELLO'), 'hello');
      expect(normalizeZhText('Hello'), 'hello');
      expect(normalizeZhText('ABC你好'), 'abc你好');
    });

    test('handles mixed content', () {
      expect(normalizeZhText('你好，World！'), '你好world');
      expect(normalizeZhText('Ａ你Ｂ好Ｃ'), 'a你b好c');
      expect(normalizeZhText('1 2 3 ４ ５ ６'), '123456');
    });

    test('handles empty string', () {
      expect(normalizeZhText(''), '');
    });

    test('handles string with only punctuation', () {
      expect(normalizeZhText('。，！？'), '');
      expect(normalizeZhText('...'), '');
    });
  });

  group('levenshteinDistance', () {
    test('returns 0 for identical strings', () {
      expect(levenshteinDistance('abc', 'abc'), 0);
      expect(levenshteinDistance('你好', '你好'), 0);
      expect(levenshteinDistance('我想喝水', '我想喝水'), 0);
    });

    test('handles empty strings', () {
      expect(levenshteinDistance('', ''), 0);
      expect(levenshteinDistance('abc', ''), 3);
      expect(levenshteinDistance('', 'abc'), 3);
      expect(levenshteinDistance('你好', ''), 2);
      expect(levenshteinDistance('', '你好'), 2);
    });

    test('calculates single insertion', () {
      expect(levenshteinDistance('abc', 'abcd'), 1);
      expect(levenshteinDistance('你好', '你好吗'), 1);
    });

    test('calculates single deletion', () {
      expect(levenshteinDistance('abcd', 'abc'), 1);
      expect(levenshteinDistance('你好吗', '你好'), 1);
    });

    test('calculates single substitution', () {
      expect(levenshteinDistance('abc', 'adc'), 1);
      expect(levenshteinDistance('你好', '我好'), 1);
      expect(levenshteinDistance('我想喝水', '我想和水'), 1);
    });

    test('calculates multiple operations', () {
      expect(levenshteinDistance('kitten', 'sitting'), 3);
      expect(levenshteinDistance('你好世界', '我爱中国'), 4);
    });

    test('handles Chinese characters correctly', () {
      // Verify each Chinese character is counted as one unit
      expect(levenshteinDistance('一', '二'), 1);
      expect(levenshteinDistance('一二', '一三'), 1);
      expect(levenshteinDistance('一二三', '四五六'), 3);
    });

    test('is symmetric', () {
      expect(
        levenshteinDistance('abc', 'def'),
        levenshteinDistance('def', 'abc'),
      );
      expect(
        levenshteinDistance('你好', '再见'),
        levenshteinDistance('再见', '你好'),
      );
    });
  });

  group('calculateCer', () {
    test('returns 0 for identical strings', () {
      expect(calculateCer('我想喝水', '我想喝水'), 0.0);
      expect(calculateCer('abc', 'abc'), 0.0);
      expect(calculateCer('', ''), 0.0);
    });

    test('returns 1 for completely different strings of same length', () {
      expect(calculateCer('你好', '我们'), 1.0);
      expect(calculateCer('ab', 'cd'), 1.0);
    });

    test('calculates partial match correctly', () {
      // Reference: 我想喝水 (4 chars)
      // Hypothesis: 我想和水 (1 substitution)
      // CER = 1/4 = 0.25
      expect(calculateCer('我想喝水', '我想和水'), 0.25);
    });

    test('handles empty reference', () {
      expect(calculateCer('', ''), 0.0);
      expect(calculateCer('', 'abc'), 1.0);
      expect(calculateCer('', '你好'), 1.0);
    });

    test('handles empty hypothesis', () {
      // All deletions: CER = length(ref) / length(ref) = 1.0
      expect(calculateCer('abc', ''), 1.0);
      expect(calculateCer('你好', ''), 1.0);
      expect(calculateCer('我想喝水', ''), 1.0);
    });

    test('can exceed 1.0 when hypothesis is much longer', () {
      // Reference: ab (2 chars)
      // Hypothesis: abcde (need 3 deletions)
      // CER = 3/2 = 1.5
      expect(calculateCer('ab', 'abcde'), 1.5);
    });

    test('calculates CER for insertion errors', () {
      // Reference: 你好 (2 chars)
      // Hypothesis: 你们好 (1 insertion)
      // CER = 1/2 = 0.5
      expect(calculateCer('你好', '你们好'), 0.5);
    });

    test('calculates CER for deletion errors', () {
      // Reference: 你好吗 (3 chars)
      // Hypothesis: 你好 (1 deletion)
      // CER = 1/3 = 0.333...
      expect(calculateCer('你好吗', '你好'), closeTo(0.333, 0.001));
    });

    test('calculates CER for mixed errors', () {
      // Reference: 我想喝水 (4 chars)
      // Hypothesis: 你想和茶吧 (5 chars, requires 3 substitutions + 1 insertion = distance 4)
      // CER = 4/4 = 1.0
      expect(calculateCer('我想喝水', '你想和茶吧'), 1.0);
    });
  });
}
