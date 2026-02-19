import 'package:flutter_test/flutter_test.dart';
import 'package:speak_to_learn/features/ml_inference/data/syllable_vocab.dart';

void main() {
  group('SyllableVocab', () {
    late SyllableVocab vocab;

    setUpAll(() async {
      TestWidgetsFlutterBinding.ensureInitialized();
      vocab = await SyllableVocab.load();
    });

    test('has correct vocabulary size', () {
      // 530 syllables + 2 special tokens = 532
      expect(vocab.size, equals(532));
    });

    test('special tokens have correct values', () {
      expect(SyllableVocab.padToken, equals(0));
      expect(SyllableVocab.bosToken, equals(1));
      expect(SyllableVocab.specialTokens, equals(2));
    });

    group('encode', () {
      test('encodes known syllables correctly', () {
        // Test a few known syllables
        final ma = vocab.encode('ma');
        expect(ma, greaterThanOrEqualTo(2)); // Should be >= special tokens
        expect(ma, lessThan(532)); // Should be within vocab size

        final ni = vocab.encode('ni');
        expect(ni, greaterThanOrEqualTo(2));
        expect(ni, lessThan(532));

        final hao = vocab.encode('hao');
        expect(hao, greaterThanOrEqualTo(2));
        expect(hao, lessThan(532));
      });

      test('is case insensitive', () {
        final lower = vocab.encode('ma');
        final upper = vocab.encode('MA');
        final mixed = vocab.encode('Ma');

        expect(upper, equals(lower));
        expect(mixed, equals(lower));
      });

      test('normalizes v to u', () {
        // The normalization replaces 'v' with 'u' in the input string
        // So 'nv' becomes 'nu', 'lv' becomes 'lu' (not 'lü')
        final nu = vocab.encode('nu');
        final nv = vocab.encode('nv');

        expect(nv, equals(nu));
      });

      test('returns padToken for unknown syllables', () {
        expect(vocab.encode('xyz'), equals(SyllableVocab.padToken));
        expect(vocab.encode('unknown'), equals(SyllableVocab.padToken));
        expect(vocab.encode(''), equals(SyllableVocab.padToken));
      });
    });

    group('decode', () {
      test('decodes special tokens correctly', () {
        expect(vocab.decode(SyllableVocab.padToken), equals('<PAD>'));
        expect(vocab.decode(SyllableVocab.bosToken), equals('<BOS>'));
      });

      test('decodes syllable tokens correctly', () {
        final maIdx = vocab.encode('ma');
        expect(vocab.decode(maIdx), equals('ma'));

        final niIdx = vocab.encode('ni');
        expect(vocab.decode(niIdx), equals('ni'));

        final haoIdx = vocab.encode('hao');
        expect(vocab.decode(haoIdx), equals('hao'));
      });

      test('returns <UNK> for unknown indices', () {
        expect(vocab.decode(9999), equals('<UNK>'));
        expect(vocab.decode(-1), equals('<UNK>'));
      });

      test('roundtrip encode/decode preserves syllable', () {
        final testSyllables = ['ma', 'ni', 'hao', 'shi', 'de'];
        for (final syl in testSyllables) {
          final idx = vocab.encode(syl);
          final decoded = vocab.decode(idx);
          expect(decoded, equals(syl));
        }
      });
    });

    group('encodeSequence', () {
      test('encodes sequence with BOS token by default', () {
        final tokens = vocab.encodeSequence(['ni', 'hao']);

        expect(tokens.length, equals(3)); // BOS + 2 syllables
        expect(tokens[0], equals(SyllableVocab.bosToken));
        expect(tokens[1], equals(vocab.encode('ni')));
        expect(tokens[2], equals(vocab.encode('hao')));
      });

      test('encodes sequence without BOS when addBos is false', () {
        final tokens = vocab.encodeSequence(['ni', 'hao'], addBos: false);

        expect(tokens.length, equals(2)); // Just 2 syllables
        expect(tokens[0], equals(vocab.encode('ni')));
        expect(tokens[1], equals(vocab.encode('hao')));
      });

      test('handles empty sequence', () {
        final withBos = vocab.encodeSequence([]);
        expect(withBos.length, equals(1));
        expect(withBos[0], equals(SyllableVocab.bosToken));

        final withoutBos = vocab.encodeSequence([], addBos: false);
        expect(withoutBos.length, equals(0));
      });

      test('handles sequence with unknown syllables', () {
        final tokens = vocab.encodeSequence(['ma', 'unknown', 'ni']);

        expect(tokens.length, equals(4)); // BOS + 3 syllables
        expect(tokens[0], equals(SyllableVocab.bosToken));
        expect(tokens[1], equals(vocab.encode('ma')));
        expect(tokens[2], equals(SyllableVocab.padToken)); // unknown -> PAD
        expect(tokens[3], equals(vocab.encode('ni')));
      });
    });

    group('vocabulary completeness', () {
      test('contains common Mandarin syllables', () {
        final common = [
          'a',
          'ai',
          'an',
          'ang',
          'ao',
          'ba',
          'bai',
          'ban',
          'bang',
          'bao',
          'ma',
          'mai',
          'man',
          'mang',
          'mao',
          'ni',
          'hao',
          'shi',
          'de',
          'le',
          'yi',
          'er',
          'san',
          'si',
          'wu',
          'liu',
          'qi',
          'ba',
          'jiu',
        ];

        for (final syl in common) {
          final idx = vocab.encode(syl);
          expect(
            idx,
            isNot(equals(SyllableVocab.padToken)),
            reason: 'Common syllable "$syl" should be in vocabulary',
          );
        }
      });

      test('contains syllables with ü', () {
        // ü syllables normalize: ü → v → u
        // So lü → lv → lu, nü → nv → nu
        // Note: lüe/nüe → lue/nue are not standard pinyin syllables
        final umlautSyllables = ['lü', 'nü'];

        for (final syl in umlautSyllables) {
          final idx = vocab.encode(syl);
          expect(
            idx,
            isNot(equals(SyllableVocab.padToken)),
            reason: 'Syllable "$syl" should be in vocabulary',
          );
        }
      });
    });
  });
}
