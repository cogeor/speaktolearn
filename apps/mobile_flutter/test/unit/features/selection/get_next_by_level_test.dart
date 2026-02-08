import 'dart:math';

import 'package:flutter_test/flutter_test.dart';
import 'package:speak_to_learn/features/selection/domain/get_next_by_level.dart';
import 'package:speak_to_learn/features/text_sequences/domain/text_sequence.dart';

import '../../../mocks/mock_repositories.dart';

/// A seeded Random for deterministic testing.
class _SeededRandom implements Random {
  _SeededRandom(this._values);

  final List<int> _values;
  int _index = 0;

  @override
  int nextInt(int max) {
    final value = _values[_index % _values.length];
    _index++;
    return value % max;
  }

  @override
  double nextDouble() => 0.5;

  @override
  bool nextBool() => true;
}

void main() {
  const level1Seq1 = TextSequence(id: 'l1-001', text: '你好', language: 'zh', hskLevel: 1);
  const level1Seq2 = TextSequence(id: 'l1-002', text: '谢谢', language: 'zh', hskLevel: 1);
  const level1Seq3 = TextSequence(id: 'l1-003', text: '再见', language: 'zh', hskLevel: 1);
  const level2Seq1 = TextSequence(id: 'l2-001', text: '学习', language: 'zh', hskLevel: 2);

  group('GetNextByLevel', () {
    test('returns null when no sequences exist for level', () async {
      final repo = MockTextSequenceRepository([level2Seq1]);
      final useCase = GetNextByLevel(textSequenceRepository: repo);

      final result = await useCase(level: 1);

      expect(result, isNull);
    });

    test('returns a sequence from the requested level', () async {
      final repo = MockTextSequenceRepository([level1Seq1, level1Seq2, level2Seq1]);
      final useCase = GetNextByLevel(
        textSequenceRepository: repo,
        random: _SeededRandom([0]),
      );

      final result = await useCase(level: 1);

      expect(result, isNotNull);
      expect(result!.hskLevel, equals(1));
    });

    test('excludes currentId from selection', () async {
      final repo = MockTextSequenceRepository([level1Seq1, level1Seq2]);
      final useCase = GetNextByLevel(
        textSequenceRepository: repo,
        random: _SeededRandom([0]),
      );

      final result = await useCase(level: 1, currentId: 'l1-001');

      expect(result, isNotNull);
      expect(result!.id, equals('l1-002'));
    });

    test('returns currentId sequence when it is the only one', () async {
      final repo = MockTextSequenceRepository([level1Seq1]);
      final useCase = GetNextByLevel(textSequenceRepository: repo);

      final result = await useCase(level: 1, currentId: 'l1-001');

      expect(result, isNotNull);
      expect(result!.id, equals('l1-001'));
    });

    test('selects randomly from available sequences', () async {
      final repo = MockTextSequenceRepository([level1Seq1, level1Seq2, level1Seq3]);

      // First call returns index 0
      final useCase1 = GetNextByLevel(
        textSequenceRepository: repo,
        random: _SeededRandom([0]),
      );
      final result1 = await useCase1(level: 1);
      expect(result1!.id, equals('l1-001'));

      // Second call returns index 1
      final useCase2 = GetNextByLevel(
        textSequenceRepository: repo,
        random: _SeededRandom([1]),
      );
      final result2 = await useCase2(level: 1);
      expect(result2!.id, equals('l1-002'));

      // Third call returns index 2
      final useCase3 = GetNextByLevel(
        textSequenceRepository: repo,
        random: _SeededRandom([2]),
      );
      final result3 = await useCase3(level: 1);
      expect(result3!.id, equals('l1-003'));
    });

    test('does not return sequences from other levels', () async {
      final repo = MockTextSequenceRepository([level1Seq1, level2Seq1]);
      final useCase = GetNextByLevel(
        textSequenceRepository: repo,
        random: _SeededRandom([0]),
      );

      final result = await useCase(level: 2);

      expect(result, isNotNull);
      expect(result!.id, equals('l2-001'));
      expect(result.hskLevel, equals(2));
    });
  });
}
