import 'package:flutter_test/flutter_test.dart';

import 'package:speak_to_learn/features/scoring/data/asr_similarity_scorer.dart';
import 'package:speak_to_learn/features/scoring/data/cer_calculator.dart';
import 'package:speak_to_learn/features/scoring/data/cer_result.dart';
import 'package:speak_to_learn/features/scoring/data/speech_recognizer.dart';
import 'package:speak_to_learn/features/recording/domain/recording.dart';
import 'package:speak_to_learn/features/text_sequences/domain/text_sequence.dart';
import 'package:speak_to_learn/core/utils/string_utils.dart';

void main() {
  group('CerCalculator ground truth tests', () {
    late CerCalculator calculator;

    setUp(() {
      calculator = CerCalculator();
    });

    test('perfect match scores 100', () {
      final result = calculator.calculate('你好', '你好');

      expect(result.score, 100);
      expect(result.cer, 0.0);
      expect(result.accuracy, 100);
      expect(result.completeness, 100);
    });

    test('perfect match ignoring punctuation scores 100', () {
      // Ground truth might have punctuation, recognition might not
      final result = calculator.calculate('你好！', '你好');

      expect(result.score, 100);
    });

    test('off by one character scores high', () {
      // 你好 (2 chars) vs 你 (1 char) = 1 edit distance
      // CER = 1/2 = 0.5, score = 50
      final result = calculator.calculate('你好', '你');

      expect(result.score, 50);
    });

    test('completely wrong scores 0', () {
      // 你好 (2 chars) vs something totally different
      final result = calculator.calculate('你好', '北京上海');

      expect(result.score, 0);
    });

    test('empty hypothesis scores 0', () {
      final result = calculator.calculate('你好', '');

      expect(result.score, 0);
    });

    test('empty reference with empty hypothesis scores 100', () {
      final result = calculator.calculate('', '');

      expect(result.score, 100);
    });

    test('long sentence ground truth match scores 100', () {
      // ts_000014: 我们一起去公园吧
      const groundTruth = '我们一起去公园吧';
      final result = calculator.calculate(groundTruth, groundTruth);

      expect(result.score, 100);
      expect(result.accuracy, 100);
    });

    test('partial recognition of long sentence scores proportionally', () {
      // ts_000014: 我们一起去公园吧 (8 chars)
      // If only "我们一起" (4 chars) recognized: 4 errors, CER = 4/8 = 0.5
      const groundTruth = '我们一起去公园吧';
      const recognized = '我们一起';
      final result = calculator.calculate(groundTruth, recognized);

      expect(result.score, 50);
    });

    test('minor misrecognition scores high', () {
      // ts_000001: 你好 (2 chars)
      // Similar sound: 里好 (1 error), CER = 1/2 = 0.5, score = 50
      final result = calculator.calculate('你好', '里好');

      expect(result.score, 50);
    });
  });

  group('CerResult scoring formula', () {
    test('CER 0 gives score 100', () {
      final result = CerResult(
        cer: 0.0,
        referenceLength: 2,
        hypothesisLength: 2,
        editDistance: 0,
      );

      expect(result.score, 100);
    });

    test('CER 0.5 gives score 50', () {
      final result = CerResult(
        cer: 0.5,
        referenceLength: 2,
        hypothesisLength: 1,
        editDistance: 1,
      );

      expect(result.score, 50);
    });

    test('CER 1.0 gives score 0', () {
      final result = CerResult(
        cer: 1.0,
        referenceLength: 2,
        hypothesisLength: 0,
        editDistance: 2,
      );

      expect(result.score, 0);
    });

    test('CER > 1.0 gives score 0', () {
      // Happens when hypothesis is much longer than reference
      final result = CerResult(
        cer: 1.5,
        referenceLength: 2,
        hypothesisLength: 5,
        editDistance: 3,
      );

      expect(result.score, 0);
    });
  });

  group('normalizeZhText', () {
    test('removes Chinese punctuation', () {
      expect(normalizeZhText('你好！'), '你好');
      expect(normalizeZhText('你好。'), '你好');
      expect(normalizeZhText('你好？'), '你好');
    });

    test('removes spaces', () {
      expect(normalizeZhText('你 好'), '你好');
      expect(normalizeZhText(' 你好 '), '你好');
    });

    test('normalizes identical text', () {
      const original = '你好';
      expect(normalizeZhText(original), normalizeZhText(original));
    });
  });

  group('AsrSimilarityScorer with MockSpeechRecognizer', () {
    test('perfect recognition returns score 100', () async {
      const expectedText = '你好';
      final mockRecognizer = MockSpeechRecognizer(
        defaultResponse: expectedText,
      );
      final scorer = AsrSimilarityScorer(recognizer: mockRecognizer);

      final sequence = TextSequence(
        id: 'ts_000001',
        text: expectedText,
        language: 'zh-CN',
      );
      final recording = Recording(
        id: 'test_recording',
        textSequenceId: sequence.id,
        createdAt: DateTime.now(),
        filePath: '/test/path.m4a',
      );

      final grade = await scorer.score(sequence, recording);

      expect(grade.overall, 100);
      expect(grade.accuracy, 100);
      expect(grade.recognizedText, expectedText);
    });

    test('no speech detected returns score 0', () async {
      final mockRecognizer = MockSpeechRecognizer(
        shouldFail: true,
        failureError: RecognitionError.noSpeechDetected,
      );
      final scorer = AsrSimilarityScorer(recognizer: mockRecognizer);

      final sequence = TextSequence(
        id: 'ts_000001',
        text: '你好',
        language: 'zh-CN',
      );
      final recording = Recording(
        id: 'test_recording',
        textSequenceId: sequence.id,
        createdAt: DateTime.now(),
        filePath: '/test/path.m4a',
      );

      final grade = await scorer.score(sequence, recording);

      expect(grade.overall, 0);
      expect(grade.details?['error'], 'noSpeechDetected');
    });

    test('partial recognition returns proportional score', () async {
      // Ground truth: 我们一起去公园吧 (8 chars)
      // Recognized: 我们一起 (4 chars) - missing 4 chars
      const groundTruth = '我们一起去公园吧';
      const recognized = '我们一起';

      final mockRecognizer = MockSpeechRecognizer(defaultResponse: recognized);
      final scorer = AsrSimilarityScorer(recognizer: mockRecognizer);

      final sequence = TextSequence(
        id: 'ts_000014',
        text: groundTruth,
        language: 'zh-CN',
      );
      final recording = Recording(
        id: 'test_recording',
        textSequenceId: sequence.id,
        createdAt: DateTime.now(),
        filePath: '/test/path.m4a',
      );

      final grade = await scorer.score(sequence, recording);

      // CER = 4/8 = 0.5, score = 50
      expect(grade.overall, 50);
    });

    test(
      'high quality recognition with punctuation difference scores 100',
      () async {
        // Ground truth has punctuation, recognition doesn't
        const groundTruth = '你好！';
        const recognized = '你好';

        final mockRecognizer = MockSpeechRecognizer(
          defaultResponse: recognized,
        );
        final scorer = AsrSimilarityScorer(recognizer: mockRecognizer);

        final sequence = TextSequence(
          id: 'ts_000001',
          text: groundTruth,
          language: 'zh-CN',
        );
        final recording = Recording(
          id: 'test_recording',
          textSequenceId: sequence.id,
          createdAt: DateTime.now(),
          filePath: '/test/path.m4a',
        );

        final grade = await scorer.score(sequence, recording);

        // After normalization, both are "你好", so score = 100
        expect(grade.overall, 100);
      },
    );
  });

  group('Ground truth sentences from dataset', () {
    // These tests verify the expected ground truth sentences produce 100 scores
    // when the speech recognizer returns exact matches

    final groundTruthSentences = {
      'ts_000001': '你好',
      'ts_000002': '谢谢',
      'ts_000003': '我爱你',
      'ts_000004': '早上好',
      'ts_000005': '晚安',
      'ts_000006': '我很高兴',
      'ts_000007': '你在哪里',
      'ts_000008': '我饿了',
      'ts_000009': '请坐',
      'ts_000010': '他是谁',
    };

    for (final entry in groundTruthSentences.entries) {
      test(
        '${entry.key}: "${entry.value}" scores 100 on exact match',
        () async {
          final mockRecognizer = MockSpeechRecognizer(
            defaultResponse: entry.value,
          );
          final scorer = AsrSimilarityScorer(recognizer: mockRecognizer);

          final sequence = TextSequence(
            id: entry.key,
            text: entry.value,
            language: 'zh-CN',
          );
          final recording = Recording(
            id: 'test_recording',
            textSequenceId: sequence.id,
            createdAt: DateTime.now(),
            filePath: '/test/path.m4a',
          );

          final grade = await scorer.score(sequence, recording);

          expect(
            grade.overall,
            100,
            reason: 'Sentence "${entry.value}" should score 100 on exact match',
          );
        },
      );
    }
  });
}
