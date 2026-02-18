import 'dart:math' show exp;

import 'package:flutter_test/flutter_test.dart';

/// Tests for V7 scorer helper functions.
///
/// These tests verify the Dart-side logic for:
/// - Softmax computation
/// - CTC alignment scoring algorithm
/// - Tone extraction from pinyin
/// - Syllable-to-character mapping
///
/// Note: Full integration tests require the model_v7.onnx asset.
void main() {
  group('V7 Scorer Helper Functions', () {
    group('Softmax', () {
      test('computes correct softmax for simple input', () {
        final logits = [1.0, 2.0, 3.0];
        final probs = _softmax(logits);

        // Sum should be 1
        expect(probs.reduce((a, b) => a + b), closeTo(1.0, 1e-6));

        // Largest logit -> largest prob
        expect(probs[2], greaterThan(probs[1]));
        expect(probs[1], greaterThan(probs[0]));
      });

      test('handles large logits without overflow', () {
        final logits = [100.0, 200.0, 300.0];
        final probs = _softmax(logits);

        // Should not have NaN or Inf
        for (final p in probs) {
          expect(p.isFinite, true);
        }
        expect(probs.reduce((a, b) => a + b), closeTo(1.0, 1e-6));
      });

      test('handles equal logits', () {
        final logits = [1.0, 1.0, 1.0];
        final probs = _softmax(logits);

        // All should be equal
        expect(probs[0], closeTo(1.0 / 3.0, 1e-6));
        expect(probs[1], closeTo(1.0 / 3.0, 1e-6));
        expect(probs[2], closeTo(1.0 / 3.0, 1e-6));
      });
    });

    group('Tone Extraction', () {
      test('extracts tone 1 from pinyin', () {
        expect(_extractTone('mā'), 1);
        expect(_extractTone('shī'), 1);
        expect(_extractTone('gū'), 1);
      });

      test('extracts tone 2 from pinyin', () {
        expect(_extractTone('má'), 2);
        expect(_extractTone('shí'), 2);
        expect(_extractTone('gú'), 2);
      });

      test('extracts tone 3 from pinyin', () {
        expect(_extractTone('mǎ'), 3);
        expect(_extractTone('shǐ'), 3);
        expect(_extractTone('gǔ'), 3);
      });

      test('extracts tone 4 from pinyin', () {
        expect(_extractTone('mà'), 4);
        expect(_extractTone('shì'), 4);
        expect(_extractTone('gù'), 4);
      });

      test('returns 0 for neutral tone', () {
        expect(_extractTone('ma'), 0);
        expect(_extractTone('de'), 0);
        expect(_extractTone('le'), 0);
      });
    });

    group('Syllable to Character Mapping', () {
      test('maps 1:1 when counts match', () {
        final syllableScores = [0.9, 0.8, 0.7];
        final result = _mapSyllablesToCharacters(syllableScores, 3, 3);

        expect(result, [0.9, 0.8, 0.7]);
      });

      test('expands syllables when more characters', () {
        final syllableScores = [0.9, 0.7];
        final result = _mapSyllablesToCharacters(syllableScores, 4, 2);

        // 4 characters from 2 syllables
        expect(result.length, 4);
        // First two characters map to first syllable
        expect(result[0], 0.9);
        expect(result[1], 0.9);
        // Last two map to second syllable
        expect(result[2], 0.7);
        expect(result[3], 0.7);
      });

      test('truncates when fewer characters', () {
        final syllableScores = [0.9, 0.8, 0.7, 0.6];
        final result = _mapSyllablesToCharacters(syllableScores, 2, 4);

        expect(result.length, 2);
        expect(result[0], 0.9);
        expect(result[1], 0.8);
      });

      test('handles empty input', () {
        final result = _mapSyllablesToCharacters([], 3, 0);
        expect(result, [0.0, 0.0, 0.0]);
      });
    });

    group('Alignment Scoring', () {
      test('finds max probability for each target', () {
        // Simulated probabilities: [time=4, vocab=5]
        final probs = [
          [0.1, 0.8, 0.05, 0.03, 0.02], // Frame 0: target 1 has 0.8
          [0.1, 0.1, 0.7, 0.05, 0.05], // Frame 1: target 2 has 0.7
          [0.1, 0.1, 0.1, 0.6, 0.1], // Frame 2: target 3 has 0.6
          [0.1, 0.2, 0.2, 0.2, 0.3], // Frame 3: mixed
        ];

        final targets = [1, 2, 3];
        final scores = _scoreWithAlignment(probs, targets);

        expect(scores.length, 3);
        expect(scores[0], closeTo(0.8, 0.01));
        expect(scores[1], closeTo(0.7, 0.01));
        expect(scores[2], closeTo(0.6, 0.01));
      });

      test('maintains monotonic frame order', () {
        // Target 1 has max at frame 3, target 2 has max at frame 1
        // But alignment must be monotonic
        final probs = [
          [0.1, 0.2, 0.1, 0.1], // Frame 0
          [0.1, 0.1, 0.9, 0.1], // Frame 1: target 2 has 0.9
          [0.1, 0.3, 0.1, 0.1], // Frame 2
          [0.1, 0.8, 0.1, 0.1], // Frame 3: target 1 has 0.8
        ];

        final targets = [1, 2]; // Target 1 first, then target 2
        final result = _scoreWithAlignmentWithFrames(probs, targets);

        // Frame for target 1 must come before frame for target 2
        expect(result.frames[0], lessThan(result.frames[1]));
      });
    });
  });
}

// Helper function copies from scorer (for testing)
List<double> _softmax(List<double> logits) {
  final maxLogit = logits.reduce((a, b) => a > b ? a : b);
  final exps = logits.map((x) => exp(x - maxLogit)).toList();
  final sumExps = exps.reduce((a, b) => a + b);
  return exps.map((x) => x / sumExps).toList();
}

int _extractTone(String syllable) {
  const toneMap = {
    'ā': 1, 'á': 2, 'ǎ': 3, 'à': 4,
    'ē': 1, 'é': 2, 'ě': 3, 'è': 4,
    'ī': 1, 'í': 2, 'ǐ': 3, 'ì': 4,
    'ō': 1, 'ó': 2, 'ǒ': 3, 'ò': 4,
    'ū': 1, 'ú': 2, 'ǔ': 3, 'ù': 4,
    'ǖ': 1, 'ǘ': 2, 'ǚ': 3, 'ǜ': 4,
  };
  for (final c in syllable.split('')) {
    if (toneMap.containsKey(c)) {
      return toneMap[c]!;
    }
  }
  return 0;
}

List<double> _mapSyllablesToCharacters(
  List<double> syllableScores,
  int characterCount,
  int syllableCount,
) {
  if (syllableScores.isEmpty) {
    return List.filled(characterCount, 0.0);
  }

  if (syllableCount == characterCount) {
    return syllableScores;
  }

  if (characterCount > syllableCount) {
    final charScores = <double>[];
    for (int i = 0; i < characterCount; i++) {
      final syllableIdx = (i * syllableCount / characterCount).floor();
      charScores.add(syllableScores[syllableIdx]);
    }
    return charScores;
  }

  return syllableScores.sublist(0, characterCount);
}

List<double> _scoreWithAlignment(
  List<List<double>> probs,
  List<int> targetIds,
) {
  final nFrames = probs.length;
  final nTargets = targetIds.length;

  if (nFrames == 0 || nTargets == 0) {
    return List<double>.filled(nTargets, 0.0);
  }

  final scores = <double>[];
  int minFrame = 0;

  for (int i = 0; i < nTargets; i++) {
    final targetId = targetIds[i];
    final maxSearchFrame = minFrame + ((nFrames - minFrame) ~/ (nTargets - i).clamp(1, nTargets));
    final searchEnd = maxSearchFrame.clamp(minFrame + 1, nFrames);

    double bestScore = 0.0;
    int bestFrame = minFrame;

    for (int t = minFrame; t < searchEnd; t++) {
      if (targetId < probs[t].length) {
        final score = probs[t][targetId];
        if (score > bestScore) {
          bestScore = score;
          bestFrame = t;
        }
      }
    }

    scores.add(bestScore);
    minFrame = bestFrame + 1;
  }

  return scores;
}

class _AlignmentResult {
  final List<double> scores;
  final List<int> frames;
  _AlignmentResult(this.scores, this.frames);
}

_AlignmentResult _scoreWithAlignmentWithFrames(
  List<List<double>> probs,
  List<int> targetIds,
) {
  final nFrames = probs.length;
  final nTargets = targetIds.length;

  final scores = <double>[];
  final frames = <int>[];
  int minFrame = 0;

  for (int i = 0; i < nTargets; i++) {
    final targetId = targetIds[i];
    final maxSearchFrame = minFrame + ((nFrames - minFrame) ~/ (nTargets - i).clamp(1, nTargets));
    final searchEnd = maxSearchFrame.clamp(minFrame + 1, nFrames);

    double bestScore = 0.0;
    int bestFrame = minFrame;

    for (int t = minFrame; t < searchEnd; t++) {
      if (targetId < probs[t].length) {
        final score = probs[t][targetId];
        if (score > bestScore) {
          bestScore = score;
          bestFrame = t;
        }
      }
    }

    scores.add(bestScore);
    frames.add(bestFrame);
    minFrame = bestFrame + 1;
  }

  return _AlignmentResult(scores, frames);
}
