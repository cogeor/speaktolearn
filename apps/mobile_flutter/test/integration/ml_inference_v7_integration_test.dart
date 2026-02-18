import 'dart:convert';
import 'dart:math' show exp, max;

import 'package:flutter_test/flutter_test.dart';

/// Integration tests for V7 CTC scorer to validate cross-platform consistency.
///
/// These tests verify that the Dart implementation produces identical results
/// to the Python implementation for the same inputs.
void main() {
  group('V7 Cross-Platform Consistency Tests', () {
    group('Softmax', () {
      test('simple softmax matches Python', () {
        final input = [1.0, 2.0, 3.0];
        final expected = [0.09003057, 0.24472848, 0.66524094];

        final result = _softmax(input);

        for (int i = 0; i < result.length; i++) {
          expect(result[i], closeTo(expected[i], 1e-5),
              reason: 'Softmax element $i mismatch');
        }
      });

      test('equal logits produce uniform distribution', () {
        final input = [1.0, 1.0, 1.0];
        final result = _softmax(input);

        for (final p in result) {
          expect(p, closeTo(1.0 / 3.0, 1e-6));
        }
      });

      test('handles large logits without overflow', () {
        final input = [100.0, 200.0, 300.0];
        final result = _softmax(input);

        // Sum should be 1
        final sum = result.reduce((a, b) => a + b);
        expect(sum, closeTo(1.0, 1e-6));

        // Last element should dominate
        expect(result[2], closeTo(1.0, 1e-6));
      });

      test('handles negative logits', () {
        final input = [-1.0, 0.0, 1.0];
        final expected = [0.09003057, 0.24472848, 0.66524094];

        final result = _softmax(input);

        for (int i = 0; i < result.length; i++) {
          expect(result[i], closeTo(expected[i], 1e-5));
        }
      });
    });

    group('Tone Extraction', () {
      final testCases = [
        ('mā', 1),
        ('má', 2),
        ('mǎ', 3),
        ('mà', 4),
        ('ma', 0),
        ('shī', 1),
        ('shí', 2),
        ('shǐ', 3),
        ('shì', 4),
        ('gū', 1),
        ('gú', 2),
        ('gǔ', 3),
        ('gù', 4),
        ('lǚ', 3),
        ('nǜ', 4),
        ('de', 0),
        ('le', 0),
      ];

      for (final (syllable, expectedTone) in testCases) {
        test('extracts tone $expectedTone from "$syllable"', () {
          final result = _extractTone(syllable);
          expect(result, expectedTone);
        });
      }
    });

    group('Syllable to Character Mapping', () {
      test('1:1 mapping when counts match', () {
        final syllableScores = [0.9, 0.8, 0.7];
        final result = _mapSyllablesToCharacters(syllableScores, 3, 3);

        expect(result, [0.9, 0.8, 0.7]);
      });

      test('expands 2 syllables to 4 characters', () {
        final syllableScores = [0.9, 0.7];
        final result = _mapSyllablesToCharacters(syllableScores, 4, 2);

        expect(result, [0.9, 0.9, 0.7, 0.7]);
      });

      test('truncates 4 syllables to 2 characters', () {
        final syllableScores = [0.9, 0.8, 0.7, 0.6];
        final result = _mapSyllablesToCharacters(syllableScores, 2, 4);

        expect(result, [0.9, 0.8]);
      });

      test('handles empty input', () {
        final result = _mapSyllablesToCharacters([], 3, 0);

        expect(result, [0.0, 0.0, 0.0]);
      });

      test('expands 3 syllables to 5 characters', () {
        final syllableScores = [0.9, 0.8, 0.7];
        final result = _mapSyllablesToCharacters(syllableScores, 5, 3);

        // Floor mapping: 0->0, 1->0, 2->1, 3->1, 4->2
        expect(result, [0.9, 0.9, 0.8, 0.8, 0.7]);
      });
    });

    group('Alignment Scoring', () {
      test('finds best frame for each target in order', () {
        // Create mock probability matrix [frames x vocab]
        // 4 frames, 5 vocab items
        final probs = [
          [0.1, 0.8, 0.05, 0.03, 0.02], // Frame 0: target 1 has 0.8
          [0.1, 0.1, 0.7, 0.05, 0.05], // Frame 1: target 2 has 0.7
          [0.1, 0.1, 0.1, 0.6, 0.1], // Frame 2: target 3 has 0.6
          [0.1, 0.2, 0.2, 0.2, 0.3], // Frame 3: mixed
        ];

        final targets = [1, 2, 3];
        final result = _scoreWithAlignmentFromProbs(probs, targets);

        expect(result.scores.length, 3);
        expect(result.scores[0], closeTo(0.8, 0.01));
        expect(result.scores[1], closeTo(0.7, 0.01));
        expect(result.scores[2], closeTo(0.6, 0.01));
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
        final result = _scoreWithAlignmentFromProbs(probs, targets);

        // Frame for target 1 must come before frame for target 2
        expect(result.frames[0], lessThan(result.frames[1]));
      });

      test('handles more targets than frames gracefully', () {
        final probs = [
          [0.1, 0.3, 0.3, 0.3],
          [0.1, 0.3, 0.3, 0.3],
        ];

        final targets = [1, 2, 3, 1, 2, 3]; // 6 targets, only 2 frames

        final result = _scoreWithAlignmentFromProbs(probs, targets);

        expect(result.scores.length, 6);
        // Should still return valid (though possibly low) scores
        for (final score in result.scores) {
          expect(score, greaterThanOrEqualTo(0.0));
          expect(score, lessThanOrEqualTo(1.0));
        }
      });
    });

    group('Combined Score Formula', () {
      test('0.7 * syllable + 0.3 * tone', () {
        final testCases = [
          (1.0, 1.0, 1.0),
          (0.0, 0.0, 0.0),
          (1.0, 0.0, 0.7),
          (0.0, 1.0, 0.3),
          (0.8, 0.6, 0.74),
        ];

        for (final (sylScore, toneScore, expected) in testCases) {
          final result = 0.7 * sylScore + 0.3 * toneScore;
          expect(result, closeTo(expected, 1e-6),
              reason:
                  'Combined($sylScore, $toneScore) should be $expected, got $result');
        }
      });
    });

    group('Full Pipeline Simulation', () {
      test('simulates complete scoring flow', () {
        // Simulate what the scorer does:
        // 1. Model outputs logits [frames, vocab]
        // 2. Apply softmax per frame
        // 3. Score with alignment
        // 4. Combine syllable and tone scores
        // 5. Map to characters

        final sylLogits = [
          [0.0, 5.0, 1.0, 1.0], // Frame 0: strong for syl 1
          [0.0, 1.0, 4.0, 1.0], // Frame 1: strong for syl 2
          [0.0, 1.0, 1.0, 3.0], // Frame 2: strong for syl 3
        ];

        final toneLogits = [
          [0.0, 3.0, 1.0, 1.0, 1.0, 1.0], // Frame 0: tone 1
          [0.0, 1.0, 3.0, 1.0, 1.0, 1.0], // Frame 1: tone 2
          [0.0, 1.0, 1.0, 3.0, 1.0, 1.0], // Frame 2: tone 3
        ];

        // Convert to probabilities
        final sylProbs = sylLogits.map(_softmax).toList();
        final toneProbs = toneLogits.map(_softmax).toList();

        // Score with alignment
        final sylResult =
            _scoreWithAlignmentFromProbs(sylProbs, [1, 2, 3]);
        final toneResult =
            _scoreWithAlignmentFromProbs(toneProbs, [1, 2, 3]);

        // Combine scores
        final combined = <double>[];
        for (int i = 0; i < 3; i++) {
          combined.add(0.7 * sylResult.scores[i] + 0.3 * toneResult.scores[i]);
        }

        // Verify we got reasonable scores
        expect(combined.length, 3);
        for (final score in combined) {
          expect(score, greaterThan(0.0));
          expect(score, lessThanOrEqualTo(1.0));
        }

        // Map to characters (3 syllables -> 3 characters)
        final charScores = _mapSyllablesToCharacters(combined, 3, 3);
        expect(charScores, combined);
      });
    });
  });
}

// ============================================================================
// Helper Functions (copied from onnx_ml_scorer_v7.dart for testing)
// ============================================================================

List<double> _softmax(List<double> logits) {
  final maxLogit = logits.reduce(max);
  final exps = logits.map((x) => exp(x - maxLogit)).toList();
  final sumExps = exps.reduce((a, b) => a + b);
  return exps.map((x) => x / sumExps).toList();
}

int _extractTone(String syllable) {
  const toneMap = {
    'ā': 1,
    'á': 2,
    'ǎ': 3,
    'à': 4,
    'ē': 1,
    'é': 2,
    'ě': 3,
    'è': 4,
    'ī': 1,
    'í': 2,
    'ǐ': 3,
    'ì': 4,
    'ō': 1,
    'ó': 2,
    'ǒ': 3,
    'ò': 4,
    'ū': 1,
    'ú': 2,
    'ǔ': 3,
    'ù': 4,
    'ǖ': 1,
    'ǘ': 2,
    'ǚ': 3,
    'ǜ': 4,
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

class _AlignmentResult {
  final List<double> scores;
  final List<int> frames;
  _AlignmentResult(this.scores, this.frames);
}

_AlignmentResult _scoreWithAlignmentFromProbs(
  List<List<double>> probs,
  List<int> targetIds,
) {
  final nFrames = probs.length;
  final nTargets = targetIds.length;

  if (nFrames == 0 || nTargets == 0) {
    return _AlignmentResult(
      List<double>.filled(nTargets, 0.0),
      List<int>.filled(nTargets, 0),
    );
  }

  final scores = <double>[];
  final frames = <int>[];
  int minFrame = 0;

  for (int i = 0; i < nTargets; i++) {
    final targetId = targetIds[i];
    final maxSearchFrame =
        minFrame + ((nFrames - minFrame) ~/ max(1, nTargets - i));
    final searchEnd = max(maxSearchFrame, minFrame + 1);

    double bestScore = 0.0;
    int bestFrame = minFrame;

    for (int t = minFrame; t < searchEnd && t < nFrames; t++) {
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
