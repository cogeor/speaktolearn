import 'dart:math';

import 'package:characters/characters.dart';

import '../../recording/domain/recording.dart';
import '../../scoring/domain/grade.dart';
import '../../text_sequences/domain/text_sequence.dart';
import '../domain/ml_scorer.dart';

/// Mock ML scorer for testing and debugging.
///
/// Generates random character scores to simulate ML model output.
/// Replace with [OnnxMlScorer] for production use.
class MockMlScorer implements MlScorer {
  MockMlScorer({Random? random}) : _random = random ?? Random();

  final Random _random;
  bool _initialized = false;

  static const _method = 'mock_ml_v1';

  @override
  bool get isReady => _initialized;

  @override
  Future<void> initialize() async {
    // Simulate model loading delay
    await Future.delayed(const Duration(milliseconds: 100));
    _initialized = true;
  }

  @override
  Future<void> dispose() async {
    _initialized = false;
  }

  @override
  Future<Grade> score(TextSequence sequence, Recording recording) async {
    if (!_initialized) {
      await initialize();
    }

    // Get character count using proper Unicode handling
    final characters = sequence.text.characters.toList();
    final charCount = characters.length;

    // Generate random scores for each character (0.0-1.0)
    final characterScores = List.generate(
      charCount,
      (_) => _random.nextDouble(),
    );

    // Overall score is average of character scores (scaled to 0-100)
    final avgScore = characterScores.isEmpty
        ? 0.0
        : characterScores.reduce((a, b) => a + b) / characterScores.length;
    final overall = (avgScore * 100).round();

    return Grade(
      overall: overall,
      method: _method,
      characterScores: characterScores,
      details: {'charCount': charCount, 'avgScore': avgScore},
    );
  }
}
