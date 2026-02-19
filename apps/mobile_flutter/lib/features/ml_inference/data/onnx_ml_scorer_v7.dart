import 'dart:io';
import 'dart:math' show exp, max, min;
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:onnxruntime/onnxruntime.dart';
import 'package:characters/characters.dart';

import '../../recording/domain/recording.dart';
import '../../scoring/domain/grade.dart';
import '../../text_sequences/domain/text_sequence.dart';
import '../domain/ml_scorer.dart';
import 'mel_extractor.dart';
import 'syllable_vocab.dart';

/// ONNX-based ML scorer for pronunciation assessment (V7 CTC architecture).
///
/// Uses the V7 model with per-frame CTC outputs for syllables and tones.
/// Single forward pass scores all syllables (unlike V6 which needed N passes).
///
/// V7 Model Input:
/// - mel: [1, 80, time] - variable length mel spectrogram
/// - audio_mask: [1, time] - padding mask (true = padded)
///
/// V7 Model Output:
/// - syllable_logits: [1, time//4, n_syllables+1] - per-frame CTC logits
/// - tone_logits: [1, time//4, n_tones+1] - per-frame CTC logits
///
/// Key differences from V6:
/// - Single forward pass instead of N passes per syllable
/// - CTC decoding or alignment-based scoring
/// - No position input needed
/// - Blank token at index 0 for both outputs
class OnnxMlScorerV7 implements MlScorer {
  OrtSession? _session;
  SyllableVocab? _vocab;
  final MelExtractor _melExtractor = MelExtractor();
  bool _isReady = false;

  static const _method = 'onnx_v7';
  static const _modelPath = 'assets/models/model_v7.onnx';

  // V7 supports variable length, but cap for safety
  static const _maxMelFrames = 1500;

  @override
  bool get isReady => _isReady;

  @override
  Future<void> initialize() async {
    if (_isReady) return;

    try {
      // Load syllable vocab
      _vocab = await SyllableVocab.load();

      // Load ONNX model from assets
      final modelBytes = await rootBundle.load(_modelPath);
      final modelData = modelBytes.buffer.asUint8List();

      // Create session
      final sessionOptions = OrtSessionOptions();
      _session = OrtSession.fromBuffer(modelData, sessionOptions);

      _isReady = true;
    } catch (e) {
      _isReady = false;
      rethrow;
    }
  }

  @override
  Future<Grade> score(TextSequence sequence, Recording recording) async {
    final totalStopwatch = Stopwatch()..start();

    if (!_isReady) {
      await initialize();
    }

    try {
      // 1. Load audio from recording file
      var stepWatch = Stopwatch()..start();
      final audioFile = File(recording.filePath);
      final audioBytes = await audioFile.readAsBytes();
      final audioSamples = _parseWavToSamples(audioBytes);
      print(
        '⏱️ [V7] Audio load: ${stepWatch.elapsedMilliseconds}ms (${audioSamples.length} samples)',
      );

      // 2. Get pinyin syllables from sequence
      var syllables = _parsePinyin(sequence.romanization ?? '');

      if (syllables.isEmpty) {
        return _fallbackScore(sequence);
      }

      // 3. Extract mel from FULL audio
      stepWatch.reset();
      stepWatch.start();
      final mel = _melExtractor.extract(audioSamples);
      final melFrames = mel[0].length;
      print(
        '⏱️ [V7] Mel extraction: ${stepWatch.elapsedMilliseconds}ms ($melFrames frames)',
      );

      // 4. Run SINGLE inference pass
      stepWatch.reset();
      stepWatch.start();
      final inferenceResult = await _runInference(mel, syllables);
      print(
        '⏱️ [V7] Inference: ${stepWatch.elapsedMilliseconds}ms (single pass for ${syllables.length} syllables)',
      );

      // 5. Score each syllable using alignment-based scoring
      final syllableScores = inferenceResult.syllableScores;
      final toneScores = inferenceResult.toneScores;

      // Combined scores: 70% syllable + 30% tone
      final combinedScores = <double>[];
      for (int i = 0; i < syllables.length; i++) {
        final sylScore = i < syllableScores.length ? syllableScores[i] : 0.0;
        final toneScore = i < toneScores.length ? toneScores[i] : 1.0;
        combinedScores.add(0.7 * sylScore + 0.3 * toneScore);
      }

      // 6. Map syllable scores to character scores
      final characters = sequence.text.characters.toList();
      final characterScores = _mapSyllablesToCharacters(
        combinedScores,
        characters.length,
        syllables.length,
      );

      // 7. Compute overall grade
      final avgScore = characterScores.isEmpty
          ? 0.0
          : characterScores.reduce((a, b) => a + b) / characterScores.length;

      totalStopwatch.stop();
      print('⏱️ [V7] TOTAL scoring: ${totalStopwatch.elapsedMilliseconds}ms');
      print(
        '📊 [V7] Scores: ${combinedScores.map((s) => s.toStringAsFixed(3)).join(", ")}',
      );

      return Grade(
        overall: (avgScore * 100).round(),
        method: _method,
        characterScores: characterScores,
        details: {
          'syllableCount': syllables.length,
          'characterCount': characters.length,
          'avgScore': avgScore,
          'melFrames': melFrames,
          'outputFrames': melFrames ~/ 4,
        },
      );
    } catch (e) {
      print('❌ [V7] Scoring error: $e');
      return _fallbackScore(sequence);
    }
  }

  /// Inference result containing per-syllable scores
  Future<_InferenceResult> _runInference(
    List<List<double>> mel,
    List<String> targetSyllables,
  ) async {
    if (_session == null || _vocab == null) {
      throw StateError('Scorer not initialized');
    }

    final melFrames = mel[0].length;
    final timeFrames = min(melFrames, _maxMelFrames);

    // Prepare mel tensor: [1, 80, time]
    final melFlat = Float32List(80 * timeFrames);
    for (int i = 0; i < 80; i++) {
      for (int t = 0; t < timeFrames; t++) {
        melFlat[i * timeFrames + t] = mel[i][t];
      }
    }

    // Audio mask: false for all frames (no padding in input)
    final audioMask = List<bool>.filled(timeFrames, false);

    final melTensor = OrtValueTensor.createTensorWithDataList(melFlat, [
      1,
      80,
      timeFrames,
    ]);
    final audioMaskTensor = OrtValueTensor.createTensorWithDataList(audioMask, [
      1,
      timeFrames,
    ]);

    List<OrtValue?>? outputs;
    OrtRunOptions? runOptions;

    try {
      final inputs = {'mel': melTensor, 'audio_mask': audioMaskTensor};

      runOptions = OrtRunOptions();
      outputs = await _session!.runAsync(runOptions, inputs, [
        'syllable_logits',
        'tone_logits',
      ]);

      if (outputs == null || outputs.length < 2) {
        throw StateError('Model did not return expected outputs');
      }

      // Extract logits: [1, time//4, vocab_size]
      final sylLogitsData = outputs[0]!.value as List;
      final toneLogitsData = outputs[1]!.value as List;

      // sylLogitsData is [[[frame0], [frame1], ...]] - 3D
      final sylFrames = (sylLogitsData[0] as List);
      final toneFrames = (toneLogitsData[0] as List);

      final outputFrames = sylFrames.length;

      // Convert to proper types
      final sylLogits = <List<double>>[];
      final toneLogits = <List<double>>[];

      for (int t = 0; t < outputFrames; t++) {
        sylLogits.add((sylFrames[t] as List).cast<double>());
        toneLogits.add((toneFrames[t] as List).cast<double>());
      }

      // Score using alignment-based method
      final syllableScores = _scoreWithAlignment(
        sylLogits,
        targetSyllables,
        isForSyllables: true,
      );
      final toneScores = _scoreWithAlignment(
        toneLogits,
        targetSyllables,
        isForSyllables: false,
      );

      return _InferenceResult(
        syllableScores: syllableScores,
        toneScores: toneScores,
        outputFrames: outputFrames,
      );
    } finally {
      outputs?.forEach((output) => output?.release());
      runOptions?.release();
      melTensor.release();
      audioMaskTensor.release();
    }
  }

  /// Alignment-based scoring: find max probability for each target in sequence order
  List<double> _scoreWithAlignment(
    List<List<double>> logits,
    List<String> targetSyllables, {
    required bool isForSyllables,
  }) {
    final nFrames = logits.length;
    final nTargets = targetSyllables.length;

    if (nFrames == 0 || nTargets == 0) {
      return List<double>.filled(nTargets, 0.0);
    }

    // Apply softmax to each frame
    final probs = logits.map((frame) => _softmax(frame)).toList();

    final scores = <double>[];
    int minFrame = 0;

    for (int i = 0; i < nTargets; i++) {
      final targetSyl = targetSyllables[i];

      int targetId;
      if (isForSyllables) {
        // Get CTC syllable ID: vocab returns 2+ for syllables, CTC has blank=0, so shift
        final vocabId = _vocab!.encode(targetSyl);
        targetId = max(1, vocabId - 1); // Ensure >= 1 (blank=0)
      } else {
        // Tone ID: extract tone 1-4/0 -> CTC 1-5
        final tone = _extractTone(targetSyl);
        targetId = tone + 1; // Shift by 1 for blank
      }

      // Search from minFrame to proportional window
      final maxSearchFrame = min(
        nFrames,
        minFrame + ((nFrames - minFrame) ~/ max(1, nTargets - i)),
      );
      final searchEnd = max(maxSearchFrame, minFrame + 1);

      // Find best frame for this target
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
      minFrame = bestFrame + 1; // Next target must come after
    }

    return scores;
  }

  /// Extract tone number from pinyin syllable.
  static int _extractTone(String syllable) {
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
    return 0; // Neutral tone
  }

  List<double> _softmax(List<double> logits) {
    final maxLogit = logits.reduce(max);
    final exps = logits.map((x) => exp(x - maxLogit)).toList();
    final sumExps = exps.reduce((a, b) => a + b);
    return exps.map((x) => x / sumExps).toList();
  }

  List<double> _parseWavToSamples(Uint8List wavBytes) {
    const headerSize = 44;
    if (wavBytes.length < headerSize) {
      return [];
    }

    final pcmBytes = wavBytes.sublist(headerSize);
    final samples = <double>[];

    for (int i = 0; i < pcmBytes.length - 1; i += 2) {
      final sample16 = pcmBytes[i] | (pcmBytes[i + 1] << 8);
      final signedSample = sample16 > 32767 ? sample16 - 65536 : sample16;
      samples.add(signedSample / 32768.0);
    }

    return samples;
  }

  List<String> _parsePinyin(String romanization) {
    return romanization.trim().split(' ').where((s) => s.isNotEmpty).toList();
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

  Grade _fallbackScore(TextSequence sequence) {
    return Grade(
      overall: 0,
      method: 'onnx_v7_fallback',
      characterScores: null,
      details: {
        'fallback': true,
        'reason': 'Model not available or error occurred',
      },
    );
  }

  @override
  Future<void> dispose() async {
    _session?.release();
    _session = null;
    _isReady = false;
  }
}

/// Result from single inference pass
class _InferenceResult {
  final List<double> syllableScores;
  final List<double> toneScores;
  final int outputFrames;

  _InferenceResult({
    required this.syllableScores,
    required this.toneScores,
    required this.outputFrames,
  });
}
