import 'dart:io';
import 'dart:math' show exp, max, sqrt;
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

/// ONNX-based ML scorer for pronunciation assessment (V6 architecture).
///
/// Uses the V6 model with full-sentence mel spectrogram and position embedding.
/// V6 uses sliding window attention for efficient processing of full 10-second
/// audio sequences.
///
/// V6 Model Input:
/// - mel: [1, 80, 1000] - full sentence mel spectrogram (always 1000 frames)
/// - position: [1, 1] - syllable position index (0-based)
/// - audio_mask: [1, 1000] - padding mask (true = padded)
///
/// Key differences from V5:
/// - Always pads audio to 10 seconds (1000 frames)
/// - No audio centering - just pads at end
/// - Max 28 syllable positions
/// - 4x CNN downsampling (vs 8x in some versions)
/// - Sliding window attention with global attention on position tokens
class OnnxMlScorerV6 implements MlScorer {
  OrtSession? _session;
  SyllableVocab? _vocab;
  // nFft must equal win_length=400 to match training (syllable_predictor_v4.py)
  final MelExtractor _melExtractor = MelExtractor();
  bool _isReady = false;

  static const _method = 'onnx_v6';
  static const _modelPath = 'assets/models/model_v6.onnx';

  // V6 supports up to 10s audio (1000 frames at 10ms hop)
  static const _maxMelFrames = 1000;

  // V6 max syllable positions
  static const _maxSyllables = 28;

  // Padding value for CMVN-normalized mel frames (matches Python training).
  // After CMVN (mean=0, std=1), -5.0 represents a 5-sigma silence floor.
  static const _cmvnPadValue = -5.0;

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

      // Create session options
      final sessionOptions = OrtSessionOptions();

      // Create session from buffer
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
        '⏱️ [V6] Audio load: ${stepWatch.elapsedMilliseconds}ms (${audioSamples.length} samples)',
      );

      // 2. Get pinyin syllables from sequence
      var syllables = _parsePinyin(sequence.romanization ?? '');

      if (syllables.isEmpty) {
        return _fallbackScore(sequence);
      }

      // 3. Clamp syllables to max (V6 supports up to 28)
      if (syllables.length > _maxSyllables) {
        print(
          '⚠️ [V6] Clamping syllables from ${syllables.length} to $_maxSyllables',
        );
        syllables = syllables.sublist(0, _maxSyllables);
      }

      // 4. Extract mel from FULL audio once (V6 uses full sentence, no centering)
      stepWatch.reset();
      stepWatch.start();
      final mel = _melExtractor.extract(audioSamples);
      print(
        '⏱️ [V6] Mel extraction: ${stepWatch.elapsedMilliseconds}ms (${mel[0].length} frames)',
      );

      // 5. Run batched inference: all N syllables in one ORT call
      stepWatch.reset();
      stepWatch.start();
      final scores = await _runBatchedInference(mel, syllables);
      print(
        '⏱️ [V6] Batched inference (${syllables.length} syllables): ${stepWatch.elapsedMilliseconds}ms'
        ' (${syllables.isNotEmpty ? (stepWatch.elapsedMilliseconds / syllables.length).toStringAsFixed(1) : "0"}ms/syllable)',
      );

      // 6. Map syllable scores to character scores
      final characters = sequence.text.characters.toList();
      final characterScores = _mapSyllablesToCharacters(
        scores,
        characters.length,
        syllables.length,
      );

      // 7. Compute overall grade
      final avgScore = characterScores.isEmpty
          ? 0.0
          : characterScores.reduce((a, b) => a + b) / characterScores.length;

      totalStopwatch.stop();
      print('⏱️ [V6] TOTAL scoring: ${totalStopwatch.elapsedMilliseconds}ms');
      print(
        '📊 [V6] Scores: ${scores.map((s) => s.toStringAsFixed(3)).join(", ")}',
      );

      return Grade(
        overall: (avgScore * 100).round(),
        method: _method,
        characterScores: characterScores,
        details: {
          'syllableCount': syllables.length,
          'characterCount': characters.length,
          'avgScore': avgScore,
        },
      );
    } catch (e) {
      print('❌ [V6] Scoring error: $e');
      return _fallbackScore(sequence);
    }
  }

  /// Run batched ONNX inference for all syllable positions in one call.
  ///
  /// The mel and audio_mask are identical for every syllable (same audio);
  /// only the position index varies. By batching N syllables into one ORT
  /// call we eliminate N-1 session round-trips.
  ///
  /// Inputs:  mel [N, 80, 1000], position [N, 1], audio_mask [N, 1000]
  /// Outputs: syllable_logits [N, 532], tone_logits [N, 5]
  Future<List<double>> _runBatchedInference(
    List<List<double>> mel,
    List<String> syllables,
  ) async {
    if (_session == null || _vocab == null) {
      throw StateError('Scorer not initialized');
    }

    final n = syllables.length;
    final timeFrames = _maxMelFrames;
    final origFrames = mel[0].length;
    final actualFrames = origFrames < timeFrames ? origFrames : timeFrames;

    // 1. Compute CMVN once over valid frames (shared across all syllables)
    const epsilon = 1e-5;
    final melRow = Float32List(80 * timeFrames);
    for (int i = 0; i < 80; i++) {
      double sum = 0.0;
      for (int t = 0; t < actualFrames; t++) {
        sum += mel[i][t];
      }
      final mean = sum / actualFrames;

      double sumSqDiff = 0.0;
      for (int t = 0; t < actualFrames; t++) {
        final diff = mel[i][t] - mean;
        sumSqDiff += diff * diff;
      }
      final std = sqrt(sumSqDiff / actualFrames);

      for (int t = 0; t < actualFrames; t++) {
        melRow[i * timeFrames + t] = (mel[i][t] - mean) / (std + epsilon);
      }
      for (int t = actualFrames; t < timeFrames; t++) {
        melRow[i * timeFrames + t] = _cmvnPadValue;
      }
    }

    // 2. Tile mel row N times → [N, 80, 1000]
    final melBatch = Float32List(n * 80 * timeFrames);
    for (int s = 0; s < n; s++) {
      melBatch.setRange(s * 80 * timeFrames, (s + 1) * 80 * timeFrames, melRow);
    }

    // 3. Build position batch [N, 1]
    final positionBatch = Int64List(n);
    for (int s = 0; s < n; s++) {
      positionBatch[s] = s;
    }

    // 4. Build audio mask batch [N, 1000] — same mask replicated
    final maskRow = List<bool>.generate(timeFrames, (t) => t >= actualFrames);
    final maskBatch = <bool>[];
    for (int s = 0; s < n; s++) {
      maskBatch.addAll(maskRow);
    }

    final melTensor = OrtValueTensor.createTensorWithDataList(melBatch, [
      n,
      80,
      timeFrames,
    ]);
    final positionTensor = OrtValueTensor.createTensorWithDataList(
      positionBatch,
      [n, 1],
    );
    final audioMaskTensor = OrtValueTensor.createTensorWithDataList(maskBatch, [
      n,
      timeFrames,
    ]);

    List<OrtValue?>? outputs;
    OrtRunOptions? runOptions;

    try {
      runOptions = OrtRunOptions();
      outputs = await _session!.runAsync(
        runOptions,
        {
          'mel': melTensor,
          'position': positionTensor,
          'audio_mask': audioMaskTensor,
        },
        ['syllable_logits', 'tone_logits'],
      );

      if (outputs == null || outputs.length < 2) {
        throw StateError('Model did not return expected outputs');
      }

      // outputs[0].value → List of N rows, each List<double> of 532 logits
      final allSylLogits = outputs[0]!.value as List;
      final allToneLogits = outputs[1]!.value as List;

      final scores = <double>[];
      for (int s = 0; s < n; s++) {
        final targetSylId = _vocab!.encode(syllables[s]);
        final targetTone = _extractTone(syllables[s]);

        final sylLogits = (allSylLogits[s] as List).cast<double>();
        final sylProbs = _softmax(sylLogits);
        final sylProb = (targetSylId >= 0 && targetSylId < sylProbs.length)
            ? sylProbs[targetSylId]
            : 0.0;

        final toneLogits = (allToneLogits[s] as List).cast<double>();
        final toneProbs = _softmax(toneLogits);
        final toneIdx = targetTone > 0 ? targetTone - 1 : 4;
        final toneProb = (toneIdx >= 0 && toneIdx < toneProbs.length)
            ? toneProbs[toneIdx]
            : 1.0;

        scores.add(0.7 * sylProb + 0.3 * toneProb);
      }
      return scores;
    } finally {
      outputs?.forEach((output) => output?.release());
      runOptions?.release();
      melTensor.release();
      positionTensor.release();
      audioMaskTensor.release();
    }
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
    return 0;
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
      method: 'onnx_v6_fallback',
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
