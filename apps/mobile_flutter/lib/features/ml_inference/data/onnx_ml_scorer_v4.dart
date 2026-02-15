import 'dart:io';
import 'dart:math' show exp, max;
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

/// ONNX-based ML scorer for pronunciation assessment.
///
/// Uses the V4 model with mel spectrogram input and autoregressive
/// pinyin context to score pronunciation at syllable/character level.
class OnnxMlScorerV4 implements MlScorer {
  OrtSession? _session;
  SyllableVocab? _vocab;
  // nFft must be 512 to match training preprocessing (not 400 default)
  final MelExtractor _melExtractor = MelExtractor(nFft: 512);
  bool _isReady = false;

  static const _method = 'onnx_v4';
  static const _modelPath = 'assets/models/model.onnx';

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

      // Create session options (default settings)
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
      print('⏱️ Audio load: ${stepWatch.elapsedMilliseconds}ms (${audioSamples.length} samples)');

      // 2. Get pinyin syllables from sequence
      final syllables = _parsePinyin(sequence.romanization ?? '');

      if (syllables.isEmpty) {
        return _fallbackScore(sequence);
      }

      // 3. Extract mel from FULL audio (same for all syllables)
      // Position encoding tells model which syllable to score
      stepWatch.reset();
      stepWatch.start();
      final mel = _melExtractor.extract(audioSamples);
      print('⏱️ Mel extraction: ${stepWatch.elapsedMilliseconds}ms (${mel[0].length} frames)');

      // 4. For each syllable, run inference with position encoding
      stepWatch.reset();
      stepWatch.start();
      final scores = <double>[];
      for (int i = 0; i < syllables.length; i++) {
        // Position mode: [BOS, 2 + syllable_index]
        // Model uses full audio + position to determine which syllable to score
        final pinyinIds = [1, 2 + i];
        final prob = await _runInference(mel, pinyinIds, syllables[i]);
        scores.add(prob);
      }
      print('⏱️ Inference (${syllables.length} syllables): ${stepWatch.elapsedMilliseconds}ms (${(stepWatch.elapsedMilliseconds / syllables.length).toStringAsFixed(1)}ms/syllable)');

      // 5. Map syllable scores to character scores
      final characters = sequence.text.characters.toList();
      final characterScores = _mapSyllablesToCharacters(
        scores,
        characters.length,
        syllables.length,
      );

      // 6. Compute overall grade
      final avgScore = characterScores.isEmpty
          ? 0.0
          : characterScores.reduce((a, b) => a + b) / characterScores.length;

      totalStopwatch.stop();
      print('⏱️ TOTAL scoring: ${totalStopwatch.elapsedMilliseconds}ms');
      print('📊 Scores: ${scores.map((s) => s.toStringAsFixed(3)).join(", ")}');

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
      // On any error, fall back to mock-like behavior
      print('❌ Scoring error: $e');
      return _fallbackScore(sequence);
    }
  }

  // Model config: max_audio_frames=100, CNN 4x downsampling gives 25 internal frames
  // Position mode: pinyin_ids = [BOS, position] always length 2
  // Combined seq = 25 audio + 2 pinyin = 27 (baked into ONNX reshape ops)
  static const _maxMelFrames = 100;

  /// Run ONNX inference for a single syllable.
  ///
  /// Returns combined score from syllable and tone predictions.
  Future<double> _runInference(
    List<List<double>> mel,
    List<int> pinyinIds,
    String targetSyllable,
  ) async {
    if (_session == null || _vocab == null) {
      throw StateError('Scorer not initialized');
    }

    // Get target syllable ID (tone-stripped)
    final targetSylId = _vocab!.encode(targetSyllable);
    // Get target tone (1-4, or 0 for neutral/unknown)
    final targetTone = _extractTone(targetSyllable);

    // Pad or truncate mel to fixed length (model expects exactly 100 frames)
    final origFrames = mel[0].length;
    final timeFrames = _maxMelFrames;
    final actualFrames = origFrames < timeFrames ? origFrames : timeFrames;

    final melFlat = Float32List(80 * timeFrames);
    for (int i = 0; i < 80; i++) {
      for (int t = 0; t < actualFrames; t++) {
        melFlat[i * timeFrames + t] = mel[i][t];
      }
    }

    final pinyinFlat = Int64List.fromList(pinyinIds);
    final audioMask = List<bool>.generate(
      timeFrames,
      (t) => t >= actualFrames,
    );
    final pinyinMask = List<bool>.filled(pinyinIds.length, false);

    final melTensor = OrtValueTensor.createTensorWithDataList(
      melFlat,
      [1, 80, timeFrames],
    );
    final pinyinTensor = OrtValueTensor.createTensorWithDataList(
      pinyinFlat,
      [1, pinyinIds.length],
    );
    final audioMaskTensor = OrtValueTensor.createTensorWithDataList(
      audioMask,
      [1, timeFrames],
    );
    final pinyinMaskTensor = OrtValueTensor.createTensorWithDataList(
      pinyinMask,
      [1, pinyinIds.length],
    );

    List<OrtValue?>? outputs;
    OrtRunOptions? runOptions;

    try {
      final inputs = {
        'mel': melTensor,
        'pinyin_ids': pinyinTensor,
        'audio_mask': audioMaskTensor,
        'pinyin_mask': pinyinMaskTensor,
      };

      runOptions = OrtRunOptions();
      // Request both syllable and tone outputs
      outputs = await _session!.runAsync(
        runOptions,
        inputs,
        ['syllable_logits', 'tone_logits'],
      );

      if (outputs == null || outputs.length < 2) {
        throw StateError('Model did not return expected outputs');
      }

      // Extract syllable probability
      final sylLogitsData = outputs[0]!.value as List;
      final sylLogits = (sylLogitsData[0] as List).cast<double>();
      final sylProbs = _softmax(sylLogits);
      final sylProb = (targetSylId >= 0 && targetSylId < sylProbs.length)
          ? sylProbs[targetSylId]
          : 0.0;

      // Extract tone probability
      final toneLogitsData = outputs[1]!.value as List;
      final toneLogits = (toneLogitsData[0] as List).cast<double>();
      final toneProbs = _softmax(toneLogits);
      // Tone indices: 0=tone1, 1=tone2, 2=tone3, 3=tone4, 4=neutral
      final toneIdx = targetTone > 0 ? targetTone - 1 : 4;
      final toneProb = (toneIdx >= 0 && toneIdx < toneProbs.length)
          ? toneProbs[toneIdx]
          : 1.0; // If no tone info, don't penalize

      // Combined score: weighted average (syllable more important than tone)
      final combined = 0.7 * sylProb + 0.3 * toneProb;

      return combined;
    } finally {
      outputs?.forEach((output) => output?.release());
      runOptions?.release();
      melTensor.release();
      pinyinTensor.release();
      audioMaskTensor.release();
      pinyinMaskTensor.release();
    }
  }

  /// Extract tone number from pinyin syllable.
  ///
  /// Returns 1-4 for tones, 0 for neutral/unknown.
  static int _extractTone(String syllable) {
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
    return 0; // Neutral tone
  }

  /// Apply softmax to logits.
  List<double> _softmax(List<double> logits) {
    final maxLogit = logits.reduce(max);
    final exps = logits.map((x) => exp(x - maxLogit)).toList();
    final sumExps = exps.reduce((a, b) => a + b);
    return exps.map((x) => x / sumExps).toList();
  }

  /// Parse WAV file bytes to raw PCM samples as List<double>.
  ///
  /// Assumes 16-bit PCM mono WAV format at 16kHz.
  List<double> _parseWavToSamples(Uint8List wavBytes) {
    // Skip 44-byte WAV header (standard PCM WAV)
    const headerSize = 44;
    if (wavBytes.length < headerSize) {
      return [];
    }

    final pcmBytes = wavBytes.sublist(headerSize);
    final samples = <double>[];

    // Parse 16-bit little-endian PCM samples
    for (int i = 0; i < pcmBytes.length - 1; i += 2) {
      final sample16 = pcmBytes[i] | (pcmBytes[i + 1] << 8);
      // Convert to signed 16-bit
      final signedSample = sample16 > 32767 ? sample16 - 65536 : sample16;
      // Normalize to [-1.0, 1.0]
      samples.add(signedSample / 32768.0);
    }

    return samples;
  }

  /// Parse pinyin romanization string to list of syllables.
  ///
  /// Expected format: space-separated pinyin syllables (e.g., "nǐ hǎo").
  List<String> _parsePinyin(String romanization) {
    return romanization
        .trim()
        .split(' ')
        .where((s) => s.isNotEmpty)
        .toList();
  }

  /// Map syllable scores to character scores.
  ///
  /// For simplicity, we assume 1-to-1 mapping for now.
  /// If counts differ, we interpolate or pad as needed.
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

    // If more characters than syllables (multi-character words),
    // repeat each score proportionally
    if (characterCount > syllableCount) {
      final charScores = <double>[];
      for (int i = 0; i < characterCount; i++) {
        final syllableIdx = (i * syllableCount / characterCount).floor();
        charScores.add(syllableScores[syllableIdx]);
      }
      return charScores;
    }

    // If fewer characters (rare), take first N scores
    return syllableScores.sublist(0, characterCount);
  }

  /// Fallback score when model fails or data is unavailable.
  ///
  /// Returns a Grade with no characterScores, so ColoredText displays
  /// white (unscored) text instead of random colors.
  Grade _fallbackScore(TextSequence sequence) {
    return Grade(
      overall: 0,
      method: '${_method}_fallback',
      characterScores: null, // No scores = white text in ColoredText
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
