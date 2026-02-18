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
  // nFft must be 512 to match training preprocessing
  final MelExtractor _melExtractor = MelExtractor(nFft: 512);
  bool _isReady = false;

  static const _method = 'onnx_v6';
  static const _modelPath = 'assets/models/model_v6.onnx';

  // V6 supports up to 10s audio (1000 frames at 10ms hop)
  static const _maxMelFrames = 1000;

  // V6 max syllable positions
  static const _maxSyllables = 28;

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
      print('⏱️ [V6] Audio load: ${stepWatch.elapsedMilliseconds}ms (${audioSamples.length} samples)');

      // 2. Get pinyin syllables from sequence
      var syllables = _parsePinyin(sequence.romanization ?? '');

      if (syllables.isEmpty) {
        return _fallbackScore(sequence);
      }

      // 3. Clamp syllables to max (V6 supports up to 28)
      if (syllables.length > _maxSyllables) {
        print('⚠️ [V6] Clamping syllables from ${syllables.length} to $_maxSyllables');
        syllables = syllables.sublist(0, _maxSyllables);
      }

      // 4. Extract mel from FULL audio once (V6 uses full sentence, no centering)
      stepWatch.reset();
      stepWatch.start();
      final mel = _melExtractor.extract(audioSamples);
      print('⏱️ [V6] Mel extraction: ${stepWatch.elapsedMilliseconds}ms (${mel[0].length} frames)');

      // 5. For each syllable position, run inference with position index
      stepWatch.reset();
      stepWatch.start();
      final scores = <double>[];
      for (int i = 0; i < syllables.length; i++) {
        final prob = await _runInference(mel, i, syllables[i]);
        scores.add(prob);
      }
      print('⏱️ [V6] Inference (${syllables.length} syllables): ${stepWatch.elapsedMilliseconds}ms (${(stepWatch.elapsedMilliseconds / syllables.length).toStringAsFixed(1)}ms/syllable)');

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
      print('📊 [V6] Scores: ${scores.map((s) => s.toStringAsFixed(3)).join(", ")}');

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

  /// Run ONNX inference for a single syllable position.
  ///
  /// V6 uses: mel [1, 80, 1000], position [1, 1], audio_mask [1, 1000]
  /// Position is 0-based index; model internally handles [BOS=1, position=2+i] tokens.
  Future<double> _runInference(
    List<List<double>> mel,
    int position,
    String targetSyllable,
  ) async {
    if (_session == null || _vocab == null) {
      throw StateError('Scorer not initialized');
    }

    // Get target syllable ID (tone-stripped)
    final targetSylId = _vocab!.encode(targetSyllable);
    // Get target tone (1-4, or 0 for neutral)
    final targetTone = _extractTone(targetSyllable);

    // V6: Always pad to fixed 1000 frames (no centering, just pad at end)
    final origFrames = mel[0].length;
    final timeFrames = _maxMelFrames;
    final actualFrames = origFrames < timeFrames ? origFrames : timeFrames;

    // Prepare mel tensor: [1, 80, 1000]
    // Copy frames starting at index 0, zero-pad remaining
    final melFlat = Float32List(80 * timeFrames);
    for (int i = 0; i < 80; i++) {
      for (int t = 0; t < actualFrames; t++) {
        melFlat[i * timeFrames + t] = mel[i][t];
      }
      // Remaining frames are already zero (Float32List default)
    }

    // V6: position is a single int64 (0-based index)
    final positionList = Int64List.fromList([position]);

    // Audio mask: true for padded frames, false for real audio frames
    final audioMask = List<bool>.generate(
      timeFrames,
      (t) => t >= actualFrames,
    );

    final melTensor = OrtValueTensor.createTensorWithDataList(
      melFlat,
      [1, 80, timeFrames],
    );
    final positionTensor = OrtValueTensor.createTensorWithDataList(
      positionList,
      [1, 1],
    );
    final audioMaskTensor = OrtValueTensor.createTensorWithDataList(
      audioMask,
      [1, timeFrames],
    );

    List<OrtValue?>? outputs;
    OrtRunOptions? runOptions;

    try {
      final inputs = {
        'mel': melTensor,
        'position': positionTensor,
        'audio_mask': audioMaskTensor,
      };

      runOptions = OrtRunOptions();
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
      final toneIdx = targetTone > 0 ? targetTone - 1 : 4;
      final toneProb = (toneIdx >= 0 && toneIdx < toneProbs.length)
          ? toneProbs[toneIdx]
          : 1.0;

      // Combined score: 70% syllable, 30% tone
      final combined = 0.7 * sylProb + 0.3 * toneProb;

      return combined;
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
    return romanization
        .trim()
        .split(' ')
        .where((s) => s.isNotEmpty)
        .toList();
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
