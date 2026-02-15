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
class OnnxMlScorer implements MlScorer {
  OrtSession? _session;
  SyllableVocab? _vocab;
  final MelExtractor _melExtractor = MelExtractor();
  bool _isReady = false;

  static const _method = 'onnx_v4';
  static const _modelPath = 'assets/models/v4_model.onnx';

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
    if (!_isReady) {
      await initialize();
    }

    try {
      // 1. Load audio from recording file
      final audioFile = File(recording.filePath);
      final audioBytes = await audioFile.readAsBytes();
      final audioSamples = _parseWavToSamples(audioBytes);

      // 2. Extract mel spectrogram
      final mel = _melExtractor.extract(audioSamples);

      // 3. Get pinyin syllables from sequence
      final syllables = _parsePinyin(sequence.romanization ?? '');

      if (syllables.isEmpty) {
        // No romanization available, return mock score
        return _fallbackScore(sequence);
      }

      // 4. For each syllable, run inference and get probability
      final scores = <double>[];
      for (int i = 0; i < syllables.length; i++) {
        final context = syllables.sublist(0, i);
        final pinyinIds = _vocab!.encodeSequence(context, addBos: true);
        final prob = await _runInference(mel, pinyinIds, syllables[i]);
        scores.add(prob);
      }

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
      return _fallbackScore(sequence);
    }
  }

  /// Run ONNX inference for a single syllable.
  Future<double> _runInference(
    List<List<double>> mel,
    List<int> pinyinIds,
    String targetSyllable,
  ) async {
    if (_session == null || _vocab == null) {
      throw StateError('Scorer not initialized');
    }

    // Get target syllable ID
    final targetId = _vocab!.encode(targetSyllable);

    // Prepare mel input: flatten to [1, 80, time]
    final timeFrames = mel[0].length;
    final melFlat = Float32List(80 * timeFrames);
    for (int i = 0; i < 80; i++) {
      for (int t = 0; t < timeFrames; t++) {
        melFlat[i * timeFrames + t] = mel[i][t];
      }
    }

    // Prepare pinyin_ids: [1, seq_len]
    final pinyinFlat = Int64List.fromList(pinyinIds.map((e) => e).toList());

    // Create masks: all ones for simplicity (no padding)
    final audioMask = Float32List(timeFrames);
    for (int i = 0; i < timeFrames; i++) {
      audioMask[i] = 1.0;
    }

    final pinyinMask = Float32List(pinyinIds.length);
    for (int i = 0; i < pinyinIds.length; i++) {
      pinyinMask[i] = 1.0;
    }

    // Create input tensors
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
      // Run inference
      final inputs = {
        'mel': melTensor,
        'pinyin_ids': pinyinTensor,
        'audio_mask': audioMaskTensor,
        'pinyin_mask': pinyinMaskTensor,
      };

      runOptions = OrtRunOptions();
      outputs = await _session!.runAsync(
        runOptions,
        inputs,
        ['syllable_logits'], // Request only syllable_logits output
      );

      // Get syllable_logits output: [1, vocab_size]
      // Outputs are returned in order requested
      if (outputs == null || outputs.isEmpty || outputs[0] == null) {
        throw StateError('Model did not return syllable_logits');
      }

      final syllableLogits = outputs[0]!;

      // Extract logits as list
      final logitsData = syllableLogits.value as List;
      // Handle batch dimension: logitsData is [1, vocab_size]
      final logits = (logitsData[0] as List).cast<double>();

      // Apply softmax
      final probs = _softmax(logits);

      // Get probability for target syllable
      if (targetId >= 0 && targetId < probs.length) {
        return probs[targetId];
      } else {
        return 0.0; // Unknown syllable
      }
    } finally {
      // Release output tensors
      outputs?.forEach((output) => output?.release());
      runOptions?.release();
      // Release tensors
      melTensor.release();
      pinyinTensor.release();
      audioMaskTensor.release();
      pinyinMaskTensor.release();
    }
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
