import 'dart:convert';
import 'package:flutter/services.dart' show rootBundle;

/// Vocabulary for encoding/decoding Mandarin syllables.
///
/// This class handles the conversion between syllable strings (pinyin)
/// and their integer token IDs used by the ML model.
///
/// Special tokens:
/// - 0: PAD (padding token)
/// - 1: BOS (beginning of sequence)
/// - 2+: syllable tokens
class SyllableVocab {
  /// Padding token ID
  static const int padToken = 0;

  /// Beginning of sequence token ID
  static const int bosToken = 1;

  /// Number of special tokens (PAD, BOS)
  static const int specialTokens = 2;

  final List<String> _syllables;
  final Map<String, int> _sylToIdx;
  final Map<int, String> _idxToSyl;

  SyllableVocab._(this._syllables, this._sylToIdx, this._idxToSyl);

  /// Load vocabulary from assets
  static Future<SyllableVocab> load() async {
    final jsonStr = await rootBundle.loadString('assets/ml/syllable_vocab.json');
    final data = json.decode(jsonStr) as Map<String, dynamic>;
    final syllables = (data['syllables'] as List).cast<String>();

    // Build lookup maps
    final sylToIdx = <String, int>{};
    final idxToSyl = <int, String>{};

    for (int i = 0; i < syllables.length; i++) {
      final idx = i + specialTokens;
      sylToIdx[syllables[i]] = idx;
      idxToSyl[idx] = syllables[i];
    }

    return SyllableVocab._(syllables, sylToIdx, idxToSyl);
  }

  /// Total vocabulary size including special tokens
  int get size => _syllables.length + specialTokens;

  /// Encode a single syllable to its token ID
  ///
  /// Returns [padToken] for unknown syllables.
  /// Normalizes input by stripping tone marks and converting to lowercase.
  int encode(String syllable) {
    // Strip tone marks, lowercase, and handle u/v variants
    final normalized = _stripTones(syllable.toLowerCase()).replaceAll('v', 'u');
    return _sylToIdx[normalized] ?? padToken;
  }

  /// Strip tone marks from pinyin syllable.
  static String _stripTones(String syllable) {
    const toneMap = {
      'ā': 'a', 'á': 'a', 'ǎ': 'a', 'à': 'a',
      'ē': 'e', 'é': 'e', 'ě': 'e', 'è': 'e',
      'ī': 'i', 'í': 'i', 'ǐ': 'i', 'ì': 'i',
      'ō': 'o', 'ó': 'o', 'ǒ': 'o', 'ò': 'o',
      'ū': 'u', 'ú': 'u', 'ǔ': 'u', 'ù': 'u',
      'ǖ': 'v', 'ǘ': 'v', 'ǚ': 'v', 'ǜ': 'v', 'ü': 'v',
    };
    return syllable.split('').map((c) => toneMap[c] ?? c).join();
  }

  /// Decode a token ID to its syllable string
  ///
  /// Returns special token names for PAD/BOS, "<UNK>" for unknown IDs.
  String decode(int idx) {
    if (idx == padToken) return '<PAD>';
    if (idx == bosToken) return '<BOS>';
    return _idxToSyl[idx] ?? '<UNK>';
  }

  /// Encode a sequence of syllables to token IDs
  ///
  /// If [addBos] is true (default), prepends the BOS token.
  List<int> encodeSequence(List<String> syllables, {bool addBos = true}) {
    final tokens = <int>[];
    if (addBos) {
      tokens.add(bosToken);
    }
    for (final syl in syllables) {
      tokens.add(encode(syl));
    }
    return tokens;
  }
}
