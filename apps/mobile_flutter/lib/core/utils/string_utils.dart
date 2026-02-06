/// Text normalization utilities for Chinese text scoring.
library;

import 'package:characters/characters.dart';

/// Normalizes Chinese text for pronunciation comparison.
///
/// Operations:
/// 1. Removes all punctuation (Chinese and ASCII)
/// 2. Removes all whitespace
/// 3. Converts full-width ASCII to half-width
/// 4. Converts to lowercase (for any ASCII)
String normalizeZhText(String input) {
  // Chinese punctuation + ASCII punctuation
  // Using Unicode property escapes for brackets
  final punctuation = RegExp(
    '[。，！？、；：""''（）【】《》〈〉…—·\\s.,!?;:"\'()\\[\\]<>]',
  );

  var result = input.replaceAll(punctuation, '');

  // Full-width ASCII to half-width
  result = _fullWidthToHalfWidth(result);

  return result.toLowerCase();
}

/// Converts full-width ASCII characters to half-width.
String _fullWidthToHalfWidth(String input) {
  final buffer = StringBuffer();
  for (final codeUnit in input.codeUnits) {
    // Full-width range: 0xFF01 (!) to 0xFF5E (~)
    // Maps to ASCII 0x21 (!) to 0x7E (~)
    if (codeUnit >= 0xFF01 && codeUnit <= 0xFF5E) {
      buffer.writeCharCode(codeUnit - 0xFEE0);
    }
    // Full-width space (0x3000) to ASCII space (0x20)
    else if (codeUnit == 0x3000) {
      buffer.writeCharCode(0x20);
    } else {
      buffer.writeCharCode(codeUnit);
    }
  }
  return buffer.toString();
}

/// Calculates Character Error Rate (CER) between reference and hypothesis.
///
/// CER = edit_distance(ref, hyp) / length(ref)
///
/// Returns a value between 0.0 (perfect match) and potentially > 1.0
/// (hypothesis much longer than reference).
double calculateCer(String reference, String hypothesis) {
  if (reference.isEmpty) {
    return hypothesis.isEmpty ? 0.0 : 1.0;
  }

  final distance = levenshteinDistance(reference, hypothesis);
  return distance / reference.length;
}

/// Levenshtein distance (edit distance) between two strings.
///
/// Uses [String.characters] for proper Unicode grapheme cluster handling,
/// which correctly handles Chinese characters and combining characters.
int levenshteinDistance(String s1, String s2) {
  if (s1.isEmpty) return s2.characters.length;
  if (s2.isEmpty) return s1.characters.length;

  // Use characters (not code units) for proper Unicode handling
  final chars1 = s1.characters.toList();
  final chars2 = s2.characters.toList();

  final len1 = chars1.length;
  final len2 = chars2.length;

  // Two-row optimization (only need current and previous row)
  var prev = List<int>.generate(len2 + 1, (i) => i);
  var curr = List<int>.filled(len2 + 1, 0);

  for (var i = 1; i <= len1; i++) {
    curr[0] = i;
    for (var j = 1; j <= len2; j++) {
      final cost = chars1[i - 1] == chars2[j - 1] ? 0 : 1;
      curr[j] = [
        prev[j] + 1, // deletion
        curr[j - 1] + 1, // insertion
        prev[j - 1] + cost, // substitution
      ].reduce((a, b) => a < b ? a : b);
    }
    final temp = prev;
    prev = curr;
    curr = temp;
  }

  return prev[len2];
}
