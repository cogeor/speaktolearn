/// Result of a Character Error Rate (CER) calculation.
class CerResult {
  const CerResult({
    required this.cer,
    required this.referenceLength,
    required this.hypothesisLength,
    required this.editDistance,
  });

  /// The Character Error Rate (0.0 = perfect, 1.0 = 100% errors).
  final double cer;

  /// Number of characters in the reference text.
  final int referenceLength;

  /// Number of characters in the hypothesis text.
  final int hypothesisLength;

  /// Levenshtein edit distance between reference and hypothesis.
  final int editDistance;

  /// Converts CER to a 0-100 score (higher is better).
  /// CER of 0 = 100, CER of 1 = 0, CER > 1 = 0.
  int get score {
    if (cer >= 1.0) return 0;
    return ((1.0 - cer) * 100).round();
  }

  /// Accuracy score derived from CER (0-100).
  /// Same as score, but named explicitly for semantic clarity.
  int get accuracy => score;

  /// Completeness score (0-100).
  /// Estimates how much of the reference text was covered by the hypothesis.
  ///
  /// Calculation:
  /// - If hypothesis is empty, completeness is 0
  /// - If hypothesis covers reference well, completeness is high
  /// - Uses the minimum of reference and hypothesis length vs edit distance
  int get completeness {
    if (referenceLength == 0) return 100;
    if (hypothesisLength == 0) return 0;

    // Approximate matched characters: we know the edit distance includes
    // insertions, deletions, and substitutions. For completeness, we care
    // about how much of the reference was covered.
    // matched = referenceLength - deletions - substitutions
    // A rough estimate: matched = referenceLength - max(0, editDistance - insertions)
    // Since editDistance = insertions + deletions + substitutions,
    // and insertions ~= max(0, hypLen - refLen),
    // deletions ~= max(0, refLen - hypLen)
    final insertions = hypothesisLength > referenceLength
        ? hypothesisLength - referenceLength
        : 0;
    final deletionsAndSubstitutions = editDistance - insertions;
    final matched = referenceLength - deletionsAndSubstitutions;

    if (matched <= 0) return 0;
    final ratio = matched / referenceLength;
    return (ratio.clamp(0.0, 1.0) * 100).round();
  }
}
