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
}
