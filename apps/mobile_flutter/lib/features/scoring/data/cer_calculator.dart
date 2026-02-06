import '../../../core/utils/string_utils.dart';
import 'cer_result.dart';

/// Calculator for Character Error Rate between reference and hypothesis text.
class CerCalculator {
  /// Calculates the CER between reference and hypothesis text.
  ///
  /// Both texts are normalized (punctuation removed, lowercased) before
  /// comparison using [normalizeZhText].
  CerResult calculate(String reference, String hypothesis) {
    final normalizedRef = normalizeZhText(reference);
    final normalizedHyp = normalizeZhText(hypothesis);

    final distance = levenshteinDistance(normalizedRef, normalizedHyp);
    final cer = calculateCer(normalizedRef, normalizedHyp);

    return CerResult(
      cer: cer,
      referenceLength: normalizedRef.length,
      hypothesisLength: normalizedHyp.length,
      editDistance: distance,
    );
  }
}
