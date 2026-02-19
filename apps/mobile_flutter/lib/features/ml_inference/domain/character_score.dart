/// Grade thresholds for mapping ML probability to user-facing grades.
///
/// Matches thresholds from Python model: probability_to_grade()
/// - bad: < 0.2
/// - almost: 0.2 - 0.4
/// - good: 0.4 - 0.6
/// - easy: >= 0.6
enum CharacterGrade { bad, almost, good, easy }

/// Extension to convert ML probability to grade.
extension ProbabilityToGrade on double {
  /// Convert probability (0.0-1.0) to character grade.
  CharacterGrade toCharacterGrade() {
    if (this >= 0.6) return CharacterGrade.easy;
    if (this >= 0.4) return CharacterGrade.good;
    if (this >= 0.2) return CharacterGrade.almost;
    return CharacterGrade.bad;
  }
}
