import 'text_sequence.dart';

/// Repository interface for accessing text sequences.
abstract class TextSequenceRepository {
  /// Returns all text sequences.
  Future<List<TextSequence>> getAll();

  /// Returns the text sequence with the given [id], or null if not found.
  Future<TextSequence?> getById(String id);

  /// Returns all text sequences that contain the given [tag].
  Future<List<TextSequence>> getByTag(String tag);

  /// Returns all text sequences with the given [difficulty] level.
  Future<List<TextSequence>> getByDifficulty(int difficulty);

  /// Returns the total count of text sequences.
  Future<int> count();
}
