import '../domain/text_sequence.dart';
import '../domain/text_sequence_repository.dart';
import 'dataset_source.dart';
import 'text_sequence_dto.dart';

class TextSequenceRepositoryImpl implements TextSequenceRepository {
  TextSequenceRepositoryImpl(this._source);

  final DatasetSource _source;

  Future<List<TextSequence>> _loadSequences() async {
    final dataset = await _source.load();
    return dataset.items.map((item) {
      final dto = TextSequenceDto.fromJson(item);
      return dto.toDomain(dataset.language);
    }).toList();
  }

  @override
  Future<List<TextSequence>> getAll() async {
    final sequences = await _loadSequences();
    return List.unmodifiable(sequences);
  }

  @override
  Future<TextSequence?> getById(String id) async {
    final sequences = await _loadSequences();
    return sequences.where((s) => s.id == id).firstOrNull;
  }

  @override
  Future<List<TextSequence>> getByTag(String tag) async {
    final sequences = await _loadSequences();
    return sequences.where((s) => s.tags?.contains(tag) ?? false).toList();
  }

  @override
  Future<List<TextSequence>> getByDifficulty(int difficulty) async {
    final sequences = await _loadSequences();
    return sequences.where((s) => s.difficulty == difficulty).toList();
  }

  @override
  Future<List<TextSequence>> getByLevel(int level) async {
    final sequences = await _loadSequences();
    return sequences.where((s) => s.hskLevel == level).toList();
  }

  @override
  Future<int> count() async {
    final sequences = await _loadSequences();
    return sequences.length;
  }
}
