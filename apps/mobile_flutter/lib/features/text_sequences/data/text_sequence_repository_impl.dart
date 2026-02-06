import '../domain/text_sequence.dart';
import '../domain/text_sequence_repository.dart';
import 'dataset_source.dart';
import 'text_sequence_dto.dart';

class TextSequenceRepositoryImpl implements TextSequenceRepository {
  TextSequenceRepositoryImpl(this._source);

  final DatasetSource _source;

  List<TextSequence>? _cache;

  Future<void> _ensureLoaded() async {
    if (_cache != null) return;

    final dataset = await _source.load();

    _cache = dataset.items.map((item) {
      final dto = TextSequenceDto.fromJson(item);
      return dto.toDomain(dataset.language);
    }).toList();
  }

  @override
  Future<List<TextSequence>> getAll() async {
    await _ensureLoaded();
    return List.unmodifiable(_cache!);
  }

  @override
  Future<TextSequence?> getById(String id) async {
    await _ensureLoaded();
    return _cache!.where((s) => s.id == id).firstOrNull;
  }

  @override
  Future<List<TextSequence>> getByTag(String tag) async {
    await _ensureLoaded();
    return _cache!.where((s) => s.tags?.contains(tag) ?? false).toList();
  }

  @override
  Future<List<TextSequence>> getByDifficulty(int difficulty) async {
    await _ensureLoaded();
    return _cache!.where((s) => s.difficulty == difficulty).toList();
  }

  @override
  Future<int> count() async {
    await _ensureLoaded();
    return _cache!.length;
  }
}
