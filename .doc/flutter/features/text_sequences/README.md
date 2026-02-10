# features/text_sequences/ Module

## Purpose

Manages loading and displaying the dataset of text sequences (sentences, phrases). Provides the data that users practice with. Does NOT own user progress (that's `progress/`).

## Folder Structure

```
text_sequences/
├── domain/
│   ├── text_sequence.dart              # Entity
│   └── text_sequence_repository.dart   # Repository interface
├── data/
│   ├── text_sequence_dto.dart          # JSON model
│   ├── dataset_source.dart             # Data source abstraction
│   └── text_sequence_repository_impl.dart
└── presentation/
    ├── sequence_list_screen.dart       # List screen UI
    └── sequence_list_controller.dart   # List state management
```

---

## Domain Layer

### `text_sequence.dart`

**Purpose**: Core entity representing a practice item.

**Implementation**:

```dart
import 'package:freezed_annotation/freezed_annotation.dart';

part 'text_sequence.freezed.dart';

/// A text sequence that can be practiced.
///
/// This is a language-agnostic container - the [language] field
/// indicates what language [text] is in.
@freezed
class TextSequence with _$TextSequence {
  const factory TextSequence({
    /// Unique identifier (e.g., "ts_000001").
    required String id,

    /// Primary text to practice (in target language).
    required String text,

    /// ISO language code (e.g., "zh-CN", "ja-JP").
    required String language,

    /// Romanization (e.g., pinyin for Chinese).
    String? romanization,

    /// Translations keyed by language code.
    /// Example: {"en": "Hello", "de": "Hallo"}
    @Default({}) Map<String, String> gloss,

    /// Individual tokens for highlighting.
    @Default([]) List<String> tokens,

    /// Categorization tags (e.g., ["hsk1", "daily"]).
    @Default([]) List<String> tags,

    /// Difficulty level (1 = easiest).
    int? difficulty,

    /// Example audio voice configurations.
    @Default([]) List<ExampleVoice> exampleAudio,
  }) = _TextSequence;
}

/// A pre-recorded voice example for a text sequence.
@freezed
class ExampleVoice with _$ExampleVoice {
  const factory ExampleVoice({
    /// Voice identifier (e.g., "f1", "m1").
    required String id,

    /// Human-readable label by language.
    /// Example: {"en": "Female", "zh": "女声"}
    @Default({}) Map<String, String> label,

    /// URI to the audio file.
    /// Schemes: "assets://", "file://", "https://"
    required String uri,

    /// Duration in milliseconds (for UI).
    int? durationMs,
  }) = _ExampleVoice;
}
```

**Design Decisions**:
- Uses `gloss` map instead of `translation` string - not English-centric
- `exampleAudio` as list of voices - flexible for any number of examples
- `tokens` for future per-word highlighting
- Freezed for immutability and generated equality

---

### `text_sequence_repository.dart`

**Purpose**: Interface for accessing text sequences.

**Implementation**:

```dart
import 'text_sequence.dart';

/// Repository interface for text sequence data.
///
/// Implementations may load from:
/// - Bundled assets (AssetDatasetSource)
/// - Downloaded files (DownloadedDatasetSource)
/// - Remote API (future)
abstract class TextSequenceRepository {
  /// Returns all text sequences in the dataset.
  Future<List<TextSequence>> getAll();

  /// Returns a single text sequence by ID, or null if not found.
  Future<TextSequence?> getById(String id);

  /// Returns sequences matching the given tag.
  Future<List<TextSequence>> getByTag(String tag);

  /// Returns sequences for a specific difficulty level.
  Future<List<TextSequence>> getByDifficulty(int difficulty);

  /// Returns total count of sequences.
  Future<int> count();
}
```

**Design Decision**: Repository returns domain entities, not DTOs. The mapping happens inside the implementation.

---

## Data Layer

### `text_sequence_dto.dart`

**Purpose**: JSON-serializable data transfer object.

**Implementation**:

```dart
import 'package:json_annotation/json_annotation.dart';
import '../domain/text_sequence.dart';

part 'text_sequence_dto.g.dart';

/// Dataset root JSON structure.
@JsonSerializable()
class DatasetDto {
  final String schemaVersion;
  final String datasetId;
  final String language;
  final DateTime generatedAt;
  final List<TextSequenceDto> items;

  DatasetDto({
    required this.schemaVersion,
    required this.datasetId,
    required this.language,
    required this.generatedAt,
    required this.items,
  });

  factory DatasetDto.fromJson(Map<String, dynamic> json) =>
      _$DatasetDtoFromJson(json);

  Map<String, dynamic> toJson() => _$DatasetDtoToJson(this);
}

/// Individual text sequence JSON structure.
@JsonSerializable()
class TextSequenceDto {
  final String id;
  final String text;
  final String? romanization;
  final Map<String, String>? gloss;
  final List<String>? tokens;
  final List<String>? tags;
  final int? difficulty;
  @JsonKey(name: 'example_audio')
  final ExampleAudioDto? exampleAudio;

  TextSequenceDto({
    required this.id,
    required this.text,
    this.romanization,
    this.gloss,
    this.tokens,
    this.tags,
    this.difficulty,
    this.exampleAudio,
  });

  factory TextSequenceDto.fromJson(Map<String, dynamic> json) =>
      _$TextSequenceDtoFromJson(json);

  /// Converts to domain entity with the dataset's language.
  TextSequence toDomain(String language) => TextSequence(
        id: id,
        text: text,
        language: language,
        romanization: romanization,
        gloss: gloss ?? {},
        tokens: tokens ?? [],
        tags: tags ?? [],
        difficulty: difficulty,
        exampleAudio: exampleAudio?.voices
                .map((v) => ExampleVoice(
                      id: v.id,
                      label: v.label ?? {},
                      uri: v.uri,
                      durationMs: v.durationMs,
                    ))
                .toList() ??
            [],
      );
}

@JsonSerializable()
class ExampleAudioDto {
  final List<VoiceDto> voices;

  ExampleAudioDto({required this.voices});

  factory ExampleAudioDto.fromJson(Map<String, dynamic> json) =>
      _$ExampleAudioDtoFromJson(json);
}

@JsonSerializable()
class VoiceDto {
  final String id;
  final Map<String, String>? label;
  final String uri;
  @JsonKey(name: 'duration_ms')
  final int? durationMs;

  VoiceDto({
    required this.id,
    this.label,
    required this.uri,
    this.durationMs,
  });

  factory VoiceDto.fromJson(Map<String, dynamic> json) =>
      _$VoiceDtoFromJson(json);
}
```

---

### `dataset_source.dart`

**Purpose**: Abstraction for where dataset JSON comes from.

**Implementation**:

```dart
import 'dart:convert';
import 'package:flutter/services.dart';
import 'text_sequence_dto.dart';

/// Abstraction for loading the dataset JSON.
///
/// Allows swapping between bundled assets and downloaded files
/// without changing repository or UI code.
abstract class DatasetSource {
  /// Loads and parses the dataset.
  Future<DatasetDto> load();

  /// Returns true if a dataset is available.
  Future<bool> isAvailable();
}

/// Loads dataset from bundled app assets.
class AssetDatasetSource implements DatasetSource {
  final String assetPath;

  const AssetDatasetSource({
    this.assetPath = 'assets/datasets/sentences.zh.json',
  });

  @override
  Future<DatasetDto> load() async {
    final jsonString = await rootBundle.loadString(assetPath);
    final json = jsonDecode(jsonString) as Map<String, dynamic>;
    return DatasetDto.fromJson(json);
  }

  @override
  Future<bool> isAvailable() async {
    try {
      await rootBundle.loadString(assetPath);
      return true;
    } catch (_) {
      return false;
    }
  }
}

/// Loads dataset from a downloaded file in app storage.
class DownloadedDatasetSource implements DatasetSource {
  final String Function() getFilePath;

  DownloadedDatasetSource({required this.getFilePath});

  @override
  Future<DatasetDto> load() async {
    final file = File(getFilePath());
    final jsonString = await file.readAsString();
    final json = jsonDecode(jsonString) as Map<String, dynamic>;
    return DatasetDto.fromJson(json);
  }

  @override
  Future<bool> isAvailable() async {
    final file = File(getFilePath());
    return file.exists();
  }
}
```

**Design Decision**: `DatasetSource` is a simple abstraction that makes the repository implementation testable and allows future download functionality without changing business logic.

---

### `text_sequence_repository_impl.dart`

**Purpose**: Concrete repository implementation.

**Implementation**:

```dart
import '../domain/text_sequence.dart';
import '../domain/text_sequence_repository.dart';
import 'dataset_source.dart';
import 'text_sequence_dto.dart';

class TextSequenceRepositoryImpl implements TextSequenceRepository {
  final DatasetSource _source;

  // Cache loaded data
  DatasetDto? _cache;
  List<TextSequence>? _entitiesCache;
  Map<String, TextSequence>? _byIdCache;

  TextSequenceRepositoryImpl(this._source);

  Future<void> _ensureLoaded() async {
    if (_cache != null) return;

    _cache = await _source.load();
    _entitiesCache = _cache!.items
        .map((dto) => dto.toDomain(_cache!.language))
        .toList();
    _byIdCache = {for (final e in _entitiesCache!) e.id: e};
  }

  @override
  Future<List<TextSequence>> getAll() async {
    await _ensureLoaded();
    return List.unmodifiable(_entitiesCache!);
  }

  @override
  Future<TextSequence?> getById(String id) async {
    await _ensureLoaded();
    return _byIdCache![id];
  }

  @override
  Future<List<TextSequence>> getByTag(String tag) async {
    await _ensureLoaded();
    return _entitiesCache!.where((e) => e.tags.contains(tag)).toList();
  }

  @override
  Future<List<TextSequence>> getByDifficulty(int difficulty) async {
    await _ensureLoaded();
    return _entitiesCache!.where((e) => e.difficulty == difficulty).toList();
  }

  @override
  Future<int> count() async {
    await _ensureLoaded();
    return _entitiesCache!.length;
  }

  /// Clears the cache. Useful for testing or when dataset updates.
  void clearCache() {
    _cache = null;
    _entitiesCache = null;
    _byIdCache = null;
  }
}
```

**Design Decision**: Caches entire dataset on first load. For MVP dataset sizes (<1000 items), this is simpler than pagination and provides fast lookups.

---

## Presentation Layer

### `sequence_list_screen.dart`

**Purpose**: Screen showing all text sequences with track toggle.

**Implementation**:

```dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'sequence_list_controller.dart';

class SequenceListScreen extends ConsumerWidget {
  const SequenceListScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final state = ref.watch(sequenceListControllerProvider);
    final controller = ref.read(sequenceListControllerProvider.notifier);

    return Scaffold(
      appBar: AppBar(
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => context.pop(),
        ),
        title: const Text('Sequences'),
      ),
      body: state.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (error, _) => Center(child: Text('Error: $error')),
        data: (items) => ListView.separated(
          itemCount: items.length,
          separatorBuilder: (_, __) => const Divider(height: 1),
          itemBuilder: (context, index) {
            final item = items[index];
            return SequenceListTile(
              item: item,
              onTap: () {
                controller.select(item.id);
                context.pop();
              },
              onToggleTrack: () => controller.toggleTracked(item.id),
            );
          },
        ),
      ),
    );
  }
}

class SequenceListTile extends StatelessWidget {
  final SequenceListItem item;
  final VoidCallback onTap;
  final VoidCallback onToggleTrack;

  const SequenceListTile({
    super.key,
    required this.item,
    required this.onTap,
    required this.onToggleTrack,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return ListTile(
      onTap: onTap,
      title: Text(
        item.text,
        style: theme.textTheme.titleMedium,
        maxLines: 2,
        overflow: TextOverflow.ellipsis,
      ),
      trailing: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (item.bestScore != null)
            Padding(
              padding: const EdgeInsets.only(right: 8),
              child: Text(
                item.bestScore!.toStringAsFixed(0),
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.onSurface.withOpacity(0.6),
                ),
              ),
            ),
          IconButton(
            icon: Icon(
              item.isTracked ? Icons.star : Icons.star_border,
              color: item.isTracked
                  ? Colors.amber
                  : theme.colorScheme.onSurface.withOpacity(0.4),
            ),
            onPressed: onToggleTrack,
          ),
        ],
      ),
    );
  }
}
```

---

### `sequence_list_controller.dart`

**Purpose**: State management for the list screen.

**Implementation**:

```dart
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:freezed_annotation/freezed_annotation.dart';
import '../domain/text_sequence.dart';
import '../domain/text_sequence_repository.dart';
import '../../progress/domain/progress_repository.dart';
import '../../practice/presentation/home_controller.dart';

part 'sequence_list_controller.freezed.dart';

/// View model for a single list item.
@freezed
class SequenceListItem with _$SequenceListItem {
  const factory SequenceListItem({
    required String id,
    required String text,
    required bool isTracked,
    double? bestScore,
  }) = _SequenceListItem;
}

final sequenceListControllerProvider = AsyncNotifierProvider<
    SequenceListController, List<SequenceListItem>>(
  SequenceListController.new,
);

class SequenceListController
    extends AsyncNotifier<List<SequenceListItem>> {
  @override
  Future<List<SequenceListItem>> build() async {
    final textSequences = ref.watch(textSequenceRepositoryProvider);
    final progress = ref.watch(progressRepositoryProvider);

    final sequences = await textSequences.getAll();
    final progressMap = await progress.getProgressMap(
      sequences.map((s) => s.id).toList(),
    );

    // Sort: tracked first, then by priority, then by text
    final items = sequences.map((s) {
      final p = progressMap[s.id];
      return SequenceListItem(
        id: s.id,
        text: s.text,
        isTracked: p?.tracked ?? false,
        bestScore: p?.bestScore,
      );
    }).toList();

    items.sort((a, b) {
      // Tracked items first
      if (a.isTracked != b.isTracked) {
        return a.isTracked ? -1 : 1;
      }
      // Then by text alphabetically
      return a.text.compareTo(b.text);
    });

    return items;
  }

  Future<void> toggleTracked(String id) async {
    final progress = ref.read(progressRepositoryProvider);
    await progress.toggleTracked(id);
    ref.invalidateSelf();
  }

  void select(String id) {
    ref.read(homeControllerProvider.notifier).setCurrentSequence(id);
  }
}
```

---

## Integration Tests

### Repository Integration Test

```dart
void main() {
  group('TextSequenceRepository', () {
    late TextSequenceRepository repository;

    setUp(() {
      // Use mock dataset source with test data
      final source = MockDatasetSource(testDataset);
      repository = TextSequenceRepositoryImpl(source);
    });

    test('getAll returns all sequences', () async {
      final sequences = await repository.getAll();
      expect(sequences.length, 3);
    });

    test('getById returns correct sequence', () async {
      final sequence = await repository.getById('ts_000001');
      expect(sequence?.text, '我想喝水。');
    });

    test('getById returns null for unknown id', () async {
      final sequence = await repository.getById('unknown');
      expect(sequence, isNull);
    });

    test('getByTag filters correctly', () async {
      final sequences = await repository.getByTag('hsk1');
      expect(sequences.every((s) => s.tags.contains('hsk1')), isTrue);
    });

    test('getByDifficulty filters correctly', () async {
      final sequences = await repository.getByDifficulty(1);
      expect(sequences.every((s) => s.difficulty == 1), isTrue);
    });
  });
}
```

### Screen Widget Test

```dart
void main() {
  group('SequenceListScreen', () {
    testWidgets('displays list of sequences', (tester) async {
      await tester.pumpWidget(
        ProviderScope(
          overrides: [
            sequenceListControllerProvider.overrideWith(() =>
              MockSequenceListController([
                SequenceListItem(id: '1', text: '你好', isTracked: true),
                SequenceListItem(id: '2', text: '谢谢', isTracked: false),
              ])
            ),
          ],
          child: MaterialApp(home: SequenceListScreen()),
        ),
      );

      await tester.pumpAndSettle();

      expect(find.text('你好'), findsOneWidget);
      expect(find.text('谢谢'), findsOneWidget);
    });

    testWidgets('toggles tracked state on star tap', (tester) async {
      final controller = MockSequenceListController([
        SequenceListItem(id: '1', text: '你好', isTracked: false),
      ]);

      await tester.pumpWidget(
        ProviderScope(
          overrides: [
            sequenceListControllerProvider.overrideWith(() => controller),
          ],
          child: MaterialApp(home: SequenceListScreen()),
        ),
      );

      await tester.pumpAndSettle();
      await tester.tap(find.byIcon(Icons.star_border));
      await tester.pumpAndSettle();

      expect(controller.toggleTrackedCalled, isTrue);
    });

    testWidgets('navigates back on item tap', (tester) async {
      // ... navigation test
    });
  });
}
```

### End-to-End Flow Test

```dart
void main() {
  group('Sequence browsing flow', () {
    testWidgets('user can browse and track sequences', (tester) async {
      await tester.pumpWidget(
        ProviderScope(
          overrides: testOverrides,
          child: const SpeakToLearnApp(),
        ),
      );
      await tester.pumpAndSettle();

      // Navigate to list
      await tester.tap(find.byIcon(Icons.list));
      await tester.pumpAndSettle();

      // Verify list screen
      expect(find.byType(SequenceListScreen), findsOneWidget);

      // Track first item
      await tester.tap(find.byIcon(Icons.star_border).first);
      await tester.pumpAndSettle();

      // Verify star is now filled
      expect(find.byIcon(Icons.star), findsOneWidget);

      // Select item
      await tester.tap(find.text('我想喝水。'));
      await tester.pumpAndSettle();

      // Verify back on home with selected sequence
      expect(find.byType(HomeScreen), findsOneWidget);
      expect(find.text('我想喝水。'), findsOneWidget);
    });
  });
}
```

---

## Notes

### Caching Strategy

The repository caches the entire dataset after first load. This is appropriate because:
- Dataset is static (generated offline)
- Size is bounded (<1000 items for MVP)
- All queries are fast in-memory operations

If dataset grows significantly, consider:
- Pagination in `getAll()`
- Lazy loading
- SQLite for indexed queries

### Future: Download Support

To add downloadable datasets:
1. Create `DownloadedDatasetSource` implementation
2. Add dataset catalog/manifest service
3. Add download UI in settings
4. Repository can combine multiple sources
