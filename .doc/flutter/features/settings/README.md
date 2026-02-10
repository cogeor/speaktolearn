# features/settings/ Module

## Purpose

Manages user preferences: UI language, target language, and other app settings. Keeps the app language-agnostic rather than English-centric.

## Folder Structure

```
settings/
├── domain/
│   └── app_settings.dart           # Settings entity
└── data/
    └── settings_repository_impl.dart  # Hive storage
```

**Note**: No presentation layer for MVP. Settings are accessed from app initialization and could have a settings screen added later.

---

## Domain Layer

### `app_settings.dart`

**Purpose**: App configuration preferences.

**Implementation**:

```dart
import 'package:freezed_annotation/freezed_annotation.dart';

part 'app_settings.freezed.dart';
part 'app_settings.g.dart';

/// User preferences and app configuration.
@freezed
class AppSettings with _$AppSettings {
  const AppSettings._();

  const factory AppSettings({
    /// UI language code (e.g., "en", "de", "zh").
    /// Used for interface text and gloss selection.
    @Default('en') String uiLanguageCode,

    /// Target language being practiced (e.g., "zh-CN", "ja-JP").
    /// Used for TTS, ASR, and dataset filtering.
    @Default('zh-CN') String targetLanguageCode,

    /// Whether to show romanization (pinyin, romaji, etc.).
    @Default(true) bool showRomanization,

    /// Whether to show translation/gloss.
    @Default(true) bool showGloss,

    /// Audio playback speed multiplier (0.5 - 2.0).
    @Default(1.0) double playbackSpeed,

    /// Whether to auto-play example after Next.
    @Default(false) bool autoPlayExample,

    /// Preferred example voice ID (e.g., "f1", "m1").
    String? preferredVoiceId,
  }) = _AppSettings;

  factory AppSettings.fromJson(Map<String, dynamic> json) =>
      _$AppSettingsFromJson(json);

  /// Default settings for new users.
  static const defaults = AppSettings();
}

/// Repository interface for app settings.
abstract class SettingsRepository {
  /// Gets current settings.
  Future<AppSettings> getSettings();

  /// Updates settings.
  Future<void> updateSettings(AppSettings settings);

  /// Resets to defaults.
  Future<void> resetToDefaults();

  /// Stream of settings changes.
  Stream<AppSettings> watchSettings();
}
```

---

## Data Layer

### `settings_repository_impl.dart`

**Purpose**: Hive-based settings storage.

**Implementation**:

```dart
import 'dart:async';
import 'package:hive/hive.dart';
import '../domain/app_settings.dart';

class SettingsRepositoryImpl implements SettingsRepository {
  final Box<dynamic> _box;

  static const _settingsKey = 'app_settings';

  final _controller = StreamController<AppSettings>.broadcast();

  SettingsRepositoryImpl(this._box);

  @override
  Future<AppSettings> getSettings() async {
    final data = _box.get(_settingsKey);
    if (data == null) {
      return AppSettings.defaults;
    }
    return AppSettings.fromJson(Map<String, dynamic>.from(data));
  }

  @override
  Future<void> updateSettings(AppSettings settings) async {
    await _box.put(_settingsKey, settings.toJson());
    _controller.add(settings);
  }

  @override
  Future<void> resetToDefaults() async {
    await _box.delete(_settingsKey);
    _controller.add(AppSettings.defaults);
  }

  @override
  Stream<AppSettings> watchSettings() {
    return _controller.stream;
  }

  void dispose() {
    _controller.close();
  }
}
```

---

## Riverpod Providers

```dart
import 'package:flutter_riverpod/flutter_riverpod.dart';

final settingsRepositoryProvider = Provider<SettingsRepository>((ref) {
  throw UnimplementedError('Override in main');
});

final appSettingsProvider = FutureProvider<AppSettings>((ref) async {
  final repository = ref.watch(settingsRepositoryProvider);
  return repository.getSettings();
});

final settingsStreamProvider = StreamProvider<AppSettings>((ref) {
  final repository = ref.watch(settingsRepositoryProvider);
  return repository.watchSettings();
});
```

---

## Usage in Other Modules

### Getting UI Language for Gloss

```dart
// In text_sequences presentation
final settings = await ref.read(appSettingsProvider.future);
final gloss = textSequence.gloss[settings.uiLanguageCode]
    ?? textSequence.gloss['en']
    ?? '';
```

### Getting Target Language for ASR

```dart
// In scoring module
final settings = await ref.read(appSettingsProvider.future);
final result = await recognizer.recognize(
  audioPath,
  languageCode: settings.targetLanguageCode,
);
```

### Configuring TTS/Playback

```dart
// In example_audio module
final settings = await ref.read(appSettingsProvider.future);
await player.setSpeed(settings.playbackSpeed);
```

---

## Integration Tests

### Settings Repository Tests

```dart
void main() {
  late Box<dynamic> box;
  late SettingsRepository repository;

  setUp(() async {
    await Hive.initFlutter();
    box = await Hive.openBox<dynamic>('test_settings');
    repository = SettingsRepositoryImpl(box);
  });

  tearDown(() async {
    await box.clear();
    await box.close();
  });

  group('SettingsRepository', () {
    test('returns defaults when no settings stored', () async {
      final settings = await repository.getSettings();

      expect(settings.uiLanguageCode, 'en');
      expect(settings.targetLanguageCode, 'zh-CN');
      expect(settings.showRomanization, isTrue);
    });

    test('persists settings', () async {
      final newSettings = AppSettings(
        uiLanguageCode: 'de',
        targetLanguageCode: 'ja-JP',
        showRomanization: false,
      );

      await repository.updateSettings(newSettings);
      final retrieved = await repository.getSettings();

      expect(retrieved.uiLanguageCode, 'de');
      expect(retrieved.targetLanguageCode, 'ja-JP');
      expect(retrieved.showRomanization, isFalse);
    });

    test('reset returns to defaults', () async {
      await repository.updateSettings(AppSettings(uiLanguageCode: 'de'));
      await repository.resetToDefaults();

      final settings = await repository.getSettings();
      expect(settings.uiLanguageCode, 'en');
    });

    test('watch emits on changes', () async {
      final stream = repository.watchSettings();
      final future = stream.first;

      await repository.updateSettings(AppSettings(uiLanguageCode: 'fr'));

      final emitted = await future;
      expect(emitted.uiLanguageCode, 'fr');
    });
  });
}
```

### Settings Integration Test

```dart
void main() {
  group('Settings integration', () {
    test('gloss selection uses uiLanguageCode', () async {
      final settings = AppSettings(uiLanguageCode: 'de');
      final sequence = TextSequence(
        id: 'ts_001',
        text: '你好',
        language: 'zh-CN',
        gloss: {
          'en': 'Hello',
          'de': 'Hallo',
        },
      );

      final gloss = sequence.gloss[settings.uiLanguageCode]
          ?? sequence.gloss['en']
          ?? '';

      expect(gloss, 'Hallo');
    });

    test('falls back to English when language not available', () async {
      final settings = AppSettings(uiLanguageCode: 'fr'); // Not in gloss
      final sequence = TextSequence(
        id: 'ts_001',
        text: '你好',
        language: 'zh-CN',
        gloss: {
          'en': 'Hello',
          'de': 'Hallo',
        },
      );

      final gloss = sequence.gloss[settings.uiLanguageCode]
          ?? sequence.gloss['en']
          ?? '';

      expect(gloss, 'Hello'); // Falls back to English
    });
  });
}
```

---

## Notes

### Language Codes

| Setting | Format | Examples |
|---------|--------|----------|
| `uiLanguageCode` | ISO 639-1 | "en", "de", "zh", "ja" |
| `targetLanguageCode` | BCP-47 | "zh-CN", "zh-TW", "ja-JP" |

### Why Separate UI and Target Languages?

- **UI Language**: What the user reads (interface, glosses)
- **Target Language**: What the user is learning (ASR, TTS)

Example: German speaker learning Chinese
- `uiLanguageCode: "de"` - Show German translations
- `targetLanguageCode: "zh-CN"` - Recognize Chinese speech

### Future Settings Screen

A settings screen could include:
- Language selection dropdowns
- Toggle switches for show/hide options
- Slider for playback speed
- Voice preference radio buttons

For MVP, these defaults are sufficient.

### Migration Strategy

If settings schema changes:
1. Check `schemaVersion` field (add if needed)
2. Migrate old format to new
3. Default missing fields to sensible values
