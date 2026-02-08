import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../app/di.dart';
import '../domain/app_settings.dart';
import '../domain/settings_repository.dart';

/// Provider for the settings controller.
final settingsControllerProvider =
    AsyncNotifierProvider<SettingsController, AppSettings>(
      SettingsController.new,
    );

/// Controller for managing application settings.
class SettingsController extends AsyncNotifier<AppSettings> {
  late SettingsRepository _repository;

  @override
  Future<AppSettings> build() async {
    _repository = ref.watch(settingsRepositoryProvider);
    return _repository.getSettings();
  }

  /// Updates the voice preference setting.
  Future<void> updateVoicePreference(VoicePreference preference) async {
    final current = state.value ?? const AppSettings();
    final updated = current.copyWith(voicePreference: preference);
    await _repository.updateSettings(updated);
    state = AsyncValue.data(updated);
  }

  /// Updates whether to show pinyin by default.
  Future<void> updateShowPinyinByDefault(bool value) async {
    final current = state.value ?? const AppSettings();
    final updated = current.copyWith(showRomanization: value);
    await _repository.updateSettings(updated);
    state = AsyncValue.data(updated);
  }

  /// Updates the current HSK level (1-6).
  Future<void> updateCurrentLevel(int level) async {
    assert(level >= 1 && level <= 6, 'Level must be between 1 and 6');
    final current = state.value ?? const AppSettings();
    final updated = current.copyWith(currentLevel: level.clamp(1, 6));
    await _repository.updateSettings(updated);
    state = AsyncValue.data(updated);
  }
}
