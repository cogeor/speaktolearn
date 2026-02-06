import 'dart:async';

import 'package:hive/hive.dart';

import '../domain/app_settings.dart';
import '../domain/settings_repository.dart';

class SettingsRepositoryImpl implements SettingsRepository {
  SettingsRepositoryImpl(this._settingsBox) {
    // Seed the stream with current settings
    getSettings().then((settings) {
      if (!_settingsController.isClosed) {
        _settingsController.add(settings);
      }
    });
  }

  final Box<dynamic> _settingsBox;
  static const _settingsKey = 'app_settings';

  final _settingsController = StreamController<AppSettings>.broadcast();

  @override
  Future<AppSettings> getSettings() async {
    final json = _settingsBox.get(_settingsKey);
    if (json == null) {
      return const AppSettings();
    }
    return AppSettings.fromJson(Map<String, dynamic>.from(json as Map));
  }

  @override
  Future<void> updateSettings(AppSettings settings) async {
    await _settingsBox.put(_settingsKey, settings.toJson());
    if (!_settingsController.isClosed) {
      _settingsController.add(settings);
    }
  }

  @override
  Future<void> resetToDefaults() async {
    await updateSettings(const AppSettings());
  }

  @override
  Stream<AppSettings> watchSettings() {
    return _settingsController.stream;
  }

  void dispose() {
    _settingsController.close();
  }
}
