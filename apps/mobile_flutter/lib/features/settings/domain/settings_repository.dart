import 'app_settings.dart';

abstract class SettingsRepository {
  Future<AppSettings> getSettings();

  Future<void> updateSettings(AppSettings settings);

  Future<void> resetToDefaults();

  Stream<AppSettings> watchSettings();
}
