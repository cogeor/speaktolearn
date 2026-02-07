import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../domain/app_settings.dart';
import 'settings_controller.dart';

/// Screen for managing application settings.
class SettingsScreen extends ConsumerWidget {
  /// Creates a SettingsScreen.
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final settingsAsync = ref.watch(settingsControllerProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Settings'),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => context.go('/'),
        ),
      ),
      body: settingsAsync.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (error, stack) => Center(child: Text('Error: $error')),
        data: (settings) {
          final controller = ref.read(settingsControllerProvider.notifier);
          return ListView(
            children: [
              // Voice preference section
              ListTile(
                leading: const Icon(Icons.record_voice_over),
                title: const Text('Voice Preference'),
                subtitle: Text(_voicePreferenceLabel(settings.voicePreference)),
                trailing: DropdownButton<VoicePreference>(
                  value: settings.voicePreference,
                  underline: const SizedBox.shrink(),
                  onChanged: (value) {
                    if (value != null) {
                      controller.updateVoicePreference(value);
                    }
                  },
                  items: VoicePreference.values.map((pref) {
                    return DropdownMenuItem(
                      value: pref,
                      child: Text(_voicePreferenceLabel(pref)),
                    );
                  }).toList(),
                ),
              ),
              const Divider(),
              SwitchListTile(
                secondary: const Icon(Icons.translate),
                title: const Text('Show Pinyin by Default'),
                subtitle: const Text('Display romanization when viewing sentences'),
                value: settings.showRomanization,
                onChanged: (value) {
                  controller.updateShowPinyinByDefault(value);
                },
              ),
              const Divider(),
            ],
          );
        },
      ),
    );
  }

  String _voicePreferenceLabel(VoicePreference pref) {
    switch (pref) {
      case VoicePreference.systemDefault:
        return 'System Default';
      case VoicePreference.male:
        return 'Male';
      case VoicePreference.female:
        return 'Female';
    }
  }
}
