import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../../app/di.dart';
import '../../stats/presentation/stats_controller.dart';
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
                subtitle: const Text(
                  'Display romanization when viewing sentences',
                ),
                value: settings.showRomanization,
                onChanged: (value) {
                  controller.updateShowPinyinByDefault(value);
                },
              ),
              const Divider(),
              // Debug section - only visible in debug builds
              if (kDebugMode) ...[
                const SizedBox(height: 16),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  child: Text(
                    'Debug',
                    style: Theme.of(context).textTheme.titleSmall?.copyWith(
                      color: Theme.of(context).colorScheme.primary,
                    ),
                  ),
                ),
                ListTile(
                  leading: const Icon(Icons.bug_report),
                  title: const Text('Generate Fake Stats'),
                  subtitle: const Text('60 days, 10 attempts/day'),
                  onTap: () => _generateFakeStats(context, ref),
                ),
                ListTile(
                  leading: const Icon(Icons.delete_forever),
                  title: const Text('Clear All Stats'),
                  subtitle: const Text('Remove all progress data'),
                  onTap: () => _clearStats(context, ref),
                ),
              ],
            ],
          );
        },
      ),
    );
  }

  Future<void> _generateFakeStats(BuildContext context, WidgetRef ref) async {
    // Show loading dialog
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => const AlertDialog(
        content: Row(
          children: [
            CircularProgressIndicator(),
            SizedBox(width: 24),
            Text('Generating fake stats...'),
          ],
        ),
      ),
    );

    try {
      final textSequenceRepo = ref.read(textSequenceRepositoryProvider);
      final progressRepo = ref.read(progressRepositoryProvider);

      // Get all sequences and pick a subset (up to 80 sequences)
      final allSequences = await textSequenceRepo.getAll();
      final sequenceIds = allSequences.take(80).map((s) => s.id).toList();

      await progressRepo.generateFakeStats(sequenceIds: sequenceIds);

      // Refresh stats
      ref.invalidate(statsControllerProvider);

      if (context.mounted) {
        Navigator.of(context).pop(); // Close loading dialog
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Fake stats generated successfully!'),
            backgroundColor: Colors.green,
          ),
        );
      }
    } catch (e) {
      if (context.mounted) {
        Navigator.of(context).pop(); // Close loading dialog
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: $e'), backgroundColor: Colors.red),
        );
      }
    }
  }

  Future<void> _clearStats(BuildContext context, WidgetRef ref) async {
    // Show confirmation dialog
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Clear All Stats?'),
        content: const Text(
          'This will remove all practice progress and attempt history. '
          'This action cannot be undone.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () => Navigator.of(context).pop(true),
            style: TextButton.styleFrom(foregroundColor: Colors.red),
            child: const Text('Clear'),
          ),
        ],
      ),
    );

    if (confirmed != true) return;

    try {
      final progressRepo = ref.read(progressRepositoryProvider);
      await progressRepo.clearAllStats();

      // Refresh stats
      ref.invalidate(statsControllerProvider);

      if (context.mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(const SnackBar(content: Text('All stats cleared.')));
      }
    } catch (e) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: $e'), backgroundColor: Colors.red),
        );
      }
    }
  }

  String _voicePreferenceLabel(VoicePreference pref) {
    switch (pref) {
      case VoicePreference.noPreference:
        return 'No Preference';
      case VoicePreference.male:
        return 'Male';
      case VoicePreference.female:
        return 'Female';
    }
  }
}
