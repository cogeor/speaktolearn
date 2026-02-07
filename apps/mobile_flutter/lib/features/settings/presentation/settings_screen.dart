import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

/// Screen for managing application settings.
class SettingsScreen extends ConsumerWidget {
  /// Creates a SettingsScreen.
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Settings'),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => context.go('/'),
        ),
      ),
      body: ListView(
        children: const [
          // Placeholder for settings items
          ListTile(
            leading: Icon(Icons.info_outline),
            title: Text('Settings coming soon'),
            subtitle: Text('Voice selection, pinyin visibility, and more'),
          ),
        ],
      ),
    );
  }
}
