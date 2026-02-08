import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../settings/presentation/settings_controller.dart';

/// A dropdown picker for selecting the current HSK level (1-6).
///
/// Displays the current level and allows the user to change it.
/// Level changes are persisted to settings and trigger a sentence reload.
class LevelPicker extends ConsumerWidget {
  const LevelPicker({
    super.key,
    this.onLevelChanged,
  });

  /// Optional callback when level changes. Called after settings are updated.
  final void Function(int level)? onLevelChanged;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final settingsAsync = ref.watch(settingsControllerProvider);
    final currentLevel = settingsAsync.valueOrNull?.currentLevel ?? 1;

    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          'Level',
          style: Theme.of(context).textTheme.titleMedium,
        ),
        const SizedBox(width: 12),
        DropdownButton<int>(
          value: currentLevel,
          underline: const SizedBox.shrink(),
          borderRadius: BorderRadius.circular(8),
          items: List.generate(6, (index) {
            final level = index + 1;
            return DropdownMenuItem<int>(
              value: level,
              child: Text(
                'HSK $level',
                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
              ),
            );
          }),
          onChanged: (newLevel) async {
            if (newLevel != null && newLevel != currentLevel) {
              await ref
                  .read(settingsControllerProvider.notifier)
                  .updateCurrentLevel(newLevel);
              onLevelChanged?.call(newLevel);
            }
          },
        ),
      ],
    );
  }
}
