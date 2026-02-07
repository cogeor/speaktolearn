import 'package:flutter/material.dart';

/// A row of filter chips for selecting HSK levels.
///
/// Displays an "All" chip to show all sequences, plus chips for each
/// HSK level (1-6). Multiple levels can be selected simultaneously.
class HskFilterChips extends StatelessWidget {
  /// Creates an HskFilterChips widget.
  const HskFilterChips({
    super.key,
    required this.selectedLevels,
    required this.onToggle,
    required this.onClear,
  });

  /// Currently selected HSK levels.
  final Set<int> selectedLevels;

  /// Called when a level is toggled.
  final void Function(int level) onToggle;

  /// Called when filter is cleared (show all).
  final VoidCallback onClear;

  @override
  Widget build(BuildContext context) {
    final isAllSelected = selectedLevels.isEmpty;

    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Row(
        children: [
          // "All" chip
          Padding(
            padding: const EdgeInsets.only(right: 8),
            child: FilterChip(
              label: const Text('All'),
              selected: isAllSelected,
              onSelected: (_) => onClear(),
              selectedColor:
                  Theme.of(context).colorScheme.primary.withValues(alpha: 0.2),
              checkmarkColor: Theme.of(context).colorScheme.primary,
            ),
          ),
          // HSK level chips
          for (int level = 1; level <= 6; level++)
            Padding(
              padding: const EdgeInsets.only(right: 8),
              child: FilterChip(
                label: Text('HSK $level'),
                selected: selectedLevels.contains(level),
                onSelected: (_) => onToggle(level),
                selectedColor: _levelColor(level).withValues(alpha: 0.2),
                checkmarkColor: _levelColor(level),
                labelStyle: TextStyle(
                  color: selectedLevels.contains(level)
                      ? _levelColor(level)
                      : null,
                ),
              ),
            ),
        ],
      ),
    );
  }

  /// Returns the color associated with the HSK level.
  /// Gradient from green (easy) to red (hard).
  Color _levelColor(int level) {
    switch (level) {
      case 1:
        return const Color(0xFF4CAF50); // Green
      case 2:
        return const Color(0xFF8BC34A); // Light green
      case 3:
        return const Color(0xFFCDDC39); // Lime
      case 4:
        return const Color(0xFFFFEB3B); // Yellow
      case 5:
        return const Color(0xFFFF9800); // Orange
      case 6:
        return const Color(0xFFF44336); // Red
      default:
        return const Color(0xFF888888); // Gray for unknown
    }
  }
}
