import 'package:flutter/material.dart';

import '../../../app/theme.dart';
import 'sequence_list_item.dart';

/// A list tile widget for displaying a sequence item.
class SequenceListTile extends StatelessWidget {
  /// Creates a SequenceListTile.
  const SequenceListTile({
    super.key,
    required this.item,
    required this.onTap,
    required this.onToggleTrack,
  });

  /// The sequence item to display.
  final SequenceListItem item;

  /// Called when the tile is tapped.
  final VoidCallback onTap;

  /// Called when the track/untrack button is pressed.
  final VoidCallback onToggleTrack;

  @override
  Widget build(BuildContext context) {
    return ListTile(
      title: Text(item.text),
      subtitle: item.bestScore != null
          ? Text(
              'Best: ${item.bestScore}',
              style: TextStyle(color: item.bestScore!.scoreColor),
            )
          : null,
      trailing: IconButton(
        icon: Icon(item.isTracked ? Icons.star : Icons.star_border),
        onPressed: onToggleTrack,
      ),
      onTap: onTap,
    );
  }
}
