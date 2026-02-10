import 'package:flutter/material.dart';

import 'sequence_list_item.dart';
import 'widgets/rating_indicator.dart';

/// A list tile widget for displaying a sequence item.
class SequenceListTile extends StatelessWidget {
  /// Creates a SequenceListTile.
  const SequenceListTile({super.key, required this.item, required this.onTap});

  /// The sequence item to display.
  final SequenceListItem item;

  /// Called when the tile is tapped.
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return ListTile(
      leading: RatingIndicator(rating: item.lastRating, size: 20),
      title: Text(item.text),
      onTap: onTap,
    );
  }
}
