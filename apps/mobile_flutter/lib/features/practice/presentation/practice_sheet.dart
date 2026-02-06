import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../text_sequences/domain/text_sequence.dart';
import '../../progress/domain/text_sequence_progress.dart';

/// Bottom sheet for practicing a text sequence.
class PracticeSheet extends ConsumerStatefulWidget {
  const PracticeSheet({
    super.key,
    required this.sequence,
    this.progress,
  });

  final TextSequence sequence;
  final TextSequenceProgress? progress;

  @override
  ConsumerState<PracticeSheet> createState() => _PracticeSheetState();
}

class _PracticeSheetState extends ConsumerState<PracticeSheet> {
  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // Drag handle
            Container(
              width: 40,
              height: 4,
              decoration: BoxDecoration(
                color: Colors.grey[400],
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            const SizedBox(height: 24),
            // Text display
            Text(
              widget.sequence.text,
              style: Theme.of(context).textTheme.bodyLarge,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }
}
