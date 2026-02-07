import 'package:flutter/material.dart';

import '../../../../app/theme.dart';
import '../home_state.dart';

/// Floating action button for recording with visual state feedback.
///
/// Displays different icons and colors based on [RecordingStatus]:
/// - idle: mic icon, primary color
/// - recording: stop icon, red with pulse animation
/// - processing: circular progress indicator
class RecordFAB extends StatefulWidget {
  const RecordFAB({
    super.key,
    required this.status,
    required this.onPressed,
  });

  final RecordingStatus status;
  final VoidCallback? onPressed;

  @override
  State<RecordFAB> createState() => _RecordFABState();
}

class _RecordFABState extends State<RecordFAB>
    with SingleTickerProviderStateMixin {
  late AnimationController _pulseController;
  late Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1000),
    );
    _pulseAnimation = Tween<double>(begin: 1.0, end: 1.15).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );
  }

  @override
  void didUpdateWidget(RecordFAB oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.status == RecordingStatus.recording) {
      _pulseController.repeat(reverse: true);
    } else {
      _pulseController.stop();
      _pulseController.reset();
    }
  }

  @override
  void dispose() {
    _pulseController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _pulseAnimation,
      builder: (context, child) {
        return Transform.scale(
          scale: widget.status == RecordingStatus.recording
              ? _pulseAnimation.value
              : 1.0,
          child: FloatingActionButton(
            onPressed: widget.status == RecordingStatus.processing
                ? null
                : widget.onPressed,
            backgroundColor: _backgroundColor,
            foregroundColor: _foregroundColor,
            child: _buildIcon(),
          ),
        );
      },
    );
  }

  Color get _backgroundColor {
    switch (widget.status) {
      case RecordingStatus.idle:
        return AppTheme.foreground;
      case RecordingStatus.recording:
        return AppTheme.error;
      case RecordingStatus.processing:
        return AppTheme.subtle;
    }
  }

  Color get _foregroundColor {
    switch (widget.status) {
      case RecordingStatus.idle:
        return AppTheme.background;
      case RecordingStatus.recording:
        return AppTheme.foreground;
      case RecordingStatus.processing:
        return AppTheme.foreground;
    }
  }

  Widget _buildIcon() {
    switch (widget.status) {
      case RecordingStatus.idle:
        return const Icon(Icons.mic, size: 28);
      case RecordingStatus.recording:
        return const Icon(Icons.stop, size: 28);
      case RecordingStatus.processing:
        return const SizedBox(
          width: 24,
          height: 24,
          child: CircularProgressIndicator(
            strokeWidth: 2.5,
            color: AppTheme.foreground,
          ),
        );
    }
  }
}
