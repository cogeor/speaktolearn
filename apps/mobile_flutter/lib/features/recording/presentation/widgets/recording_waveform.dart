import 'package:audio_waveforms/audio_waveforms.dart';
import 'package:flutter/material.dart';

/// Widget that displays a real-time audio waveform during recording.
///
/// Wraps the [AudioWaveforms] widget from the audio_waveforms package
/// and applies consistent styling.
class RecordingWaveform extends StatelessWidget {
  const RecordingWaveform({
    super.key,
    required this.controller,
    this.height = 60.0,
  });

  /// The recorder controller that provides amplitude data.
  final RecorderController controller;

  /// Height of the waveform visualization.
  final double height;

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;

    return SizedBox(
      height: height,
      child: AudioWaveforms(
        recorderController: controller,
        size: Size(MediaQuery.of(context).size.width - 48, height),
        waveStyle: WaveStyle(
          waveColor: colorScheme.primary,
          extendWaveform: true,
          showMiddleLine: false,
          spacing: 4.0,
          waveThickness: 3.0,
          showDurationLabel: false,
          durationStyle: TextStyle(
            color: colorScheme.onSurface,
            fontSize: 12,
          ),
        ),
      ),
    );
  }
}
