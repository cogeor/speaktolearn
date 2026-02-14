import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:record/record.dart' as record_pkg;
import 'package:uuid/uuid.dart';

import '../../../core/result.dart';
import '../domain/audio_recorder.dart';

/// AudioRecorder implementation using the record package.
class RecordPluginRecorder implements AudioRecorder {
  RecordPluginRecorder() : _recorder = record_pkg.AudioRecorder();

  final record_pkg.AudioRecorder _recorder;
  String? _currentPath;
  static const _uuid = Uuid();

  @override
  bool get isRecording => _currentPath != null;

  @override
  Future<Result<String, RecordingError>> start() async {
    if (isRecording) {
      return const Failure(RecordingError.alreadyRecording);
    }

    // Check permission
    final status = await Permission.microphone.request();
    if (!status.isGranted) {
      return const Failure(RecordingError.permissionDenied);
    }

    // Check if recorder is available
    if (!await _recorder.hasPermission()) {
      return const Failure(RecordingError.noMicrophone);
    }

    try {
      final tempDir = await getTemporaryDirectory();
      // Use .wav extension for ML-compatible audio format
      final fileName = '${_uuid.v4()}.wav';
      _currentPath = '${tempDir.path}/$fileName';

      await _recorder.start(
        const record_pkg.RecordConfig(
          // WAV encoder for uncompressed PCM audio
          encoder: record_pkg.AudioEncoder.wav,
          // 16kHz sample rate for ML model compatibility
          sampleRate: 16000,
          // Mono audio (1 channel) for ML processing
          numChannels: 1,
          // 16-bit PCM (default for WAV)
          bitRate: 256000, // Higher bitrate for WAV
        ),
        path: _currentPath!,
      );

      return Success(_currentPath!);
    } catch (e) {
      _currentPath = null;
      return const Failure(RecordingError.recordingFailed);
    }
  }

  @override
  Future<Result<String, RecordingError>> stop() async {
    if (!isRecording) {
      return const Failure(RecordingError.notRecording);
    }

    try {
      final path = await _recorder.stop();
      final resultPath = _currentPath!;
      _currentPath = null;
      return Success(path ?? resultPath);
    } catch (e) {
      _currentPath = null;
      return const Failure(RecordingError.recordingFailed);
    }
  }

  @override
  Future<void> cancel() async {
    if (isRecording) {
      await _recorder.cancel();
      _currentPath = null;
    }
  }

  @override
  Future<void> dispose() async {
    await _recorder.dispose();
    _currentPath = null;
  }
}
