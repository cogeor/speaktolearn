import 'package:freezed_annotation/freezed_annotation.dart';

import '../../../core/domain/use_case.dart';
import 'audio_recorder.dart';

part 'record_audio_use_case.freezed.dart';

/// Parameters for starting a recording.
@freezed
class RecordAudioParams with _$RecordAudioParams {
  const factory RecordAudioParams({required String textSequenceId}) =
      _RecordAudioParams;
}

/// Result of a recording operation.
sealed class RecordAudioResult {
  const RecordAudioResult();
}

/// Recording started successfully.
final class RecordingStarted extends RecordAudioResult {
  const RecordingStarted();
}

/// Recording stopped successfully with the file path.
final class RecordingStopped extends RecordAudioResult {
  final String filePath;
  const RecordingStopped(this.filePath);
}

/// Recording failed with an error.
final class RecordingFailed extends RecordAudioResult {
  final RecordingError error;
  const RecordingFailed(this.error);
}

/// Use case for starting an audio recording.
///
/// Encapsulates the logic for:
/// - Starting the recorder
/// - Handling permission errors
/// - Managing recording state
class StartRecordingUseCase
    extends FutureUseCase<RecordAudioParams, RecordAudioResult> {
  StartRecordingUseCase({required AudioRecorder recorder})
    : _recorder = recorder;

  final AudioRecorder _recorder;

  @override
  Future<RecordAudioResult> run(RecordAudioParams input) async {
    final result = await _recorder.start();

    return result.when(
      success: (_) => const RecordingStarted(),
      failure: (error) => RecordingFailed(error),
    );
  }
}

/// Use case for stopping an audio recording.
///
/// Returns the file path of the recorded audio on success.
class StopRecordingUseCase extends FutureUseCase<NoParams, RecordAudioResult> {
  StopRecordingUseCase({required AudioRecorder recorder})
    : _recorder = recorder;

  final AudioRecorder _recorder;

  @override
  Future<RecordAudioResult> run(NoParams input) async {
    final result = await _recorder.stop();

    return result.when(
      success: (filePath) => RecordingStopped(filePath),
      failure: (error) => RecordingFailed(error),
    );
  }
}
