# features/recording/ Module

## Purpose

Handles audio capture from the microphone and storage of user recordings. Each text sequence keeps only the latest recording (overwritten on each attempt).

## Folder Structure

```
recording/
├── domain/
│   ├── recording.dart                  # Entity
│   ├── audio_recorder.dart             # Recording interface
│   └── recording_repository.dart       # Storage interface
├── data/
│   ├── record_plugin_recorder.dart     # `record` package impl
│   └── recording_repository_impl.dart  # File system impl
└── presentation/
    └── recording_controller.dart       # State management
```

---

## Domain Layer

### `recording.dart`

**Purpose**: Represents a user's recorded audio.

**Implementation**:

```dart
import 'package:freezed_annotation/freezed_annotation.dart';

part 'recording.freezed.dart';

/// A user's audio recording for a text sequence.
@freezed
class Recording with _$Recording {
  const factory Recording({
    /// Unique identifier.
    required String id,

    /// The text sequence this recording is for.
    required String textSequenceId,

    /// When the recording was created.
    required DateTime createdAt,

    /// Path to the audio file.
    required String filePath,

    /// Duration in milliseconds.
    int? durationMs,

    /// Sample rate in Hz.
    int? sampleRate,

    /// MIME type (e.g., "audio/m4a").
    String? mimeType,
  }) = _Recording;
}
```

---

### `audio_recorder.dart`

**Purpose**: Interface for audio capture.

**Implementation**:

```dart
import '../../../core/result.dart';

/// Error types for recording operations.
enum RecordingError {
  permissionDenied,
  noMicrophone,
  alreadyRecording,
  notRecording,
  encodingFailed,
  unknown,
}

/// Interface for audio recording.
abstract class AudioRecorder {
  /// Whether recording is currently in progress.
  bool get isRecording;

  /// Starts recording to a temporary file.
  ///
  /// Returns the path where audio will be saved.
  Future<Result<String, RecordingError>> start();

  /// Stops recording and finalizes the file.
  ///
  /// Returns the path to the recorded file.
  Future<Result<String, RecordingError>> stop();

  /// Cancels recording and deletes the temporary file.
  Future<void> cancel();

  /// Disposes resources.
  Future<void> dispose();
}
```

---

### `recording_repository.dart`

**Purpose**: Interface for storing and retrieving recordings.

**Implementation**:

```dart
import 'recording.dart';

/// Repository for user recordings.
///
/// Only the latest recording per text sequence is kept.
/// New recordings replace old ones.
abstract class RecordingRepository {
  /// Saves a recording, replacing any existing one for this sequence.
  ///
  /// [tempPath] is the temporary file from the recorder.
  /// Returns the saved Recording with its permanent path.
  Future<Recording> saveLatest(String textSequenceId, String tempPath);

  /// Gets the latest recording for a sequence, or null if none exists.
  Future<Recording?> getLatest(String textSequenceId);

  /// Deletes the recording for a sequence.
  Future<void> deleteLatest(String textSequenceId);

  /// Checks if a recording exists for a sequence.
  Future<bool> hasRecording(String textSequenceId);
}
```

---

## Data Layer

### `record_plugin_recorder.dart`

**Purpose**: Implementation using the `record` package.

**Implementation**:

```dart
import 'package:record/record.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:uuid/uuid.dart';
import '../../../core/result.dart';
import '../domain/audio_recorder.dart';

class RecordPluginRecorder implements AudioRecorder {
  final AudioRecorder _recorder;
  final Uuid _uuid;

  String? _currentPath;
  bool _isRecording = false;

  RecordPluginRecorder()
      : _recorder = AudioRecorder(),
        _uuid = const Uuid();

  @override
  bool get isRecording => _isRecording;

  @override
  Future<Result<String, RecordingError>> start() async {
    if (_isRecording) {
      return const Failure(RecordingError.alreadyRecording);
    }

    // Check permission
    final status = await Permission.microphone.request();
    if (!status.isGranted) {
      return const Failure(RecordingError.permissionDenied);
    }

    // Check if device has microphone
    final hasPermission = await _recorder.hasPermission();
    if (!hasPermission) {
      return const Failure(RecordingError.noMicrophone);
    }

    try {
      // Generate temp file path
      final tempDir = await getTemporaryDirectory();
      final fileName = '${_uuid.v4()}.m4a';
      _currentPath = '${tempDir.path}/$fileName';

      // Start recording
      await _recorder.start(
        RecordConfig(
          encoder: AudioEncoder.aacLc,
          sampleRate: 44100,
          bitRate: 128000,
        ),
        path: _currentPath!,
      );

      _isRecording = true;
      return Success(_currentPath!);
    } catch (e) {
      return const Failure(RecordingError.unknown);
    }
  }

  @override
  Future<Result<String, RecordingError>> stop() async {
    if (!_isRecording) {
      return const Failure(RecordingError.notRecording);
    }

    try {
      final path = await _recorder.stop();
      _isRecording = false;

      if (path == null) {
        return const Failure(RecordingError.encodingFailed);
      }

      return Success(path);
    } catch (e) {
      _isRecording = false;
      return const Failure(RecordingError.unknown);
    }
  }

  @override
  Future<void> cancel() async {
    if (_isRecording) {
      await _recorder.stop();
      _isRecording = false;
    }
    // Delete temp file if exists
    if (_currentPath != null) {
      final file = File(_currentPath!);
      if (await file.exists()) {
        await file.delete();
      }
      _currentPath = null;
    }
  }

  @override
  Future<void> dispose() async {
    await cancel();
    await _recorder.dispose();
  }
}
```

---

### `recording_repository_impl.dart`

**Purpose**: File system storage for recordings.

**Implementation**:

```dart
import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:uuid/uuid.dart';
import '../domain/recording.dart';
import '../domain/recording_repository.dart';

class RecordingRepositoryImpl implements RecordingRepository {
  final Uuid _uuid;

  /// In-memory cache of recording metadata.
  /// Key: textSequenceId, Value: Recording
  final Map<String, Recording> _cache = {};

  RecordingRepositoryImpl() : _uuid = const Uuid();

  Future<String> get _recordingsDir async {
    final appDir = await getApplicationDocumentsDirectory();
    final dir = Directory('${appDir.path}/recordings');
    if (!await dir.exists()) {
      await dir.create(recursive: true);
    }
    return dir.path;
  }

  String _fileName(String textSequenceId) => '$textSequenceId.m4a';

  @override
  Future<Recording> saveLatest(String textSequenceId, String tempPath) async {
    final dir = await _recordingsDir;
    final destPath = '$dir/${_fileName(textSequenceId)}';

    // Delete existing file if present
    final destFile = File(destPath);
    if (await destFile.exists()) {
      await destFile.delete();
    }

    // Move temp file to permanent location
    final tempFile = File(tempPath);
    await tempFile.rename(destPath);

    // Create recording entity
    final recording = Recording(
      id: _uuid.v4(),
      textSequenceId: textSequenceId,
      createdAt: DateTime.now(),
      filePath: destPath,
      mimeType: 'audio/m4a',
    );

    _cache[textSequenceId] = recording;
    return recording;
  }

  @override
  Future<Recording?> getLatest(String textSequenceId) async {
    // Check cache first
    if (_cache.containsKey(textSequenceId)) {
      return _cache[textSequenceId];
    }

    // Check file system
    final dir = await _recordingsDir;
    final path = '$dir/${_fileName(textSequenceId)}';
    final file = File(path);

    if (await file.exists()) {
      final stat = await file.stat();
      final recording = Recording(
        id: textSequenceId, // Use textSequenceId as ID for simplicity
        textSequenceId: textSequenceId,
        createdAt: stat.modified,
        filePath: path,
        mimeType: 'audio/m4a',
      );
      _cache[textSequenceId] = recording;
      return recording;
    }

    return null;
  }

  @override
  Future<void> deleteLatest(String textSequenceId) async {
    _cache.remove(textSequenceId);

    final dir = await _recordingsDir;
    final path = '$dir/${_fileName(textSequenceId)}';
    final file = File(path);

    if (await file.exists()) {
      await file.delete();
    }
  }

  @override
  Future<bool> hasRecording(String textSequenceId) async {
    if (_cache.containsKey(textSequenceId)) {
      return true;
    }

    final dir = await _recordingsDir;
    final path = '$dir/${_fileName(textSequenceId)}';
    return File(path).exists();
  }
}
```

---

## Presentation Layer

### `recording_controller.dart`

**Purpose**: State management for recording operations.

**Implementation**:

```dart
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:freezed_annotation/freezed_annotation.dart';
import '../domain/audio_recorder.dart';
import '../domain/recording_repository.dart';
import '../domain/recording.dart';
import '../../scoring/domain/pronunciation_scorer.dart';
import '../../scoring/domain/grade.dart';
import '../../progress/domain/progress_repository.dart';
import '../../progress/domain/score_attempt.dart';
import '../../text_sequences/domain/text_sequence.dart';

part 'recording_controller.freezed.dart';

@freezed
class RecordingState with _$RecordingState {
  const factory RecordingState({
    @Default(false) bool isRecording,
    @Default(false) bool isScoring,
    @Default(false) bool isPlaying,
    @Default(false) bool hasLatestRecording,
    String? error,
  }) = _RecordingState;
}

final recordingControllerProvider =
    StateNotifierProvider<RecordingController, RecordingState>(
  (ref) => RecordingController(
    recorder: ref.watch(audioRecorderProvider),
    repository: ref.watch(recordingRepositoryProvider),
    scorer: ref.watch(pronunciationScorerProvider),
    progress: ref.watch(progressRepositoryProvider),
  ),
);

class RecordingController extends StateNotifier<RecordingState> {
  final AudioRecorder _recorder;
  final RecordingRepository _repository;
  final PronunciationScorer _scorer;
  final ProgressRepository _progress;

  String? _currentSequenceId;

  RecordingController({
    required AudioRecorder recorder,
    required RecordingRepository repository,
    required PronunciationScorer scorer,
    required ProgressRepository progress,
  })  : _recorder = recorder,
        _repository = repository,
        _scorer = scorer,
        _progress = progress,
        super(const RecordingState());

  /// Starts recording for a text sequence.
  Future<void> startRecording(String textSequenceId) async {
    _currentSequenceId = textSequenceId;

    final result = await _recorder.start();
    result.when(
      success: (_) {
        state = state.copyWith(isRecording: true, error: null);
      },
      failure: (error) {
        state = state.copyWith(
          isRecording: false,
          error: _errorMessage(error),
        );
      },
    );
  }

  /// Stops recording, saves, scores, and returns the result.
  Future<ScoreResult?> stopAndScore(TextSequence textSequence) async {
    if (!state.isRecording || _currentSequenceId != textSequence.id) {
      return null;
    }

    // Stop recording
    final stopResult = await _recorder.stop();
    if (stopResult.isFailure) {
      state = state.copyWith(
        isRecording: false,
        error: _errorMessage(stopResult.errorOrNull!),
      );
      return null;
    }

    final tempPath = stopResult.valueOrNull!;
    state = state.copyWith(isRecording: false, isScoring: true);

    try {
      // Save recording
      final recording = await _repository.saveLatest(textSequence.id, tempPath);

      // Score
      final grade = await _scorer.score(textSequence, recording);

      // Save attempt
      final attempt = ScoreAttempt(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        textSequenceId: textSequence.id,
        gradedAt: DateTime.now(),
        score: grade.overall,
        method: grade.method,
        recognizedText: grade.recognizedText,
        details: grade.details,
      );
      await _progress.saveAttempt(attempt);

      state = state.copyWith(
        isScoring: false,
        hasLatestRecording: true,
      );

      return ScoreResult(score: grade.overall, grade: grade);
    } catch (e) {
      state = state.copyWith(
        isScoring: false,
        error: 'Scoring failed: $e',
      );
      return null;
    }
  }

  /// Checks if a recording exists for a sequence.
  Future<void> checkLatestRecording(String textSequenceId) async {
    final hasRecording = await _repository.hasRecording(textSequenceId);
    state = state.copyWith(hasLatestRecording: hasRecording);
  }

  /// Replays the latest recording for a sequence.
  Future<void> replayLatest(String textSequenceId) async {
    final recording = await _repository.getLatest(textSequenceId);
    if (recording == null) return;

    // Use audio player to play the file
    // (Would need to inject AudioPlayer)
    state = state.copyWith(isPlaying: true);

    // ... play audio ...

    state = state.copyWith(isPlaying: false);
  }

  /// Cancels an in-progress recording.
  Future<void> cancel() async {
    await _recorder.cancel();
    state = state.copyWith(isRecording: false);
  }

  String _errorMessage(RecordingError error) => switch (error) {
        RecordingError.permissionDenied => 'Microphone permission denied',
        RecordingError.noMicrophone => 'No microphone available',
        RecordingError.alreadyRecording => 'Already recording',
        RecordingError.notRecording => 'Not recording',
        RecordingError.encodingFailed => 'Encoding failed',
        RecordingError.unknown => 'Unknown error',
      };
}

/// Result of a scoring operation.
class ScoreResult {
  final double score;
  final Grade grade;

  const ScoreResult({required this.score, required this.grade});
}
```

---

## Integration Tests

### Recorder Integration Test

```dart
void main() {
  group('AudioRecorder', () {
    late AudioRecorder recorder;

    setUp(() {
      recorder = RecordPluginRecorder();
    });

    tearDown(() async {
      await recorder.dispose();
    });

    test('starts and stops recording', () async {
      final startResult = await recorder.start();
      expect(startResult.isSuccess, isTrue);
      expect(recorder.isRecording, isTrue);

      await Future.delayed(Duration(seconds: 1));

      final stopResult = await recorder.stop();
      expect(stopResult.isSuccess, isTrue);
      expect(recorder.isRecording, isFalse);

      // Verify file exists
      final path = stopResult.valueOrNull!;
      expect(await File(path).exists(), isTrue);
    });

    test('cancel deletes temp file', () async {
      final startResult = await recorder.start();
      final path = startResult.valueOrNull!;

      await recorder.cancel();

      expect(recorder.isRecording, isFalse);
      expect(await File(path).exists(), isFalse);
    });

    test('cannot start while already recording', () async {
      await recorder.start();
      final secondStart = await recorder.start();

      expect(secondStart.errorOrNull, RecordingError.alreadyRecording);
    });
  });
}
```

### Repository Integration Test

```dart
void main() {
  group('RecordingRepository', () {
    late RecordingRepository repository;
    late Directory tempDir;

    setUp(() async {
      tempDir = await Directory.systemTemp.createTemp('test_recordings');
      repository = RecordingRepositoryImpl();
    });

    tearDown(() async {
      await tempDir.delete(recursive: true);
    });

    test('saves and retrieves recording', () async {
      // Create a temp file to simulate recorded audio
      final tempFile = File('${tempDir.path}/temp.m4a');
      await tempFile.writeAsString('audio data');

      final recording = await repository.saveLatest('ts_001', tempFile.path);

      expect(recording.textSequenceId, 'ts_001');
      expect(await File(recording.filePath).exists(), isTrue);

      final retrieved = await repository.getLatest('ts_001');
      expect(retrieved, isNotNull);
      expect(retrieved!.filePath, recording.filePath);
    });

    test('replaces existing recording', () async {
      final file1 = File('${tempDir.path}/temp1.m4a');
      await file1.writeAsString('audio 1');
      final recording1 = await repository.saveLatest('ts_001', file1.path);

      final file2 = File('${tempDir.path}/temp2.m4a');
      await file2.writeAsString('audio 2');
      final recording2 = await repository.saveLatest('ts_001', file2.path);

      // Should be different recording IDs
      expect(recording2.id, isNot(recording1.id));

      // Only latest should exist
      final latest = await repository.getLatest('ts_001');
      expect(latest!.id, recording2.id);
    });

    test('delete removes recording', () async {
      final tempFile = File('${tempDir.path}/temp.m4a');
      await tempFile.writeAsString('audio data');
      await repository.saveLatest('ts_001', tempFile.path);

      await repository.deleteLatest('ts_001');

      expect(await repository.hasRecording('ts_001'), isFalse);
      expect(await repository.getLatest('ts_001'), isNull);
    });
  });
}
```

### Recording Flow Integration Test

```dart
void main() {
  group('Recording flow', () {
    testWidgets('complete record-score-save flow', (tester) async {
      // Mock dependencies
      final mockRecorder = MockAudioRecorder();
      final mockRepository = MockRecordingRepository();
      final mockScorer = MockPronunciationScorer();
      final mockProgress = MockProgressRepository();

      when(() => mockRecorder.start()).thenAnswer(
        (_) async => Success('/tmp/recording.m4a'),
      );
      when(() => mockRecorder.stop()).thenAnswer(
        (_) async => Success('/tmp/recording.m4a'),
      );
      when(() => mockRepository.saveLatest(any(), any())).thenAnswer(
        (_) async => Recording(
          id: '1',
          textSequenceId: 'ts_001',
          createdAt: DateTime.now(),
          filePath: '/recordings/ts_001.m4a',
        ),
      );
      when(() => mockScorer.score(any(), any())).thenAnswer(
        (_) async => Grade(overall: 75.0, method: 'asr_cer_v1'),
      );

      final controller = RecordingController(
        recorder: mockRecorder,
        repository: mockRepository,
        scorer: mockScorer,
        progress: mockProgress,
      );

      final textSequence = TextSequence(
        id: 'ts_001',
        text: '你好',
        language: 'zh',
      );

      // Start recording
      await controller.startRecording('ts_001');
      expect(controller.state.isRecording, isTrue);

      // Stop and score
      final result = await controller.stopAndScore(textSequence);

      expect(result, isNotNull);
      expect(result!.score, 75.0);
      expect(controller.state.isRecording, isFalse);
      expect(controller.state.isScoring, isFalse);

      // Verify attempt was saved
      verify(() => mockProgress.saveAttempt(any())).called(1);
    });
  });
}
```

---

## Notes

### File Naming Convention

Recordings are stored as `{textSequenceId}.m4a`:
- Simple, predictable paths
- One file per sequence (overwritten)
- Easy to find/delete

### Audio Format

- **Format**: AAC in M4A container
- **Sample rate**: 44100 Hz
- **Bit rate**: 128 kbps

This provides good quality while keeping file sizes reasonable (~1 MB/minute).

### Why Replace Instead of Keep History?

1. **Storage**: Audio files are large; keeping all attempts would grow quickly
2. **Simplicity**: Only need latest for replay feature
3. **Privacy**: Users might want to discard bad attempts

If history is needed, could add a "favorites" feature later.
