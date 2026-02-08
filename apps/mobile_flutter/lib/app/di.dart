import 'dart:convert';
import 'dart:io';

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:hive/hive.dart';
import 'package:path_provider/path_provider.dart';

import '../core/audio/audio_player.dart';
import '../core/storage/hive_boxes.dart';
import '../features/example_audio/data/example_audio_repository_impl.dart';
import '../features/example_audio/data/just_audio_player.dart';
import '../features/example_audio/domain/example_audio_repository.dart';
import '../features/progress/data/progress_repository_impl.dart';
import '../features/progress/domain/progress_repository.dart';
import '../features/recording/data/record_plugin_recorder.dart';
import '../features/recording/data/recording_repository_impl.dart';
import '../features/recording/domain/audio_recorder.dart';
import '../features/recording/domain/recording_repository.dart';
import '../features/scoring/data/asr_similarity_scorer.dart';
import '../features/scoring/data/cer_calculator.dart';
import '../features/scoring/data/speech_recognizer.dart';
import '../features/scoring/data/speech_to_text_recognizer.dart';
import '../features/scoring/domain/pronunciation_scorer.dart';
import '../features/selection/domain/get_next_by_level.dart';
import '../features/recording/presentation/recording_controller.dart';
import '../features/recording/presentation/recording_state.dart';
import '../features/settings/data/settings_repository_impl.dart';
import '../features/settings/domain/settings_repository.dart';
import '../features/text_sequences/data/dataset_source.dart';
import '../features/text_sequences/data/text_sequence_repository_impl.dart';
import '../features/text_sequences/domain/text_sequence_repository.dart';

/// Provider for Hive boxes. Override in createOverrides().
final hiveBoxProvider = Provider.family<Box<dynamic>, String>((ref, boxName) {
  throw UnimplementedError(
    'hiveBoxProvider must be overridden with actual boxes',
  );
});

/// Provider for audio player instance.
final audioPlayerProvider = Provider<AudioPlayer>((ref) {
  return JustAudioPlayer();
});

/// Provider for dataset source.
final datasetSourceProvider = Provider<DatasetSource>((ref) {
  return AssetDatasetSource();
});

/// Provider for text sequence repository.
final textSequenceRepositoryProvider = Provider<TextSequenceRepository>((ref) {
  final datasetSource = ref.watch(datasetSourceProvider);
  return TextSequenceRepositoryImpl(datasetSource);
});

/// Provider for progress repository.
final progressRepositoryProvider = Provider<ProgressRepository>((ref) {
  final progressBox = ref.watch(hiveBoxProvider(HiveBoxes.progress));
  final attemptsBox = ref.watch(hiveBoxProvider(HiveBoxes.attempts));
  return ProgressRepositoryImpl(
    progressBox: progressBox,
    attemptsBox: attemptsBox,
  );
});

/// Provider for recording repository.
final recordingRepositoryProvider = Provider<RecordingRepository>((ref) {
  return RecordingRepositoryImpl();
});

/// Provider for example audio repository.
final exampleAudioRepositoryProvider = Provider<ExampleAudioRepository>((ref) {
  return ExampleAudioRepositoryImpl();
});

/// Provider for settings repository.
final settingsRepositoryProvider = Provider<SettingsRepository>((ref) {
  final settingsBox = ref.watch(hiveBoxProvider(HiveBoxes.settings));
  return SettingsRepositoryImpl(settingsBox);
});

/// Provider for audio recorder.
final audioRecorderProvider = Provider<AudioRecorder>((ref) {
  return RecordPluginRecorder();
});

/// Provider for speech recognizer.
final speechRecognizerProvider = Provider<SpeechRecognizer>((ref) {
  return SpeechToTextRecognizer();
});

/// Provider for pronunciation scorer.
final pronunciationScorerProvider = Provider<PronunciationScorer>((ref) {
  final recognizer = ref.watch(speechRecognizerProvider);
  final calculator = CerCalculator();
  return AsrSimilarityScorer(recognizer: recognizer, calculator: calculator);
});

/// Provider for GetNextByLevel use case.
final getNextByLevelProvider = Provider<GetNextByLevel>((ref) {
  return GetNextByLevel(
    textSequenceRepository: ref.watch(textSequenceRepositoryProvider),
  );
});

/// Provider for recording controller.
final recordingControllerProvider =
    StateNotifierProvider<RecordingController, RecordingState>((ref) {
      return RecordingController(
        recorder: ref.watch(audioRecorderProvider),
        repository: ref.watch(recordingRepositoryProvider),
        scorer: ref.watch(pronunciationScorerProvider),
        progressRepository: ref.watch(progressRepositoryProvider),
        audioPlayer: ref.watch(audioPlayerProvider),
      );
    });

const _datasetFingerprintKey = '__dataset_fingerprint_v1';
const _datasetAssetPath = 'assets/datasets/sentences.zh.json';

Future<String?> _loadDatasetFingerprint() async {
  try {
    final jsonString = await rootBundle.loadString(_datasetAssetPath);
    final json = jsonDecode(jsonString) as Map<String, dynamic>;
    final datasetId = json['dataset_id'] as String?;
    final generatedAt = json['generated_at'] as String?;
    if (datasetId == null || generatedAt == null) {
      return null;
    }
    return '$datasetId|$generatedAt';
  } catch (_) {
    return null;
  }
}

Future<void> _clearRecordingsDir() async {
  try {
    final appDir = await getApplicationDocumentsDirectory();
    final recordingsDir = Directory('${appDir.path}/recordings');
    if (await recordingsDir.exists()) {
      await recordingsDir.delete(recursive: true);
    }
  } catch (_) {
    // Non-fatal; best-effort cleanup.
  }
}

Future<void> _resetStateIfDatasetChanged({
  required Box<dynamic> progressBox,
  required Box<dynamic> attemptsBox,
  required Box<dynamic> settingsBox,
}) async {
  final currentFingerprint = await _loadDatasetFingerprint();
  if (currentFingerprint == null) return;

  final stored = settingsBox.get(_datasetFingerprintKey) as String?;

  // If the fingerprint changes (or this key is absent from older app versions),
  // invalidate per-sequence state that is keyed by sequence IDs.
  if (stored != currentFingerprint) {
    await progressBox.clear();
    await attemptsBox.clear();
    await _clearRecordingsDir();
    await settingsBox.put(_datasetFingerprintKey, currentFingerprint);
  }
}

/// Creates provider overrides for app initialization.
///
/// Opens Hive boxes and returns overrides for the hiveBoxProvider.
Future<List<Override>> createOverrides() async {
  // Open all required boxes
  final progressBox = await Hive.openBox<dynamic>(HiveBoxes.progress);
  final attemptsBox = await Hive.openBox<dynamic>(HiveBoxes.attempts);
  final settingsBox = await Hive.openBox<dynamic>(HiveBoxes.settings);
  await _resetStateIfDatasetChanged(
    progressBox: progressBox,
    attemptsBox: attemptsBox,
    settingsBox: settingsBox,
  );

  return [
    hiveBoxProvider(HiveBoxes.progress).overrideWithValue(progressBox),
    hiveBoxProvider(HiveBoxes.attempts).overrideWithValue(attemptsBox),
    hiveBoxProvider(HiveBoxes.settings).overrideWithValue(settingsBox),
  ];
}
