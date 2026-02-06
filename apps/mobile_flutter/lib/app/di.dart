import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:hive/hive.dart';

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
import '../features/scoring/domain/pronunciation_scorer.dart';
import '../features/selection/domain/get_next_tracked.dart';
import '../features/selection/domain/sequence_ranker.dart';
import '../features/settings/data/settings_repository_impl.dart';
import '../features/settings/domain/settings_repository.dart';
import '../features/text_sequences/data/dataset_source.dart';
import '../features/text_sequences/data/text_sequence_repository_impl.dart';
import '../features/text_sequences/domain/text_sequence_repository.dart';

/// Provider for Hive boxes. Override in createOverrides().
final hiveBoxProvider = Provider.family<Box<dynamic>, String>((ref, boxName) {
  throw UnimplementedError('hiveBoxProvider must be overridden with actual boxes');
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
  // Using mock for now until real implementation is added
  return MockSpeechRecognizer(defaultResponse: '');
});

/// Provider for pronunciation scorer.
final pronunciationScorerProvider = Provider<PronunciationScorer>((ref) {
  final recognizer = ref.watch(speechRecognizerProvider);
  final calculator = CerCalculator();
  return AsrSimilarityScorer(
    recognizer: recognizer,
    calculator: calculator,
  );
});

/// Provider for sequence ranker.
final sequenceRankerProvider = Provider<SequenceRanker>((ref) {
  return DefaultSequenceRanker();
});

/// Provider for GetNextTrackedSequence use case.
final getNextTrackedSequenceProvider = Provider<GetNextTrackedSequence>((ref) {
  return GetNextTrackedSequence(
    textSequenceRepository: ref.watch(textSequenceRepositoryProvider),
    progressRepository: ref.watch(progressRepositoryProvider),
    ranker: ref.watch(sequenceRankerProvider),
  );
});

/// Creates provider overrides for app initialization.
///
/// Opens Hive boxes and returns overrides for the hiveBoxProvider.
Future<List<Override>> createOverrides() async {
  // Open all required boxes
  final progressBox = await Hive.openBox<dynamic>(HiveBoxes.progress);
  final attemptsBox = await Hive.openBox<dynamic>(HiveBoxes.attempts);
  final settingsBox = await Hive.openBox<dynamic>(HiveBoxes.settings);

  return [
    hiveBoxProvider(HiveBoxes.progress).overrideWithValue(progressBox),
    hiveBoxProvider(HiveBoxes.attempts).overrideWithValue(attemptsBox),
    hiveBoxProvider(HiveBoxes.settings).overrideWithValue(settingsBox),
  ];
}
