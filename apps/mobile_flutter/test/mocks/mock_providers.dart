import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:speak_to_learn/app/di.dart';
import 'package:speak_to_learn/features/text_sequences/domain/text_sequence.dart';

import 'mock_audio.dart';
import 'mock_repositories.dart';

/// Sample test sequences for use in tests.
List<TextSequence> get testSequences => [
  const TextSequence(
    id: 'test-001',
    text: '你好',
    language: 'zh',
    tags: ['greeting'],
    difficulty: 1,
    voices: [
      ExampleVoice(id: 'male', uri: 'assets://examples/male/test-001.m4a'),
      ExampleVoice(id: 'female', uri: 'assets://examples/female/test-001.m4a'),
    ],
  ),
  const TextSequence(
    id: 'test-002',
    text: '谢谢',
    language: 'zh',
    tags: ['courtesy', 'basic'],
    difficulty: 1,
    voices: [
      ExampleVoice(id: 'male', uri: 'assets://examples/male/test-002.m4a'),
      ExampleVoice(id: 'female', uri: 'assets://examples/female/test-002.m4a'),
    ],
  ),
  const TextSequence(
    id: 'test-003',
    text: '再见',
    language: 'zh',
    tags: ['farewell', 'basic'],
    difficulty: 1,
    voices: [
      ExampleVoice(id: 'male', uri: 'assets://examples/male/test-003.m4a'),
      ExampleVoice(id: 'female', uri: 'assets://examples/female/test-003.m4a'),
    ],
  ),
];

/// Creates provider overrides with mock repositories for testing.
///
/// Uses in-memory mock implementations instead of real Hive storage.
List<Override> createTestOverrides({List<TextSequence>? sequences}) {
  final mockTextSequenceRepo = MockTextSequenceRepository(
    sequences ?? testSequences,
  );
  final mockProgressRepo = MockProgressRepository();
  final mockRecordingRepo = MockRecordingRepository();
  final mockExampleAudioRepo = MockExampleAudioRepository();
  final mockSettingsRepo = MockSettingsRepository();

  return [
    textSequenceRepositoryProvider.overrideWithValue(mockTextSequenceRepo),
    progressRepositoryProvider.overrideWithValue(mockProgressRepo),
    recordingRepositoryProvider.overrideWithValue(mockRecordingRepo),
    exampleAudioRepositoryProvider.overrideWithValue(mockExampleAudioRepo),
    settingsRepositoryProvider.overrideWithValue(mockSettingsRepo),
  ];
}

/// Creates provider overrides with mocked audio services for testing.
///
/// Includes fake audio recorder and player for integration tests.
List<Override> createTestOverridesWithMockedAudio({
  List<TextSequence>? sequences,
  FakeAudioRecorder? recorder,
  FakeAudioPlayer? player,
}) {
  final baseOverrides = createTestOverrides(sequences: sequences);
  final fakeRecorder = recorder ?? FakeAudioRecorder();
  final fakePlayer = player ?? FakeAudioPlayer();

  return [
    ...baseOverrides,
    audioRecorderProvider.overrideWithValue(fakeRecorder),
    audioPlayerProvider.overrideWithValue(fakePlayer),
  ];
}
