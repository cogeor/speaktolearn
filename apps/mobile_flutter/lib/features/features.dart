// Features module exports.
//
// This barrel file exports public APIs from feature modules.
// Only exports domain layer (entities and interfaces) to avoid
// exposing implementation details.

// Example Audio
export 'example_audio/domain/example_audio_repository.dart';

// Practice
export 'practice/presentation/home_state.dart';

// Progress
export 'progress/domain/progress_repository.dart';
export 'progress/domain/score_attempt.dart';
export 'progress/domain/text_sequence_progress.dart';

// Recording
export 'recording/domain/audio_recorder.dart';
export 'recording/domain/recording.dart';
export 'recording/domain/recording_repository.dart';

// Scoring
export 'scoring/domain/grade.dart';
export 'scoring/domain/pronunciation_scorer.dart';

// Selection
export 'selection/domain/get_next_tracked.dart';
export 'selection/domain/ranked_sequence.dart';
export 'selection/domain/sequence_ranker.dart';

// Settings
export 'settings/domain/app_settings.dart';
export 'settings/domain/settings_repository.dart';

// Text Sequences
export 'text_sequences/domain/text_sequence.dart';
export 'text_sequences/domain/text_sequence_repository.dart';
