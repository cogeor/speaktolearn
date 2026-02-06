import '../../text_sequences/domain/text_sequence.dart';
import '../../../core/audio/audio_source.dart';

/// Repository for resolving and managing example audio files.
abstract class ExampleAudioRepository {
  /// Resolves a URI to an AudioSource.
  ///
  /// Handles different URI schemes:
  /// - `assets://` - Flutter asset files
  /// - `file://` - Local file system
  /// - `https://` - Remote files (with caching)
  Future<AudioSource> resolve(String uri);

  /// Resolves audio for a specific voice in a sequence.
  Future<AudioSource?> resolveVoice(TextSequence sequence, String voiceId);

  /// Prefetches all example audio for a sequence.
  Future<void> prefetch(TextSequence sequence);

  /// Checks if audio at the URI is available locally (cached or asset).
  Future<bool> isAvailableLocally(String uri);
}
