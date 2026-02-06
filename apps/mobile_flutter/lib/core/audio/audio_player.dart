import 'audio_source.dart';

/// Represents the current state of audio playback.
///
/// Used by [AudioPlayer] implementations to communicate their status
/// through [AudioPlayer.stateStream] and [AudioPlayer.currentState].
enum PlaybackState {
  /// Player is initialized but no audio is loaded.
  idle,

  /// Audio is being loaded or buffered.
  loading,

  /// Audio is actively playing.
  playing,

  /// Playback is paused; can be resumed.
  paused,

  /// Playback finished successfully.
  completed,

  /// An error occurred during loading or playback.
  error,
}

/// Abstract interface for audio playback functionality.
///
/// Implementations should handle loading audio from various sources,
/// controlling playback, and emitting state/position updates via streams.
///
/// Example usage:
/// ```dart
/// final player = SomeAudioPlayerImpl();
/// await player.load(AssetAudioSource('assets/audio/test.opus'));
/// await player.play();
///
/// player.stateStream.listen((state) {
///   print('State: $state');
/// });
///
/// player.positionStream.listen((position) {
///   print('Position: ${position.inSeconds}s');
/// });
/// ```
abstract interface class AudioPlayer {
  /// Stream of playback state changes.
  ///
  /// Emits a new [PlaybackState] whenever the player state changes.
  Stream<PlaybackState> get stateStream;

  /// Stream of playback position updates.
  ///
  /// Emits the current playback position periodically during playback.
  Stream<Duration> get positionStream;

  /// The current playback state.
  ///
  /// For reactive updates, prefer listening to [stateStream].
  PlaybackState get currentState;

  /// The total duration of the loaded audio, if available.
  ///
  /// Returns `null` if no audio is loaded or duration is unknown.
  Duration? get duration;

  /// Loads audio from the specified [source].
  ///
  /// Transitions state to [PlaybackState.loading] then to
  /// [PlaybackState.paused] when ready, or [PlaybackState.error] on failure.
  Future<void> load(AudioSource source);

  /// Starts or resumes playback.
  ///
  /// Transitions state to [PlaybackState.playing].
  /// Has no effect if no audio is loaded.
  Future<void> play();

  /// Pauses playback.
  ///
  /// Transitions state to [PlaybackState.paused].
  /// Playback can be resumed with [play].
  Future<void> pause();

  /// Stops playback and resets position to the beginning.
  ///
  /// Transitions state to [PlaybackState.idle].
  Future<void> stop();

  /// Seeks to the specified [position] in the audio.
  ///
  /// The [position] is clamped to the valid range (0 to [duration]).
  Future<void> seek(Duration position);

  /// Releases resources used by the player.
  ///
  /// The player should not be used after calling this method.
  Future<void> dispose();
}
