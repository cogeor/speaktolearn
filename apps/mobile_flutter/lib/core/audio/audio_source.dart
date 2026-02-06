/// Represents different sources for audio playback.
///
/// This sealed class enables exhaustive pattern matching when handling
/// different audio origins (assets, files, or URLs).
sealed class AudioSource {
  const AudioSource();
}

/// Audio loaded from app assets.
///
/// Used for bundled audio files in the assets directory.
/// Example: `AssetAudioSource('assets/audio/test.opus')`
final class AssetAudioSource extends AudioSource {
  /// The asset path relative to the app bundle.
  final String assetPath;

  const AssetAudioSource(this.assetPath);

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is AssetAudioSource &&
          runtimeType == other.runtimeType &&
          assetPath == other.assetPath;

  @override
  int get hashCode => assetPath.hashCode;

  @override
  String toString() => 'AssetAudioSource($assetPath)';
}

/// Audio loaded from a local file.
///
/// Used for recordings or downloaded audio stored on the device.
/// Example: `FileAudioSource('/data/recordings/test.m4a')`
final class FileAudioSource extends AudioSource {
  /// The absolute file path on the device.
  final String filePath;

  const FileAudioSource(this.filePath);

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is FileAudioSource &&
          runtimeType == other.runtimeType &&
          filePath == other.filePath;

  @override
  int get hashCode => filePath.hashCode;

  @override
  String toString() => 'FileAudioSource($filePath)';
}

/// Audio streamed from a URL.
///
/// Used for remote audio content that should be streamed or cached.
/// Example: `UrlAudioSource('https://example.com/audio.opus')`
final class UrlAudioSource extends AudioSource {
  /// The URL to stream audio from.
  final String url;

  const UrlAudioSource(this.url);

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is UrlAudioSource &&
          runtimeType == other.runtimeType &&
          url == other.url;

  @override
  int get hashCode => url.hashCode;

  @override
  String toString() => 'UrlAudioSource($url)';
}
