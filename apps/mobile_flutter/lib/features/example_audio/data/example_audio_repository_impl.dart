import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:dio/dio.dart';
import 'package:path_provider/path_provider.dart';

import '../../../core/audio/audio_source.dart';
import '../../text_sequences/domain/text_sequence.dart';
import '../domain/example_audio_repository.dart';

/// Implementation of [ExampleAudioRepository] that resolves URIs to audio sources.
///
/// Supports the following URI schemes:
/// - `assets://` - Flutter asset files
/// - `file://` - Local file system paths
/// - `http://` / `https://` - Remote URLs
class ExampleAudioRepositoryImpl implements ExampleAudioRepository {
  /// Creates an [ExampleAudioRepositoryImpl].
  ///
  /// Optionally takes a [Dio] instance for HTTP requests. If not provided,
  /// a default instance will be created.
  ExampleAudioRepositoryImpl({Dio? dio}) : _dio = dio ?? Dio();

  final Dio _dio;
  static const int _maxCacheFiles = 200;

  @override
  Future<AudioSource> resolve(String uri) async {
    if (uri.startsWith('assets://')) {
      final path = uri.substring('assets://'.length);
      return AssetAudioSource(path);
    }

    if (uri.startsWith('file://')) {
      final path = uri.substring('file://'.length);
      return FileAudioSource(path);
    }

    if (uri.startsWith('http://') || uri.startsWith('https://')) {
      return _resolveRemote(uri);
    }

    throw ArgumentError('Unknown URI scheme: $uri');
  }

  @override
  Future<AudioSource?> resolveVoice(
    TextSequence sequence,
    String voiceId,
  ) async {
    final voices = sequence.voices;
    if (voices == null || voices.isEmpty) return null;

    try {
      final voice = voices.firstWhere((v) => v.id == voiceId);
      return resolve(voice.uri);
    } catch (_) {
      // Voice not found
      return null;
    }
  }

  @override
  Future<void> prefetch(TextSequence sequence) async {
    final voices = sequence.voices;
    if (voices == null || voices.isEmpty) return;

    for (final voice in voices) {
      // Only prefetch remote URIs
      if (voice.uri.startsWith('http://') || voice.uri.startsWith('https://')) {
        final cachedPath = await _getCachedPath(voice.uri);
        final file = File(cachedPath);
        if (!await file.exists()) {
          try {
            await _downloadToCache(voice.uri, cachedPath);
          } catch (_) {
            // Ignore prefetch errors
          }
        }
      }
    }
  }

  @override
  Future<bool> isAvailableLocally(String uri) async {
    if (uri.startsWith('assets://')) {
      // Assets are always available locally
      return true;
    }

    if (uri.startsWith('file://')) {
      final path = uri.substring('file://'.length);
      return File(path).exists();
    }

    if (uri.startsWith('http://') || uri.startsWith('https://')) {
      final cachedPath = await _getCachedPath(uri);
      return File(cachedPath).exists();
    }

    return false;
  }

  Future<AudioSource> _resolveRemote(String uri) async {
    final cachedPath = await _getCachedPath(uri);
    final file = File(cachedPath);

    if (await file.exists()) {
      return FileAudioSource(cachedPath);
    }

    await _downloadToCache(uri, cachedPath);
    return FileAudioSource(cachedPath);
  }

  Future<String> _getCachedPath(String uri) async {
    final cacheDir = await getApplicationCacheDirectory();
    final audioDir = Directory('${cacheDir.path}/audio_cache');
    if (!await audioDir.exists()) {
      await audioDir.create(recursive: true);
    }
    final hash = md5.convert(utf8.encode(uri)).toString();
    return '${audioDir.path}/$hash.m4a';
  }

  Future<void> _downloadToCache(String uri, String path) async {
    await _dio.download(uri, path);
    await _cleanupCache();
  }

  Future<void> _cleanupCache() async {
    final cacheDir = await getApplicationCacheDirectory();
    final audioDir = Directory('${cacheDir.path}/audio_cache');
    if (!await audioDir.exists()) return;

    final files = await audioDir.list().toList();
    if (files.length <= _maxCacheFiles) return;

    // Sort by last accessed (oldest first) and delete excess
    final fileStats = <FileSystemEntity, DateTime>{};
    for (final file in files) {
      if (file is File) {
        final stat = await file.stat();
        fileStats[file] = stat.accessed;
      }
    }

    final sortedFiles = fileStats.entries.toList()
      ..sort((a, b) => a.value.compareTo(b.value));

    final toDelete = sortedFiles.length - _maxCacheFiles;
    for (var i = 0; i < toDelete; i++) {
      await sortedFiles[i].key.delete();
    }
  }
}
