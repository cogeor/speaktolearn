import 'dart:io';

import 'package:path_provider/path_provider.dart';

import '../domain/recording.dart';
import '../domain/recording_repository.dart';

/// File-system based implementation of [RecordingRepository].
class RecordingRepositoryImpl implements RecordingRepository {
  RecordingRepositoryImpl();

  /// In-memory cache of recordings by text sequence ID.
  final Map<String, Recording> _cache = {};

  /// Gets the recordings directory path.
  Future<String> _getRecordingsDir() async {
    final appDir = await getApplicationDocumentsDirectory();
    final recordingsDir = Directory('${appDir.path}/recordings');
    if (!await recordingsDir.exists()) {
      await recordingsDir.create(recursive: true);
    }
    return recordingsDir.path;
  }

  /// Gets the file path for a text sequence's recording.
  /// Using .wav extension for ML-compatible 16kHz PCM audio.
  Future<String> _getFilePath(String textSequenceId) async {
    final dir = await _getRecordingsDir();
    return '$dir/$textSequenceId.wav';
  }

  @override
  Future<void> saveLatest(Recording recording) async {
    // Delete existing recording if any
    await deleteLatest(recording.textSequenceId);

    // Copy file to recordings directory
    final destPath = await _getFilePath(recording.textSequenceId);
    final sourceFile = File(recording.filePath);
    await sourceFile.copy(destPath);

    // Update cache with new path
    final savedRecording = Recording(
      id: recording.id,
      textSequenceId: recording.textSequenceId,
      createdAt: recording.createdAt,
      filePath: destPath,
      durationMs: recording.durationMs,
      sampleRate: recording.sampleRate,
      mimeType: recording.mimeType,
    );
    _cache[recording.textSequenceId] = savedRecording;
  }

  @override
  Future<Recording?> getLatest(String textSequenceId) async {
    // Check cache first
    if (_cache.containsKey(textSequenceId)) {
      return _cache[textSequenceId];
    }

    // Check file system
    final filePath = await _getFilePath(textSequenceId);
    final file = File(filePath);
    if (await file.exists()) {
      final stat = await file.stat();
      final recording = Recording(
        id: textSequenceId,
        textSequenceId: textSequenceId,
        createdAt: stat.modified,
        filePath: filePath,
      );
      _cache[textSequenceId] = recording;
      return recording;
    }

    return null;
  }

  @override
  Future<void> deleteLatest(String textSequenceId) async {
    _cache.remove(textSequenceId);
    final filePath = await _getFilePath(textSequenceId);
    final file = File(filePath);
    if (await file.exists()) {
      await file.delete();
    }
  }

  @override
  Future<bool> hasRecording(String textSequenceId) async {
    if (_cache.containsKey(textSequenceId)) {
      return true;
    }
    final filePath = await _getFilePath(textSequenceId);
    return File(filePath).exists();
  }
}
