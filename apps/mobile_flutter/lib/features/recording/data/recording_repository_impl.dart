import 'dart:convert';
import 'dart:io';

import 'package:path_provider/path_provider.dart';

import '../domain/recording.dart';
import '../domain/recording_metadata.dart';
import '../domain/recording_repository.dart';

/// File-system based implementation of [RecordingRepository].
///
/// Each save produces a timestamped file pair:
///   `{textSequenceId}_{yyyyMMddTHHmmssZ}.wav`
///   `{textSequenceId}_{yyyyMMddTHHmmssZ}.json`
///
/// An index file `recordings/index.json` maps each textSequenceId to the
/// stem (filename without extension) of the most recently saved recording.
/// [getLatest] reads the index; [listHistory] scans the directory.
class RecordingRepositoryImpl implements RecordingRepository {
  RecordingRepositoryImpl();

  /// In-memory cache of recordings by text sequence ID.
  final Map<String, Recording> _cache = {};

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  /// Gets the recordings directory, creating it if absent.
  Future<String> _getRecordingsDir() async {
    final appDir = await getApplicationDocumentsDirectory();
    final recordingsDir = Directory('${appDir.path}/recordings');
    if (!await recordingsDir.exists()) {
      await recordingsDir.create(recursive: true);
    }
    return recordingsDir.path;
  }

  /// Returns the path to the shared index file.
  Future<String> _getIndexPath() async {
    final dir = await _getRecordingsDir();
    return '$dir/index.json';
  }

  /// Reads the index file. Returns an empty map if missing or unreadable.
  Future<Map<String, String>> _readIndex() async {
    try {
      final file = File(await _getIndexPath());
      if (!await file.exists()) return {};
      final raw = await file.readAsString();
      final decoded = jsonDecode(raw) as Map<String, dynamic>;
      return decoded.map((k, v) => MapEntry(k, v as String));
    } catch (_) {
      return {};
    }
  }

  /// Overwrites the index file with [index].
  Future<void> _writeIndex(Map<String, String> index) async {
    final file = File(await _getIndexPath());
    await file.writeAsString(const JsonEncoder.withIndent('  ').convert(index));
  }

  /// Formats [dt] as `yyyyMMddTHHmmssZ` (no colons, always UTC).
  String _formatTimestamp(DateTime dt) {
    final u = dt.toUtc();
    final y = u.year.toString().padLeft(4, '0');
    final mo = u.month.toString().padLeft(2, '0');
    final d = u.day.toString().padLeft(2, '0');
    final h = u.hour.toString().padLeft(2, '0');
    final mi = u.minute.toString().padLeft(2, '0');
    final s = u.second.toString().padLeft(2, '0');
    return '$y$mo${d}T$h$mi${s}Z';
  }

  /// Returns the stem (no extension) for a new timestamped file.
  String _stem(String textSequenceId, DateTime createdAt) =>
      '${textSequenceId}_${_formatTimestamp(createdAt)}';

  // ---------------------------------------------------------------------------
  // RecordingRepository implementation
  // ---------------------------------------------------------------------------

  @override
  Future<void> saveLatest(Recording recording) async {
    final dir = await _getRecordingsDir();
    final stem = _stem(recording.textSequenceId, recording.createdAt);
    final destPath = '$dir/$stem.wav';

    // Copy the temporary file to its permanent location.
    final sourceFile = File(recording.filePath);
    await sourceFile.copy(destPath);

    // Update the index so getLatest can find this recording.
    final index = await _readIndex();
    index[recording.textSequenceId] = stem;
    await _writeIndex(index);

    // Update cache with the persisted path.
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
  Future<void> saveMetadata(RecordingMetadata metadata) async {
    // The stem for the metadata sidecar must match the latest WAV file.
    final index = await _readIndex();
    final stem = index[metadata.textSequenceId];
    if (stem == null) {
      // No recording saved yet – skip writing the sidecar.
      return;
    }
    final dir = await _getRecordingsDir();
    final metadataPath = '$dir/$stem.json';
    final file = File(metadataPath);
    await file.writeAsString(
      const JsonEncoder.withIndent('  ').convert(metadata.toJson()),
    );
  }

  @override
  Future<Recording?> getLatest(String textSequenceId) async {
    // Check cache first.
    if (_cache.containsKey(textSequenceId)) {
      return _cache[textSequenceId];
    }

    // Consult the index for the most-recently-saved stem.
    final index = await _readIndex();
    final stem = index[textSequenceId];
    if (stem == null) return null;

    final dir = await _getRecordingsDir();
    final filePath = '$dir/$stem.wav';
    final file = File(filePath);
    if (!await file.exists()) return null;

    final stat = await file.stat();
    final recording = Recording(
      id: stem,
      textSequenceId: textSequenceId,
      createdAt: stat.modified,
      filePath: filePath,
    );
    _cache[textSequenceId] = recording;
    return recording;
  }

  @override
  Future<void> deleteLatest(String textSequenceId) async {
    _cache.remove(textSequenceId);

    final index = await _readIndex();
    final stem = index.remove(textSequenceId);
    await _writeIndex(index);

    if (stem != null) {
      final dir = await _getRecordingsDir();

      final wavFile = File('$dir/$stem.wav');
      if (await wavFile.exists()) await wavFile.delete();

      final jsonFile = File('$dir/$stem.json');
      if (await jsonFile.exists()) await jsonFile.delete();
    }
  }

  @override
  Future<bool> hasRecording(String textSequenceId) async {
    if (_cache.containsKey(textSequenceId)) return true;
    final index = await _readIndex();
    final stem = index[textSequenceId];
    if (stem == null) return false;
    final dir = await _getRecordingsDir();
    return File('$dir/$stem.wav').exists();
  }

  @override
  Future<List<Recording>> listHistory(String textSequenceId) async {
    final dir = await _getRecordingsDir();
    final directory = Directory(dir);
    if (!await directory.exists()) return [];

    final prefix = '${textSequenceId}_';
    final List<Recording> results = [];

    await for (final entity in directory.list()) {
      if (entity is! File) continue;
      final name = entity.uri.pathSegments.last;
      if (!name.startsWith(prefix) || !name.endsWith('.wav')) continue;

      final stem = name.substring(0, name.length - 4); // strip ".wav"
      final stat = await entity.stat();
      results.add(
        Recording(
          id: stem,
          textSequenceId: textSequenceId,
          createdAt: stat.modified,
          filePath: entity.path,
        ),
      );
    }

    // Sort oldest first.
    results.sort((a, b) => a.createdAt.compareTo(b.createdAt));
    return results;
  }
}
