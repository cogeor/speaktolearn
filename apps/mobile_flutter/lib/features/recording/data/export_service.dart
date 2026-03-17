import 'dart:io';

import 'package:archive/archive_io.dart';
import 'package:path_provider/path_provider.dart';

/// Service that zips all WAV+JSON recording pairs and returns the zip path.
class ExportService {
  ExportService();

  /// Lists all WAV+JSON pairs in the recordings directory, zips them into
  /// `corpus_{timestamp}.zip` in the cache directory, and returns the path.
  Future<ExportResult> exportCorpus() async {
    final appDir = await getApplicationDocumentsDirectory();
    final recordingsDir = Directory('${appDir.path}/recordings');

    if (!await recordingsDir.exists()) {
      return ExportResult(zipPath: '', fileCount: 0, totalBytes: 0);
    }

    final files = <File>[];
    int totalBytes = 0;

    await for (final entity in recordingsDir.list()) {
      if (entity is! File) continue;
      final name = entity.uri.pathSegments.last;
      if (!name.endsWith('.wav') && !name.endsWith('.json')) continue;
      // Skip the index file
      if (name == 'index.json') continue;
      files.add(entity);
      final stat = await entity.stat();
      totalBytes += stat.size;
    }

    if (files.isEmpty) {
      return ExportResult(zipPath: '', fileCount: 0, totalBytes: 0);
    }

    // Build timestamp string safe for filenames.
    final now = DateTime.now().toUtc();
    final ts =
        '${now.year.toString().padLeft(4, '0')}'
        '${now.month.toString().padLeft(2, '0')}'
        '${now.day.toString().padLeft(2, '0')}'
        'T${now.hour.toString().padLeft(2, '0')}'
        '${now.minute.toString().padLeft(2, '0')}'
        '${now.second.toString().padLeft(2, '0')}Z';

    final cacheDir = await getTemporaryDirectory();
    final zipPath = '${cacheDir.path}/corpus_$ts.zip';

    final encoder = ZipFileEncoder();
    encoder.create(zipPath);
    for (final file in files) {
      encoder.addFile(file);
    }
    encoder.close();

    return ExportResult(
      zipPath: zipPath,
      fileCount: files.length,
      totalBytes: totalBytes,
    );
  }

  /// Returns the count of WAV files and total size across all recordings.
  Future<RecordingStats> getStats() async {
    final appDir = await getApplicationDocumentsDirectory();
    final recordingsDir = Directory('${appDir.path}/recordings');

    if (!await recordingsDir.exists()) {
      return RecordingStats(wavCount: 0, totalBytes: 0);
    }

    int wavCount = 0;
    int totalBytes = 0;

    await for (final entity in recordingsDir.list()) {
      if (entity is! File) continue;
      final name = entity.uri.pathSegments.last;
      if (!name.endsWith('.wav')) continue;
      wavCount++;
      final stat = await entity.stat();
      totalBytes += stat.size;
    }

    return RecordingStats(wavCount: wavCount, totalBytes: totalBytes);
  }
}

/// Result of a corpus export operation.
class ExportResult {
  const ExportResult({
    required this.zipPath,
    required this.fileCount,
    required this.totalBytes,
  });

  /// Absolute path of the generated zip file. Empty if no files were found.
  final String zipPath;

  /// Number of files included in the zip.
  final int fileCount;

  /// Combined size of all source files in bytes.
  final int totalBytes;

  bool get hasFiles => zipPath.isNotEmpty && fileCount > 0;
}

/// Summary stats about the recordings on-device.
class RecordingStats {
  const RecordingStats({required this.wavCount, required this.totalBytes});

  final int wavCount;
  final int totalBytes;

  String get formattedSize {
    if (totalBytes < 1024) return '${totalBytes}B';
    if (totalBytes < 1024 * 1024) {
      return '${(totalBytes / 1024).toStringAsFixed(1)}KB';
    }
    return '${(totalBytes / (1024 * 1024)).toStringAsFixed(1)}MB';
  }
}
