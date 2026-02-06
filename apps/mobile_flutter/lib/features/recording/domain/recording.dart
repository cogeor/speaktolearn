import 'package:freezed_annotation/freezed_annotation.dart';

part 'recording.freezed.dart';
part 'recording.g.dart';

@freezed
class Recording with _$Recording {
  const factory Recording({
    required String id,
    required String textSequenceId,
    required DateTime createdAt,
    required String filePath,
    int? durationMs,
    int? sampleRate,
    String? mimeType,
  }) = _Recording;

  factory Recording.fromJson(Map<String, dynamic> json) =>
      _$RecordingFromJson(json);
}
