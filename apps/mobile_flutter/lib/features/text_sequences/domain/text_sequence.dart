import 'package:freezed_annotation/freezed_annotation.dart';

part 'text_sequence.freezed.dart';
part 'text_sequence.g.dart';

@freezed
class TextSequence with _$TextSequence {
  const factory TextSequence({
    required String id,
    required String text,
    required String language,
    String? romanization,
    Map<String, String>? gloss,
    List<String>? tokens,
    List<String>? tags,
    int? difficulty,
    List<ExampleVoice>? voices,
  }) = _TextSequence;

  factory TextSequence.fromJson(Map<String, dynamic> json) =>
      _$TextSequenceFromJson(json);
}

@freezed
class ExampleVoice with _$ExampleVoice {
  const factory ExampleVoice({
    required String id,
    Map<String, String>? label,
    required String uri,
    int? durationMs,
  }) = _ExampleVoice;

  factory ExampleVoice.fromJson(Map<String, dynamic> json) =>
      _$ExampleVoiceFromJson(json);
}
