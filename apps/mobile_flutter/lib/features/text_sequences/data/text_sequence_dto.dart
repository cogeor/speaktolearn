import 'package:json_annotation/json_annotation.dart';

import '../domain/text_sequence.dart';
import 'example_audio_dto.dart';

part 'text_sequence_dto.g.dart';

@JsonSerializable()
class TextSequenceDto {
  final String id;
  final String text;
  final String? romanization;
  final Map<String, String>? gloss;
  final List<String>? tokens;
  final List<String>? tags;
  final int? difficulty;
  @JsonKey(name: 'example_audio')
  final ExampleAudioDto? exampleAudio;

  TextSequenceDto({
    required this.id,
    required this.text,
    this.romanization,
    this.gloss,
    this.tokens,
    this.tags,
    this.difficulty,
    this.exampleAudio,
  });

  factory TextSequenceDto.fromJson(Map<String, dynamic> json) =>
      _$TextSequenceDtoFromJson(json);

  Map<String, dynamic> toJson() => _$TextSequenceDtoToJson(this);

  /// Parses HSK level from tags (e.g., "hsk1", "hsk2", etc.).
  int? _parseHskFromTags() {
    if (tags == null) return null;
    for (final tag in tags!) {
      final match = RegExp(r'^hsk(\d)$', caseSensitive: false).firstMatch(tag);
      if (match != null) {
        return int.parse(match.group(1)!);
      }
    }
    return null;
  }

  TextSequence toDomain(String language) => TextSequence(
        id: id,
        text: text,
        language: language,
        romanization: romanization,
        gloss: gloss ?? {},
        tokens: tokens ?? [],
        tags: tags ?? [],
        difficulty: difficulty ?? 1,
        hskLevel: _parseHskFromTags(),
        voices: exampleAudio?.voices
                .map((v) => ExampleVoice(
                      id: v.id,
                      label: v.label,
                      uri: v.uri,
                      durationMs: v.durationMs,
                    ))
                .toList() ??
            [],
      );
}
