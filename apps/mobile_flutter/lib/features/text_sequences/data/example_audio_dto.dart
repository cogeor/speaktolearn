import 'package:json_annotation/json_annotation.dart';

part 'example_audio_dto.g.dart';

@JsonSerializable()
class VoiceDto {
  final String id;
  final Map<String, String> label;
  final String uri;
  @JsonKey(name: 'duration_ms')
  final int durationMs;

  VoiceDto({
    required this.id,
    required this.label,
    required this.uri,
    required this.durationMs,
  });

  factory VoiceDto.fromJson(Map<String, dynamic> json) =>
      _$VoiceDtoFromJson(json);

  Map<String, dynamic> toJson() => _$VoiceDtoToJson(this);
}

@JsonSerializable()
class ExampleAudioDto {
  final List<VoiceDto> voices;

  ExampleAudioDto({
    required this.voices,
  });

  factory ExampleAudioDto.fromJson(Map<String, dynamic> json) =>
      _$ExampleAudioDtoFromJson(json);

  Map<String, dynamic> toJson() => _$ExampleAudioDtoToJson(this);
}
