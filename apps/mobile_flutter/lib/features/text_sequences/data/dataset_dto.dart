import 'package:json_annotation/json_annotation.dart';

part 'dataset_dto.g.dart';

@JsonSerializable()
class DatasetDto {
  @JsonKey(name: 'schema_version')
  final String schemaVersion;

  @JsonKey(name: 'dataset_id')
  final String datasetId;

  final String language;

  @JsonKey(name: 'generated_at')
  final String generatedAt;

  final List<Map<String, dynamic>> items;

  DatasetDto({
    required this.schemaVersion,
    required this.datasetId,
    required this.language,
    required this.generatedAt,
    required this.items,
  });

  factory DatasetDto.fromJson(Map<String, dynamic> json) =>
      _$DatasetDtoFromJson(json);

  Map<String, dynamic> toJson() => _$DatasetDtoToJson(this);
}
