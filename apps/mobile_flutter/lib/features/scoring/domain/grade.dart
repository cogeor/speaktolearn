import 'package:freezed_annotation/freezed_annotation.dart';

part 'grade.freezed.dart';
part 'grade.g.dart';

@freezed
class Grade with _$Grade {
  const factory Grade({
    required int overall,
    required String method,
    String? recognizedText,
    Map<String, dynamic>? details,
  }) = _Grade;

  factory Grade.fromJson(Map<String, dynamic> json) =>
      _$GradeFromJson(json);
}
