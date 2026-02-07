import 'package:freezed_annotation/freezed_annotation.dart';

part 'sequence_list_item.freezed.dart';

@freezed
class SequenceListItem with _$SequenceListItem {
  const factory SequenceListItem({
    required String id,
    required String text,
    required bool isTracked,
    int? bestScore,
    int? hskLevel,
  }) = _SequenceListItem;
}
