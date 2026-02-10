import 'package:freezed_annotation/freezed_annotation.dart';

import '../../progress/domain/sentence_rating.dart';

part 'sequence_list_item.freezed.dart';

@freezed
class SequenceListItem with _$SequenceListItem {
  const factory SequenceListItem({
    required String id,
    required String text,
    SentenceRating? lastRating,
    int? hskLevel,
  }) = _SequenceListItem;
}
