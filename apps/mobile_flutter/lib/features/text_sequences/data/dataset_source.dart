import 'dataset_dto.dart';

/// Abstract class for loading datasets from various sources.
abstract class DatasetSource {
  /// Loads and returns the dataset.
  Future<DatasetDto> load();

  /// Checks if the data source is available.
  Future<bool> isAvailable();
}
