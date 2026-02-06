import 'dart:convert';
import 'package:flutter/services.dart' show rootBundle;
import 'dataset_dto.dart';

/// Abstract class for loading datasets from various sources.
abstract class DatasetSource {
  /// Loads and returns the dataset.
  Future<DatasetDto> load();

  /// Checks if the data source is available.
  Future<bool> isAvailable();
}

/// Loads datasets from Flutter assets.
class AssetDatasetSource extends DatasetSource {
  AssetDatasetSource({this.assetPath = 'assets/datasets/sentences.zh.json'});

  final String assetPath;

  @override
  Future<DatasetDto> load() async {
    final jsonString = await rootBundle.loadString(assetPath);
    final json = jsonDecode(jsonString) as Map<String, dynamic>;
    return DatasetDto.fromJson(json);
  }

  @override
  Future<bool> isAvailable() async {
    try {
      await rootBundle.loadString(assetPath);
      return true;
    } catch (_) {
      return false;
    }
  }
}
