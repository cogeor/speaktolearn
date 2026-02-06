import 'package:hive/hive.dart';

/// Names of Hive boxes used in the app.
abstract class HiveBoxes {
  static const String progress = 'progress';
  static const String attempts = 'attempts';
  static const String settings = 'settings';
}

/// Container for opened Hive boxes.
class HiveBoxManager {
  final Box<dynamic> progressBox;
  final Box<dynamic> attemptsBox;
  final Box<dynamic> settingsBox;

  const HiveBoxManager({
    required this.progressBox,
    required this.attemptsBox,
    required this.settingsBox,
  });
}
