import 'package:hive_flutter/hive_flutter.dart';
import 'hive_boxes.dart';

/// Initializes Hive for Flutter and opens all required boxes.
///
/// Returns a [HiveBoxManager] containing references to all opened boxes.
Future<HiveBoxManager> initHive() async {
  await Hive.initFlutter();

  final progressBox = await Hive.openBox<dynamic>(HiveBoxes.progress);
  final attemptsBox = await Hive.openBox<dynamic>(HiveBoxes.attempts);
  final settingsBox = await Hive.openBox<dynamic>(HiveBoxes.settings);

  return HiveBoxManager(
    progressBox: progressBox,
    attemptsBox: attemptsBox,
    settingsBox: settingsBox,
  );
}
