import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'app/app.dart';
import 'app/di.dart';
import 'core/storage/hive_init.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Initialize Hive
  await initHive();

  // Create provider overrides
  final overrides = await createOverrides();

  runApp(
    ProviderScope(
      overrides: overrides,
      child: const SpeakToLearnApp(),
    ),
  );
}
