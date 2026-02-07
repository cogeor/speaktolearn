import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:speak_to_learn/app/router.dart';
import 'package:speak_to_learn/app/theme.dart';

import '../mocks/mock_providers.dart';

/// A test wrapper for the SpeakToLearn app.
///
/// Wraps the given widget in MaterialApp with router and theme,
/// along with test provider overrides.
class TestApp extends StatelessWidget {
  const TestApp({
    super.key,
    required this.child,
    this.overrides = const [],
  });

  final Widget child;
  final List<Override> overrides;

  @override
  Widget build(BuildContext context) {
    return ProviderScope(
      overrides: [
        ...createTestOverrides(),
        ...overrides,
      ],
      child: MaterialApp(
        theme: AppTheme.darkTheme,
        home: child,
        debugShowCheckedModeBanner: false,
      ),
    );
  }
}

/// A test wrapper with router for navigation testing.
class TestAppWithRouter extends ConsumerWidget {
  const TestAppWithRouter({
    super.key,
    this.overrides = const [],
    this.initialLocation = '/',
  });

  final List<Override> overrides;
  final String initialLocation;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final router = ref.watch(routerProvider);

    return MaterialApp.router(
      theme: AppTheme.darkTheme,
      routerConfig: router,
      debugShowCheckedModeBanner: false,
    );
  }
}
