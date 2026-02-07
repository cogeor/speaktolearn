import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../features/practice/presentation/home_screen.dart';
import '../features/text_sequences/presentation/sequence_list_screen.dart';

/// Router provider for the application.
final routerProvider = Provider<GoRouter>((ref) {
  return GoRouter(
    initialLocation: '/',
    routes: [
      GoRoute(
        path: '/',
        name: 'home',
        builder: (context, state) => const HomeScreen(),
      ),
      GoRoute(
        path: '/list',
        name: 'list',
        builder: (context, state) => const SequenceListScreen(),
      ),
    ],
  );
});
