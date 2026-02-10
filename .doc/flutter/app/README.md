# app/ Module

## Purpose

Application-level configuration: entry point, dependency injection, routing, and theming. This module wires together all features but contains no business logic.

## Folder Structure

```
app/
├── app.dart          # MaterialApp widget
├── router.dart       # GoRouter configuration
├── di.dart           # Riverpod provider wiring
└── theme.dart        # Dark theme definition
```

---

## Files

### `app.dart`

**Purpose**: Root widget that configures MaterialApp with theme and routing.

**Responsibilities**:
- Create `MaterialApp.router` with GoRouter
- Apply dark theme
- Set up error handling widgets

**Implementation**:

```dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'router.dart';
import 'theme.dart';

class SpeakToLearnApp extends ConsumerWidget {
  const SpeakToLearnApp({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final router = ref.watch(routerProvider);

    return MaterialApp.router(
      title: 'SpeakToLearn',
      theme: darkTheme,
      routerConfig: router,
      debugShowCheckedModeBanner: false,
    );
  }
}
```

**Dependencies**: `router.dart`, `theme.dart`

---

### `router.dart`

**Purpose**: Declarative navigation with GoRouter.

**Responsibilities**:
- Define all app routes
- Handle navigation transitions
- Expose router as Riverpod provider

**Routes**:

| Path | Screen | Description |
|------|--------|-------------|
| `/` | `HomeScreen` | Trainer with single sentence |
| `/list` | `SequenceListScreen` | Browse all sequences |

**Implementation**:

```dart
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import '../features/practice/presentation/home_screen.dart';
import '../features/text_sequences/presentation/sequence_list_screen.dart';

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
```

**Design Decision**: Only 2 screens to keep UI minimal. Practice happens via bottom sheet on Home, not a separate screen.

---

### `di.dart`

**Purpose**: Dependency injection wiring using Riverpod provider overrides.

**Responsibilities**:
- Create concrete repository implementations
- Wire dependencies between features
- Provide `ProviderScope` overrides for `main.dart`

**Provider Hierarchy**:

```
Core Providers
├── hiveProvider (Hive boxes)
├── audioPlayerProvider (just_audio)
└── pathProvider (path_provider)

Repository Providers
├── datasetSourceProvider → AssetDatasetSource
├── textSequenceRepositoryProvider → TextSequenceRepositoryImpl
├── progressRepositoryProvider → ProgressRepositoryImpl
├── recordingRepositoryProvider → RecordingRepositoryImpl
├── exampleAudioRepositoryProvider → ExampleAudioRepositoryImpl
└── settingsRepositoryProvider → SettingsRepositoryImpl

Service Providers
├── audioRecorderProvider → RecordPluginRecorder
├── speechRecognizerProvider → SpeechToTextRecognizer
├── pronunciationScorerProvider → AsrSimilarityScorer
└── sequenceRankerProvider → DefaultSequenceRanker

Controller Providers
├── homeControllerProvider
└── sequenceListControllerProvider
```

**Implementation**:

```dart
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:hive_flutter/hive_flutter.dart';
import 'package:just_audio/just_audio.dart';
import 'package:path_provider/path_provider.dart';

// Core
final hiveBoxProvider = Provider<Box<dynamic>>((ref) {
  throw UnimplementedError('Initialize in main');
});

// Repositories
final textSequenceRepositoryProvider = Provider<TextSequenceRepository>((ref) {
  final source = ref.watch(datasetSourceProvider);
  return TextSequenceRepositoryImpl(source);
});

final progressRepositoryProvider = Provider<ProgressRepository>((ref) {
  final box = ref.watch(hiveBoxProvider);
  return ProgressRepositoryImpl(box);
});

// Services
final pronunciationScorerProvider = Provider<PronunciationScorer>((ref) {
  final recognizer = ref.watch(speechRecognizerProvider);
  return AsrSimilarityScorer(recognizer);
});

// Controllers
final homeControllerProvider = StateNotifierProvider<HomeController, HomeState>((ref) {
  return HomeController(
    textSequences: ref.watch(textSequenceRepositoryProvider),
    progress: ref.watch(progressRepositoryProvider),
    ranker: ref.watch(sequenceRankerProvider),
  );
});

// Create overrides for main.dart
Future<List<Override>> createOverrides() async {
  await Hive.initFlutter();
  final progressBox = await Hive.openBox('progress');
  final settingsBox = await Hive.openBox('settings');

  return [
    hiveBoxProvider.overrideWithValue(progressBox),
    // ... other overrides
  ];
}
```

**Design Decision**: Use late initialization with `createOverrides()` rather than FutureProvider for critical dependencies. This ensures the app doesn't render until storage is ready.

---

### `theme.dart`

**Purpose**: Define the dark, black/white theme.

**Responsibilities**:
- Create `ThemeData` with color scheme
- Define text styles for different contexts
- Export reusable style constants

**Color Palette**:

| Token | Value | Usage |
|-------|-------|-------|
| `background` | `#000000` | All backgrounds |
| `onBackground` | `#FFFFFF` | Primary text |
| `onBackgroundSecondary` | `#FFFFFF` @ 60% | Secondary text |
| `divider` | `#FFFFFF` @ 12% | Separators |
| `primary` | `#FFFFFF` | Buttons, icons |
| `onPrimary` | `#000000` | Text on buttons |

**Text Styles**:

| Style | Size | Usage |
|-------|------|-------|
| `displayLarge` | 36sp | Trainer sentence |
| `titleMedium` | 18sp | List item text |
| `labelLarge` | 16sp | Buttons |
| `bodySmall` | 12sp | Scores, metadata |

**Implementation**:

```dart
import 'package:flutter/material.dart';

const _black = Color(0xFF000000);
const _white = Color(0xFFFFFFFF);

final darkTheme = ThemeData(
  useMaterial3: true,
  brightness: Brightness.dark,
  scaffoldBackgroundColor: _black,
  colorScheme: const ColorScheme.dark(
    surface: _black,
    onSurface: _white,
    primary: _white,
    onPrimary: _black,
  ),
  dividerColor: _white.withOpacity(0.12),
  textTheme: const TextTheme(
    displayLarge: TextStyle(
      fontSize: 36,
      fontWeight: FontWeight.w400,
      color: _white,
    ),
    titleMedium: TextStyle(
      fontSize: 18,
      fontWeight: FontWeight.w400,
      color: _white,
    ),
    labelLarge: TextStyle(
      fontSize: 16,
      fontWeight: FontWeight.w500,
      color: _white,
    ),
    bodySmall: TextStyle(
      fontSize: 12,
      fontWeight: FontWeight.w400,
      color: _white,
    ),
  ),
  iconTheme: const IconThemeData(color: _white),
  appBarTheme: const AppBarTheme(
    backgroundColor: _black,
    foregroundColor: _white,
    elevation: 0,
  ),
  elevatedButtonTheme: ElevatedButtonThemeData(
    style: ElevatedButton.styleFrom(
      backgroundColor: _white,
      foregroundColor: _black,
      minimumSize: const Size(120, 48),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(24),
      ),
    ),
  ),
  outlinedButtonTheme: OutlinedButtonThemeData(
    style: OutlinedButton.styleFrom(
      foregroundColor: _white,
      side: const BorderSide(color: _white),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(24),
      ),
    ),
  ),
  iconButtonTheme: IconButtonThemeData(
    style: IconButton.styleFrom(
      foregroundColor: _white,
    ),
  ),
  bottomSheetTheme: const BottomSheetThemeData(
    backgroundColor: _black,
    shape: RoundedRectangleBorder(
      borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
    ),
  ),
);

// Semantic colors for scores
extension ScoreColors on ThemeData {
  Color scoreColor(double score) {
    if (score >= 80) return const Color(0xFF4CAF50); // Green
    if (score >= 50) return const Color(0xFFFFC107); // Amber
    return const Color(0xFFF44336); // Red
  }
}
```

**Design Decision**: Pure black background (#000) for OLED battery savings and distraction-free focus. Score colors are the only non-black/white elements.

---

## Integration Tests

### `app_test.dart`

```dart
void main() {
  group('App initialization', () {
    testWidgets('renders home screen on launch', (tester) async {
      await tester.pumpWidget(
        ProviderScope(
          overrides: testOverrides,
          child: const SpeakToLearnApp(),
        ),
      );

      expect(find.byType(HomeScreen), findsOneWidget);
    });

    testWidgets('navigates to list screen', (tester) async {
      await tester.pumpWidget(
        ProviderScope(
          overrides: testOverrides,
          child: const SpeakToLearnApp(),
        ),
      );

      await tester.tap(find.byIcon(Icons.list));
      await tester.pumpAndSettle();

      expect(find.byType(SequenceListScreen), findsOneWidget);
    });

    testWidgets('navigates back from list to home', (tester) async {
      await tester.pumpWidget(
        ProviderScope(
          overrides: testOverrides,
          child: const SpeakToLearnApp(),
        ),
      );

      // Navigate to list
      await tester.tap(find.byIcon(Icons.list));
      await tester.pumpAndSettle();

      // Tap back
      await tester.tap(find.byIcon(Icons.arrow_back));
      await tester.pumpAndSettle();

      expect(find.byType(HomeScreen), findsOneWidget);
    });
  });

  group('Theme', () {
    testWidgets('applies dark theme', (tester) async {
      await tester.pumpWidget(
        ProviderScope(
          overrides: testOverrides,
          child: const SpeakToLearnApp(),
        ),
      );

      final scaffold = tester.widget<Scaffold>(find.byType(Scaffold));
      expect(scaffold.backgroundColor, const Color(0xFF000000));
    });
  });
}
```
