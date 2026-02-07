import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:speak_to_learn/app/theme.dart';
import 'package:speak_to_learn/features/text_sequences/domain/text_sequence.dart';

import '../mocks/mock_audio.dart';
import '../mocks/mock_providers.dart';

/// Extension on WidgetTester for common test operations.
extension PumpHelpers on WidgetTester {
  /// Pumps a widget wrapped in MaterialApp with test providers.
  ///
  /// Use this for testing individual widgets in isolation.
  Future<void> pumpApp(
    Widget widget, {
    List<Override> overrides = const [],
    List<TextSequence>? sequences,
  }) async {
    await pumpWidget(
      ProviderScope(
        overrides: [
          ...createTestOverrides(sequences: sequences),
          ...overrides,
        ],
        child: MaterialApp(
          theme: AppTheme.darkTheme,
          home: widget,
          debugShowCheckedModeBanner: false,
        ),
      ),
    );
  }

  /// Pumps a widget wrapped with mocked audio services.
  ///
  /// Use this for testing recording and playback functionality.
  Future<void> pumpAppWithMockedAudio(
    Widget widget, {
    List<Override> overrides = const [],
    List<TextSequence>? sequences,
    FakeAudioRecorder? recorder,
    FakeAudioPlayer? player,
  }) async {
    await pumpWidget(
      ProviderScope(
        overrides: [
          ...createTestOverridesWithMockedAudio(
            sequences: sequences,
            recorder: recorder,
            player: player,
          ),
          ...overrides,
        ],
        child: MaterialApp(
          theme: AppTheme.darkTheme,
          home: widget,
          debugShowCheckedModeBanner: false,
        ),
      ),
    );
  }

  /// Pumps and settles the widget tree until animations complete.
  Future<void> pumpAndSettleApp(
    Widget widget, {
    List<Override> overrides = const [],
    List<TextSequence>? sequences,
    Duration duration = const Duration(seconds: 5),
  }) async {
    await pumpApp(widget, overrides: overrides, sequences: sequences);
    await pumpAndSettle(duration);
  }

  /// Taps on a widget by finder and pumps.
  Future<void> tapAndPump(Finder finder, {int pumps = 1}) async {
    await tap(finder);
    for (var i = 0; i < pumps; i++) {
      await pump();
    }
  }

  /// Taps on a widget and settles all animations.
  Future<void> tapAndSettle(Finder finder) async {
    await tap(finder);
    await pumpAndSettle();
  }
}
