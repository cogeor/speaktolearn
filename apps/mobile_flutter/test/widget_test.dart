import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:speak_to_learn/app/theme.dart';

void main() {
  test('AppTheme provides dark theme', () {
    final theme = AppTheme.darkTheme;

    expect(theme.brightness, Brightness.dark);
    expect(theme.scaffoldBackgroundColor, AppTheme.background);
    expect(theme.colorScheme.primary, AppTheme.foreground);
  });

  test('ScoreColors extension returns correct colors', () {
    expect(85.scoreColor, AppTheme.success);
    expect(65.scoreColor, AppTheme.warning);
    expect(30.scoreColor, AppTheme.error);
  });
}
