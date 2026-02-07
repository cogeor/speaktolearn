# Implementation Review Report

**Date:** 2026-02-07
**Scope:** Loops 02b-04, A2-A3, B2, C2, E2-E4, T2
**Commits Reviewed:** 11 commits (c05a52a → e67e67a)

---

## Executive Summary

The implementation successfully delivered 13 planned loops covering recording UI, settings, stats visualization, filtering, and integration tests. However, this review identified **several critical issues** that require attention before production release:

1. **Duplicate state management** between HomeState and RecordingState creates synchronization complexity
2. **VoicePreference setting is dead code** - saved but never used for audio playback
3. **Stats calculations are incorrect** - averages best scores per sequence, not all attempts
4. **AudioPlayer resource leak** - never disposed in RecordingController
5. **Missing integration test coverage** for Settings and Stats screens

---

## Table of Contents

1. [Critical Issues](#1-critical-issues)
2. [Architecture & Design Issues](#2-architecture--design-issues)
3. [Dead Code & Unused Features](#3-dead-code--unused-features)
4. [Integration Gaps](#4-integration-gaps)
5. [Testing Gaps](#5-testing-gaps)
6. [Performance Concerns](#6-performance-concerns)
7. [Deviations from Study Documents](#7-deviations-from-study-documents)
8. [Recommendations by Priority](#8-recommendations-by-priority)

---

## 1. Critical Issues

### 1.1 AudioPlayer Never Disposed (Memory Leak)

**Location:** `recording_controller.dart:302-306`

**Issue:** The `RecordingController` receives `AudioPlayer` via dependency injection but never disposes it:

```dart
@override
void dispose() {
  _cancelTimers();
  waveformController.dispose();
  super.dispose();
  // MISSING: _audioPlayer.dispose();
}
```

**Impact:**
- `JustAudioPlayer` creates a `StreamController<PlaybackState>.broadcast()` that leaks
- Platform audio resources accumulate over navigation cycles
- Potential crash on devices with limited resources

**Fix:** Add `await _audioPlayer.dispose();` before `super.dispose()`

---

### 1.2 Race Condition in stopAndScore() Failure Path

**Location:** `recording_controller.dart:237-246`

**Issue:** The failure callback is synchronous while success is async:

```dart
return stopResult.when(
  success: (filePath) async {
    // ... async scoring operations
    return grade;
  },
  failure: (error) {  // NOT async - returns immediately
    HapticFeedback.heavyImpact();
    state = state.copyWith(/* ... */);
    return null;
  },
);
```

**Impact:** Caller receives `null` before recorder actually stops, potentially allowing premature re-recording.

**Fix:** Make failure callback `async` and await any cleanup operations.

---

### 1.3 Duplicate State Management (RecordingStatus)

**Location:** `home_state.dart:22` vs `recording_state.dart:10-12`

**Issue:** Two separate state systems track recording status:

| HomeState | RecordingState |
|-----------|----------------|
| `RecordingStatus.idle` | `isRecording: false, isScoring: false` |
| `RecordingStatus.recording` | `isRecording: true` |
| `RecordingStatus.processing` | `isScoring: true` |

This requires manual synchronization in `home_screen.dart:40-48`:

```dart
ref.listen<RecordingState>(recordingControllerProvider, (previous, next) {
  if (next.isRecording && state.recordingStatus != RecordingStatus.recording) {
    controller.setRecordingStatus(RecordingStatus.recording);
  }
  // ... more sync logic
});
```

**Impact:**
- Two sources of truth for the same concept
- Risk of state divergence
- Maintenance burden

**Fix:** Remove `recordingStatus` from HomeState; derive it from RecordingState using a computed getter.

---

### 1.4 Stats Average Score Calculation is Wrong

**Location:** `stats_controller.dart:29-47`

**Issue:** Calculates average of **best scores per sequence**, not average of all attempts:

```dart
if (progress.bestScore != null) {
  totalScore += progress.bestScore!;
  scoredAttempts++;
}
// Result: average of best scores only
```

**Impact:** "Average Score" stat card shows inflated numbers that don't reflect actual practice performance.

**Fix:** Sum all attempt scores from `TextSequenceProgress.attempts` or use a dedicated repository method.

---

## 2. Architecture & Design Issues

### 2.1 HomeController Has Recording-Specific Methods

**Location:** `home_controller.dart:116-123`

Methods that exist only to sync recording state:
- `setRecordingStatus(RecordingStatus status)`
- `setLatestScore(int? score)`

These violate separation of concerns. HomeController should manage home screen state, not recording state.

**Recommendation:** Remove these methods; use RecordingController state directly.

---

### 2.2 Timer Management Issues

**Location:** `recording_controller.dart:99-111`

**Problems:**
1. Auto-stop timer fires asynchronously but isn't awaited
2. Countdown timer continues after auto-stop fires (briefly out of sync)
3. No synchronization between countdown reaching zero and auto-stop

```dart
_autoStopTimer = Timer(duration, () {
  if (state.isRecording && _currentTextSequence != null) {
    stopAndScore(_currentTextSequence!);  // Fire and forget
  }
});
```

**Recommendation:** Restructure timer logic to ensure countdown and auto-stop are synchronized.

---

### 2.3 Stream Consumption Without Timeout

**Location:** `recording_controller.dart:270-276`

```dart
await _audioPlayer.stateStream.firstWhere(
  (playbackState) =>
      playbackState == PlaybackState.completed ||
      playbackState == PlaybackState.idle ||
      playbackState == PlaybackState.error,
);
```

**Issue:** No timeout protection. If player never emits terminal state, UI hangs indefinitely.

**Fix:** Add `.timeout(Duration(seconds: 30))` with fallback.

---

### 2.4 Inconsistent Error Handling

**Location:** Multiple files

| Location | Issue |
|----------|-------|
| `home_screen.dart:60-67` | Only handles startRecording errors, not stopAndScore |
| `recording_controller.dart:226-234` | Generic catch loses error type info |
| `practice_sheet.dart` | No error display for scoring failures |

**Recommendation:** Centralize error handling; use typed error classes.

---

## 3. Dead Code & Unused Features

### 3.1 VoicePreference - Complete Dead Code

**Status:** UI exists, saves to database, but NEVER used for audio playback

**Files involved:**
- `app_settings.dart:7-11` - VoicePreference enum defined
- `settings_screen.dart:33-52` - UI dropdown exists
- `settings_controller.dart:31-37` - updateVoicePreference() saves value

**Not connected to:**
- `example_audio_controller.dart` - doesn't read setting
- `practice_sheet.dart` - shows all voices, ignores preference
- Any audio playback code

**Evidence:** Search for `voicePreference|VoicePreference` found zero references outside settings feature.

---

### 3.2 Unused AppSettings Fields

**Location:** `app_settings.dart`

| Field | UI | Saved | Used |
|-------|:--:|:-----:|:----:|
| `uiLanguageCode` | - | Yes | No |
| `targetLanguageCode` | - | Yes | No |
| `showGloss` | - | Yes | No |
| `playbackSpeed` | - | Yes | No |
| `autoPlayExample` | - | Yes | No |
| `preferredVoiceId` | - | Yes | No |

All these fields are persisted to Hive but never read by any feature.

---

### 3.3 checkLatestRecording() Never Called

**Location:** `recording_controller.dart:253-256`

```dart
Future<void> checkLatestRecording(String textSequenceId) async {
  final recording = await _repository.getLatest(textSequenceId);
  state = state.copyWith(hasLatestRecording: recording != null);
}
```

This method is defined but never invoked. The `hasLatestRecording` state is only set in `stopAndScore()`, meaning replay button won't appear when revisiting a previously-recorded sequence.

---

### 3.4 DurationFormatting Extension Unused

**Location:** `recording_duration_calculator.dart:33-40`

```dart
extension DurationFormatting on Duration {
  String toCountdownString() { /* ... */ }
}
```

Defined but never used anywhere in the codebase.

---

### 3.5 cancel() Method Not Integrated

**Location:** `recording_controller.dart:287-299`

The `cancel()` method exists but is never called from UI. If user dismisses practice sheet while recording, there's no cleanup mechanism.

**Should be called from:** WillPopScope or navigation handler in practice_sheet.dart

---

## 4. Integration Gaps

### 4.1 showRomanization Setting Not Reactive

**Location:** `home_screen.dart:167-183`

The pinyin setting is read once on initialization:

```dart
void didChangeDependencies() {
  if (!_initializedFromSettings) {
    final settingsAsync = ref.read(settingsControllerProvider);
    // ... set _showPinyin from settings ONCE
  }
}
```

**Problem:** Changes in Settings screen don't propagate back to home screen without manual toggle.

**Fix:** Watch the settings provider reactively instead of one-time read.

---

### 4.2 Waveform in Practice Sheet Only

**Location:** `practice_sheet.dart:375-394`

Waveform widget exists in practice sheet but not in home screen, despite home screen having recording FAB. Users recording from home screen don't see waveform feedback.

**Recommendation:** Consider adding waveform to home screen during recording.

---

### 4.3 Score Display Inconsistency

**PracticeSheet (`practice_sheet.dart:178-231`):**
- Shows Latest score AND Best score
- Shows accuracy/completeness breakdown
- Shows recognized text comparison

**HomeScreen (`home_screen.dart:290-307`):**
- Only shows Best score
- No breakdown
- No recognized text

**Issue:** `latestScore` is stored in HomeState but never displayed.

---

### 4.4 Heatmap Data Limitation

**Location:** `stats_controller.dart:36-44`

Heatmap uses `lastAttemptAt` per sequence, not individual attempt timestamps. If user practices same sequence twice on same day, it's counted as one attempt in heatmap.

---

## 5. Testing Gaps

### 5.1 Missing Test Coverage

| Feature | Integration Tests | Unit Tests |
|---------|:-----------------:|:----------:|
| Settings Screen | None | None |
| Stats Screen | None | None |
| Audio Playback | None | None |
| Permission Handling | None | None |
| Error Recovery | None | None |

### 5.2 Fragile Test Infrastructure

**Location:** `test_helpers.dart:276-298`

```dart
final sequenceGesture = find.byWidgetPredicate(
  (widget) =>
      widget is GestureDetector &&
      widget.child is Text &&
      (widget.child as Text).style?.fontSize != null,
);
```

This widget finder relies on:
- Widget type hierarchy
- Text styling details
- Fragile assumptions that break with UI changes

### 5.3 Mock Limitations

| Mock | Missing Scenarios |
|------|-------------------|
| MockSpeechRecognizer | Recognition delays, network failures |
| MockAudioRecorder | Permission denied, disk full, recording errors |
| MockAudioPlayer | File not found, codec errors |

---

## 6. Performance Concerns

### 6.1 Redundant Provider Invalidation

**Location:** `sequence_list_controller.dart:89-103`

```dart
void toggleHskFilter(int level) {
  ref.read(hskFilterProvider.notifier).state = /* updated */;
  ref.invalidateSelf();  // REDUNDANT - already invalidated by watching
}
```

This causes two rebuild cycles instead of one.

### 6.2 Stats Recomputed on Every View

**Location:** `stats_controller.dart`

No caching mechanism. Stats are recomputed every time the screen is viewed, iterating through all tracked progress.

### 6.3 Filtering on Every Build

**Location:** `sequence_list_controller.dart:49-54`

Filter logic runs on every controller rebuild (O(n log n) for filter + sort). Acceptable for small datasets but could be optimized with caching.

---

## 7. Deviations from Study Documents

### 7.1 From S.md (UX Research)

| Requirement | Status | Notes |
|-------------|--------|-------|
| Tap-to-Start/Stop toggle | Implemented | Working correctly |
| 56dp FAB minimum | Implemented | Uses default FAB size |
| Recording state indicators | Partial | Timer countdown exists but not displayed in UI |
| Waveform visualization | Implemented | Works in practice sheet |
| Haptic feedback | Implemented | All patterns from spec |
| Post-recording playback | Partial | Replay exists but `checkLatestRecording()` never called |
| Color-coded matching | Not Implemented | Grade shows recognized text but no character-level highlighting |

### 7.2 From I.md (Introspection)

| Finding | Status |
|---------|--------|
| Implement real SpeechRecognizer | Done (in parent loops) |
| Fix score display bug (both showing bestScore) | Fixed |
| Save recording after scoring | Still not done - `_repository.saveLatest()` not called |
| Add unit tests for AsrSimilarityScorer | Not done |

### 7.3 From T.md (Template Patterns)

| Pattern | Status |
|---------|--------|
| Use Case abstraction | Done (E1) |
| Assessment Status enum | Partially - HomeState has RecordingStatus |
| Auto-stop timer | Done (E2) |
| Granular scoring (accuracy/completeness) | Done (E5) |
| Per-character breakdown | Not done |

---

## 8. Recommendations by Priority

### P0 - Critical (Fix Before Release)

| Issue | Location | Effort |
|-------|----------|--------|
| AudioPlayer memory leak | recording_controller.dart:302-306 | 1 line |
| Remove duplicate RecordingStatus | home_state.dart, home_screen.dart | Medium |
| Fix stats average calculation | stats_controller.dart:29-47 | Small |
| Wire VoicePreference to audio | example_audio_controller.dart | Medium |

### P1 - High Priority

| Issue | Location | Effort |
|-------|----------|--------|
| Add Settings integration tests | integration_test/ | Medium |
| Add Stats integration tests | integration_test/ | Medium |
| Call checkLatestRecording() on load | practice_sheet.dart | Small |
| Make pinyin setting reactive | home_screen.dart | Small |
| Add timeout to stream consumption | recording_controller.dart:270-276 | Small |

### P2 - Medium Priority

| Issue | Location | Effort |
|-------|----------|--------|
| Fix streak calculation edge cases | stats_controller.dart:65-102 | Medium |
| Remove redundant invalidateSelf | sequence_list_controller.dart:96,102 | Trivial |
| Call cancel() on sheet dismiss | practice_sheet.dart | Small |
| Add countdown display to UI | home_screen.dart | Medium |
| Add per-character highlighting | practice_sheet.dart | Large |

### P3 - Low Priority (Tech Debt)

| Issue | Location | Effort |
|-------|----------|--------|
| Remove unused settings fields | app_settings.dart | Small |
| Remove DurationFormatting extension | recording_duration_calculator.dart | Trivial |
| Refactor test helpers | test_helpers.dart:276-298 | Medium |
| Add stats caching | stats_controller.dart | Medium |
| Extract shared score display widget | Multiple files | Medium |

---

## Summary Statistics

| Category | Count |
|----------|-------|
| Critical Issues | 4 |
| Architecture Issues | 4 |
| Dead Code Items | 5 |
| Integration Gaps | 4 |
| Missing Tests | 6 feature areas |
| Performance Issues | 3 |
| Study Document Deviations | 8 |

---

## Appendix: Files Reviewed

### Modified in This Implementation
- `apps/mobile_flutter/lib/features/practice/presentation/home_screen.dart`
- `apps/mobile_flutter/lib/features/practice/presentation/home_controller.dart`
- `apps/mobile_flutter/lib/features/practice/presentation/practice_sheet.dart`
- `apps/mobile_flutter/lib/features/practice/presentation/widgets/record_fab.dart`
- `apps/mobile_flutter/lib/features/recording/presentation/recording_controller.dart`
- `apps/mobile_flutter/lib/features/recording/presentation/recording_state.dart`
- `apps/mobile_flutter/lib/features/recording/domain/recording_duration_calculator.dart`
- `apps/mobile_flutter/lib/features/recording/presentation/widgets/recording_waveform.dart`
- `apps/mobile_flutter/lib/features/settings/presentation/settings_screen.dart`
- `apps/mobile_flutter/lib/features/settings/presentation/settings_controller.dart`
- `apps/mobile_flutter/lib/features/settings/domain/app_settings.dart`
- `apps/mobile_flutter/lib/features/stats/presentation/stats_screen.dart`
- `apps/mobile_flutter/lib/features/stats/presentation/widgets/activity_heatmap.dart`
- `apps/mobile_flutter/lib/features/text_sequences/presentation/sequence_list_screen.dart`
- `apps/mobile_flutter/lib/features/text_sequences/presentation/sequence_list_controller.dart`
- `apps/mobile_flutter/lib/features/text_sequences/presentation/widgets/hsk_filter_chips.dart`
- `apps/mobile_flutter/lib/core/haptics/haptic_patterns.dart`
- `apps/mobile_flutter/integration_test/practice_flow_test.dart`
- `apps/mobile_flutter/integration_test/test_helpers.dart`

### Study Documents Referenced
- `.delegate/work/20260207-191201-ux-features-implementation/S.md`
- `.delegate/work/20260207-191201-ux-features-implementation/I.md`
- `.delegate/work/20260207-191201-ux-features-implementation/T.md`
- `.delegate/work/20260207-192522-deferred-features-recording/LOOPS.yaml`
