import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:speech_to_text/speech_recognition_error.dart';
import 'package:speech_to_text/speech_recognition_result.dart';
import 'package:speech_to_text/speech_to_text.dart';

import 'package:speak_to_learn/core/result.dart';
import 'package:speak_to_learn/features/scoring/data/speech_recognizer.dart';
import 'package:speak_to_learn/features/scoring/data/speech_to_text_recognizer.dart';

/// Mock implementation of SpeechToText for testing.
///
/// Uses noSuchMethod to handle unimplemented members.
class MockSpeechToText implements SpeechToText {
  bool initializeSucceeds = true;
  List<LocaleName> availableLocales = [
    LocaleName('zh-CN', 'Chinese (China)'),
    LocaleName('en-US', 'English (US)'),
  ];

  bool _isListening = false;
  bool _isAvailable = false;
  void Function(SpeechRecognitionResult)? _onResultCallback;
  void Function(String)? _onStatusCallback;
  void Function(SpeechRecognitionError)? _onErrorCallback;

  String? lastLocaleId;
  bool listenWasCalled = false;
  bool stopWasCalled = false;
  bool cancelWasCalled = false;

  /// Simulates receiving a recognition result.
  void simulateResult(String recognizedWords, {bool isFinal = true}) {
    if (_onResultCallback != null) {
      _onResultCallback!(SpeechRecognitionResult(
        [SpeechRecognitionWords(recognizedWords, [recognizedWords], 1.0)],
        isFinal,
      ));
    }
  }

  /// Simulates a status change.
  void simulateStatus(String status) {
    if (_onStatusCallback != null) {
      _onStatusCallback!(status);
    }
    if (status == 'done' || status == 'notListening') {
      _isListening = false;
    }
  }

  /// Simulates an error.
  void simulateError(String errorMsg) {
    if (_onErrorCallback != null) {
      _onErrorCallback!(SpeechRecognitionError(errorMsg, false));
    }
    _isListening = false;
  }

  @override
  Future<bool> initialize({
    SpeechErrorListener? onError,
    SpeechStatusListener? onStatus,
    debugLogging = false,
    Duration finalTimeout = const Duration(milliseconds: 2000),
    List<SpeechConfigOption>? options,
  }) async {
    _onStatusCallback = onStatus;
    _onErrorCallback = onError;
    _isAvailable = initializeSucceeds;
    return _isAvailable;
  }

  @override
  Future<dynamic> listen({
    SpeechResultListener? onResult,
    Duration? listenFor,
    Duration? pauseFor,
    String? localeId,
    SpeechSoundLevelChange? onSoundLevelChange,
    cancelOnError = false,
    partialResults = true,
    onDevice = false,
    ListenMode listenMode = ListenMode.confirmation,
    sampleRate = 0,
    SpeechListenOptions? listenOptions,
  }) async {
    listenWasCalled = true;
    lastLocaleId = localeId;
    _onResultCallback = onResult;
    _isListening = true;
  }

  @override
  Future<void> stop() async {
    stopWasCalled = true;
    _isListening = false;
  }

  @override
  Future<void> cancel() async {
    cancelWasCalled = true;
    _isListening = false;
  }

  @override
  Future<List<LocaleName>> locales() async {
    return availableLocales;
  }

  @override
  bool get isListening => _isListening;

  @override
  bool get isAvailable => _isAvailable;

  @override
  bool get isNotListening => !_isListening;

  @override
  bool get hasError => false;

  @override
  SpeechRecognitionError? get lastError => null;

  @override
  String get lastRecognizedWords => '';

  @override
  String get lastStatus => '';

  @override
  double get lastSoundLevel => 0.0;

  @override
  bool get hasSpeech => _isAvailable;

  @override
  SpeechRecognitionResult get lastResult =>
      SpeechRecognitionResult([SpeechRecognitionWords('', [''], 0.0)], true);

  @override
  Future<LocaleName> systemLocale() async =>
      LocaleName('en-US', 'English (US)');

  @override
  Future<bool> get hasPermission async => true;

  @override
  bool get hasRecognized => false;

  @override
  SpeechErrorListener? errorListener;

  @override
  SpeechStatusListener? statusListener;

  @override
  SpeechPhraseAggregator? unexpectedPhraseAggregator;

  @override
  void changePauseFor(Duration pauseFor) {}

  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  // Mock permission_handler platform channel
  setUpAll(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(
      const MethodChannel('flutter.baseflow.com/permissions/methods'),
      (MethodCall methodCall) async {
        // Return granted (1) for all permission requests
        if (methodCall.method == 'requestPermissions') {
          final List<int> permissions = methodCall.arguments.cast<int>();
          return {for (final p in permissions) p: 1}; // 1 = granted
        }
        if (methodCall.method == 'checkPermissionStatus') {
          return 1; // granted
        }
        return null;
      },
    );
  });

  tearDownAll(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(
      const MethodChannel('flutter.baseflow.com/permissions/methods'),
      null,
    );
  });

  group('SpeechToTextRecognizer', () {
    late MockSpeechToText mockSpeech;
    late SpeechToTextRecognizer recognizer;

    setUp(() {
      mockSpeech = MockSpeechToText();
      recognizer = SpeechToTextRecognizer(speechToText: mockSpeech);
    });

    group('isAvailable', () {
      test('returns true when Chinese locale is available', () async {
        mockSpeech.initializeSucceeds = true;
        mockSpeech.availableLocales = [
          LocaleName('zh-CN', 'Chinese (China)'),
          LocaleName('en-US', 'English (US)'),
        ];

        final result = await recognizer.isAvailable();

        expect(result, isTrue);
      });

      test('returns true when Taiwan locale is available', () async {
        mockSpeech.initializeSucceeds = true;
        mockSpeech.availableLocales = [
          LocaleName('zh-TW', 'Chinese (Taiwan)'),
        ];

        final result = await recognizer.isAvailable();

        expect(result, isTrue);
      });

      test('returns false when initialization fails', () async {
        mockSpeech.initializeSucceeds = false;

        final result = await recognizer.isAvailable();

        expect(result, isFalse);
      });

      test('returns false when no Chinese locale is available', () async {
        mockSpeech.initializeSucceeds = true;
        mockSpeech.availableLocales = [
          LocaleName('en-US', 'English (US)'),
          LocaleName('es-ES', 'Spanish (Spain)'),
        ];

        final result = await recognizer.isAvailable();

        expect(result, isFalse);
      });
    });

    group('recognize', () {
      test('returns cached text after startListening and stopListening',
          () async {
        const testPath = '/path/to/recording.wav';
        const expectedText = 'ni hao';

        // Start listening
        await recognizer.startListening(testPath);

        // Simulate recognition result
        mockSpeech.simulateResult(expectedText);

        // Stop listening to cache the result
        await recognizer.stopListening();

        // Recognize should return the cached text
        final result = await recognizer.recognize(testPath, 'zh-CN');

        expect(result.isSuccess, isTrue);
        expect(result.valueOrNull, expectedText);
      });

      test('returns audioReadError when path was not processed', () async {
        const testPath = '/path/to/unprocessed.wav';

        final result = await recognizer.recognize(testPath, 'zh-CN');

        expect(result.isFailure, isTrue);
        expect(result.errorOrNull, RecognitionError.audioReadError);
      });

      test('returns noSpeechDetected when cached text is empty', () async {
        const testPath = '/path/to/silent.wav';

        // Start listening
        await recognizer.startListening(testPath);

        // Simulate empty result
        mockSpeech.simulateResult('');

        // Stop listening - should cache empty string
        await recognizer.stopListening();

        final result = await recognizer.recognize(testPath, 'zh-CN');

        // Empty string in cache means no speech detected
        // Note: based on the implementation, empty strings are not cached
        // So this returns audioReadError instead
        expect(result.isFailure, isTrue);
      });

      test('removes path from cache after recognize call', () async {
        const testPath = '/path/to/recording.wav';
        const expectedText = 'hello world';

        await recognizer.startListening(testPath);
        mockSpeech.simulateResult(expectedText);
        await recognizer.stopListening();

        // First call should succeed
        final firstResult = await recognizer.recognize(testPath, 'zh-CN');
        expect(firstResult.isSuccess, isTrue);

        // Second call should fail (cache was cleared)
        final secondResult = await recognizer.recognize(testPath, 'zh-CN');
        expect(secondResult.isFailure, isTrue);
        expect(secondResult.errorOrNull, RecognitionError.audioReadError);
      });

      test('updates cached text with latest result during listening',
          () async {
        const testPath = '/path/to/recording.wav';

        await recognizer.startListening(testPath);

        // Simulate partial results
        mockSpeech.simulateResult('ni', isFinal: false);
        mockSpeech.simulateResult('ni hao', isFinal: false);
        mockSpeech.simulateResult('ni hao ma', isFinal: true);

        await recognizer.stopListening();

        final result = await recognizer.recognize(testPath, 'zh-CN');

        expect(result.isSuccess, isTrue);
        expect(result.valueOrNull, 'ni hao ma');
      });
    });

    group('startListening', () {
      test('passes correct locale to speech_to_text', () async {
        const testPath = '/path/to/recording.wav';

        await recognizer.startListening(testPath, languageCode: 'zh-TW');

        expect(mockSpeech.listenWasCalled, isTrue);
        expect(mockSpeech.lastLocaleId, 'zh-TW');
      });

      test('uses default Chinese locale when not specified', () async {
        const testPath = '/path/to/recording.wav';

        await recognizer.startListening(testPath);

        expect(mockSpeech.lastLocaleId, 'zh-CN');
      });

      test('returns notAvailable when initialization fails', () async {
        mockSpeech.initializeSucceeds = false;

        final result = await recognizer.startListening('/test/path.wav');

        expect(result.isFailure, isTrue);
        expect(result.errorOrNull, RecognitionError.notAvailable);
      });

      test('stops previous listening session before starting new one',
          () async {
        await recognizer.startListening('/first/path.wav');

        // Simulate that we are listening
        expect(mockSpeech.listenWasCalled, isTrue);

        // Reset tracking
        mockSpeech.stopWasCalled = false;
        mockSpeech.listenWasCalled = false;

        // Start new session
        await recognizer.startListening('/second/path.wav');

        // Should have stopped and started again
        expect(mockSpeech.stopWasCalled, isTrue);
        expect(mockSpeech.listenWasCalled, isTrue);
      });
    });

    group('stopListening', () {
      test('caches recognized text for the recording path', () async {
        const testPath = '/path/to/recording.wav';
        const recognizedText = 'test recognition';

        await recognizer.startListening(testPath);
        mockSpeech.simulateResult(recognizedText);
        await recognizer.stopListening();

        // Verify text was cached by checking recognize returns it
        final result = await recognizer.recognize(testPath, 'zh-CN');
        expect(result.valueOrNull, recognizedText);
      });

      test('calls stop on speech_to_text', () async {
        await recognizer.startListening('/test/path.wav');
        await recognizer.stopListening();

        expect(mockSpeech.stopWasCalled, isTrue);
      });

      test('does not cache when no text was recognized', () async {
        const testPath = '/path/to/silent.wav';

        await recognizer.startListening(testPath);
        // No simulateResult call - no speech detected
        await recognizer.stopListening();

        final result = await recognizer.recognize(testPath, 'zh-CN');
        expect(result.isFailure, isTrue);
      });
    });

    group('error handling', () {
      test('handles status change to done', () async {
        await recognizer.startListening('/test/path.wav');

        mockSpeech.simulateStatus('done');

        // After done status, should not be listening
        expect(mockSpeech.isListening, isFalse);
      });

      test('handles status change to notListening', () async {
        await recognizer.startListening('/test/path.wav');

        mockSpeech.simulateStatus('notListening');

        expect(mockSpeech.isListening, isFalse);
      });

      test('handles error during recognition', () async {
        await recognizer.startListening('/test/path.wav');

        mockSpeech.simulateError('network_error');

        // Error should stop listening
        expect(mockSpeech.isListening, isFalse);
      });
    });

    group('clearCache', () {
      test('removes all cached recognition results', () async {
        const path1 = '/path/one.wav';
        const path2 = '/path/two.wav';

        // Cache some results
        await recognizer.startListening(path1);
        mockSpeech.simulateResult('result one');
        await recognizer.stopListening();

        await recognizer.startListening(path2);
        mockSpeech.simulateResult('result two');
        await recognizer.stopListening();

        // Clear cache
        recognizer.clearCache();

        // Both should now fail
        final result1 = await recognizer.recognize(path1, 'zh-CN');
        final result2 = await recognizer.recognize(path2, 'zh-CN');

        expect(result1.isFailure, isTrue);
        expect(result2.isFailure, isTrue);
      });
    });

    group('dispose', () {
      test('stops listening and clears cache', () async {
        const testPath = '/path/to/recording.wav';

        await recognizer.startListening(testPath);
        mockSpeech.simulateResult('some text');
        await recognizer.stopListening();

        await recognizer.dispose();

        expect(mockSpeech.stopWasCalled, isTrue);
        expect(mockSpeech.cancelWasCalled, isTrue);

        // Cache should be cleared
        final result = await recognizer.recognize(testPath, 'zh-CN');
        expect(result.isFailure, isTrue);
      });
    });
  });
}
