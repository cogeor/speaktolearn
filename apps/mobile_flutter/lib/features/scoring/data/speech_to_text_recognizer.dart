import 'dart:async';

import 'package:permission_handler/permission_handler.dart';
import 'package:speech_to_text/speech_recognition_error.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;

import '../../../core/result.dart';
import 'speech_recognizer.dart';

/// SpeechRecognizer implementation using the speech_to_text package.
///
/// IMPORTANT: The speech_to_text package only supports live microphone input,
/// not pre-recorded audio files. This implementation provides a workaround
/// by caching recognized text during live recording sessions.
///
/// Usage pattern:
/// 1. Call [startListening] when recording begins
/// 2. Call [stopListening] when recording ends
/// 3. Call [recognize] with the audio path to retrieve cached result
class SpeechToTextRecognizer implements SpeechRecognizer {
  SpeechToTextRecognizer({
    stt.SpeechToText? speechToText,
  }) : _speech = speechToText ?? stt.SpeechToText();

  final stt.SpeechToText _speech;
  bool _isInitialized = false;
  bool _isListening = false;

  /// Cache of recognized text from live sessions.
  /// Maps recording path to recognized text.
  final Map<String, String> _recognizedTextCache = {};

  /// Current recording path being processed.
  String? _currentRecordingPath;

  /// Latest recognized text from current session.
  String _currentRecognizedText = '';

  /// Default locale for Chinese speech recognition.
  static const String _chineseLocale = 'zh-CN';

  /// Timeout for recognition in seconds.
  static const int _listenTimeoutSeconds = 30;

  /// Initializes the speech recognition engine.
  Future<bool> _initialize() async {
    if (_isInitialized) return true;

    // Request microphone permission
    final status = await Permission.microphone.request();
    if (!status.isGranted) {
      return false;
    }

    // Request speech recognition permission (iOS)
    final speechStatus = await Permission.speech.request();
    if (!speechStatus.isGranted && !speechStatus.isLimited) {
      // On Android, speech permission may not be required
      // Continue if we have microphone permission
    }

    _isInitialized = await _speech.initialize(
      onStatus: _onStatus,
      onError: _onError,
    );

    return _isInitialized;
  }

  void _onStatus(String status) {
    // Handle status changes: listening, notListening, done
    if (status == 'done' || status == 'notListening') {
      _isListening = false;
    }
  }

  void _onError(SpeechRecognitionError error) {
    // Log error for debugging
    _isListening = false;
  }

  /// Starts live speech recognition.
  ///
  /// Call this when recording begins. The [recordingPath] should match
  /// the path that will be passed to [recognize] later.
  Future<Result<void, RecognitionError>> startListening(
    String recordingPath, {
    String languageCode = _chineseLocale,
  }) async {
    final initialized = await _initialize();
    if (!initialized) {
      return const Failure(RecognitionError.notAvailable);
    }

    if (_isListening) {
      await stopListening();
    }

    _currentRecordingPath = recordingPath;
    _currentRecognizedText = '';

    try {
      await _speech.listen(
        onResult: (result) {
          _currentRecognizedText = result.recognizedWords;
        },
        localeId: languageCode,
        listenFor: Duration(seconds: _listenTimeoutSeconds),
        pauseFor: const Duration(seconds: 3),
        listenOptions: stt.SpeechListenOptions(
          partialResults: true,
          cancelOnError: false,
          listenMode: stt.ListenMode.dictation,
        ),
      );
      _isListening = true;
      return const Success(null);
    } catch (e) {
      return const Failure(RecognitionError.recognitionFailed);
    }
  }

  /// Stops live speech recognition and caches the result.
  Future<void> stopListening() async {
    if (_isListening) {
      await _speech.stop();
      _isListening = false;
    }

    // Cache the recognized text
    if (_currentRecordingPath != null && _currentRecognizedText.isNotEmpty) {
      _recognizedTextCache[_currentRecordingPath!] = _currentRecognizedText;
    }

    _currentRecordingPath = null;
  }

  @override
  Future<Result<String, RecognitionError>> recognize(
    String audioPath,
    String languageCode,
  ) async {
    // Check cache for pre-recorded recognition result
    final cachedText = _recognizedTextCache.remove(audioPath);

    if (cachedText != null && cachedText.isNotEmpty) {
      return Success(cachedText);
    }

    // If no cached result, recognition was not performed during recording
    // or no speech was detected
    if (cachedText != null && cachedText.isEmpty) {
      return const Failure(RecognitionError.noSpeechDetected);
    }

    // Audio file was not processed - this indicates the live recognition
    // was not started for this recording
    return const Failure(RecognitionError.audioReadError);
  }

  @override
  Future<bool> isAvailable() async {
    final initialized = await _initialize();
    if (!initialized) return false;

    // Check if Chinese locale is available
    final locales = await _speech.locales();
    return locales.any((locale) =>
      locale.localeId.startsWith('zh') ||
      locale.localeId.contains('CN') ||
      locale.localeId.contains('TW')
    );
  }

  /// Clears the recognition cache.
  void clearCache() {
    _recognizedTextCache.clear();
  }

  /// Disposes resources.
  Future<void> dispose() async {
    await _speech.stop();
    await _speech.cancel();
    _recognizedTextCache.clear();
    _isInitialized = false;
  }
}
