import '../../recording/domain/recording.dart';
import '../../text_sequences/domain/text_sequence.dart';
import '../domain/grade.dart';
import '../domain/pronunciation_scorer.dart';
import 'cer_calculator.dart';
import 'speech_recognizer.dart';

/// Pronunciation scorer using ASR (Automatic Speech Recognition) and CER.
///
/// This scorer:
/// 1. Uses a [SpeechRecognizer] to transcribe the user's recording
/// 2. Compares the transcription to the expected text using CER
/// 3. Converts the CER to a 0-100 score
class AsrSimilarityScorer implements PronunciationScorer {
  AsrSimilarityScorer({
    required SpeechRecognizer recognizer,
    CerCalculator? calculator,
  }) : _recognizer = recognizer,
       _calculator = calculator ?? CerCalculator();

  final SpeechRecognizer _recognizer;
  final CerCalculator _calculator;

  static const _method = 'asr_cer_v1';

  @override
  Future<Grade> score(TextSequence sequence, Recording recording) async {
    final recognitionResult = await _recognizer.recognize(
      recording.filePath,
      sequence.language,
    );

    return recognitionResult.when(
      success: (recognizedText) {
        final cerResult = _calculator.calculate(sequence.text, recognizedText);

        return Grade(
          overall: cerResult.score,
          method: _method,
          accuracy: cerResult.accuracy,
          completeness: cerResult.completeness,
          recognizedText: recognizedText,
          details: {
            'cer': cerResult.cer,
            'editDistance': cerResult.editDistance,
            'referenceLength': cerResult.referenceLength,
            'hypothesisLength': cerResult.hypothesisLength,
          },
        );
      },
      failure: (error) {
        // Return a zero score on recognition failure
        return Grade(
          overall: 0,
          method: _method,
          accuracy: 0,
          completeness: 0,
          details: {'error': error.name},
        );
      },
    );
  }
}
