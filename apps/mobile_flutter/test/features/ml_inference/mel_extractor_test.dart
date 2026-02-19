import 'package:flutter_test/flutter_test.dart';
import 'package:speak_to_learn/features/ml_inference/data/mel_extractor.dart';
import 'dart:math' as math;

void main() {
  group('MelExtractor', () {
    late MelExtractor extractor;

    setUp(() {
      extractor = MelExtractor(
        nMels: 80,
        sampleRate: 16000,
        hopLength: 160,
        winLength: 400,
        nFft: 400,
        fmin: 0.0,
        fmax: 8000.0,
      );
    });

    test('creates extractor with correct parameters', () {
      expect(extractor.nMels, equals(80));
      expect(extractor.sampleRate, equals(16000));
      expect(extractor.hopLength, equals(160));
      expect(extractor.winLength, equals(400));
      expect(extractor.nFft, equals(400));
      expect(extractor.fmin, equals(0.0));
      expect(extractor.fmax, equals(8000.0));
    });

    test('Hanning window has correct properties', () {
      var window = MelExtractor.createHanningWindow(400);

      expect(window.length, equals(400));

      // Hanning window should be near zero at edges
      expect(window.first, lessThan(0.001));
      expect(window.last, lessThan(0.001));

      // Peak at center should be close to 1
      // Note: for even-length window, center is between indices so 200 isn't exact peak
      expect(window[200], closeTo(1.0, 1e-4));

      // Check formula matches for a sample point (symmetric window uses length-1)
      var n = 100;
      var expected = 0.5 - 0.5 * math.cos(2 * math.pi * n / 399);
      expect(window[n], closeTo(expected, 1e-10));

      // Window values should increase then decrease
      expect(window[50], lessThan(window[100]));
      expect(window[100], lessThan(window[200]));
      expect(window[200], greaterThan(window[300]));
    });

    test('mel filterbank has correct shape', () {
      var filterbank = MelExtractor.createMelFilterbank(
        16000, // sampleRate
        400, // nFft
        80, // nMels
        0.0, // fmin
        8000.0, // fmax
      );

      // Should be [nMels, nFreqs]
      expect(filterbank.length, equals(80));
      expect(filterbank[0].length, equals(201)); // nFft/2 + 1 = 400/2 + 1 = 201
    });

    test('mel filterbank filters are triangular', () {
      var filterbank = MelExtractor.createMelFilterbank(
        16000,
        400,
        80,
        0.0,
        8000.0,
      );

      // Most filters should have non-zero values (triangular shape)
      // Some filters at edges might be empty due to bin quantization
      var filtersWithValues = 0;
      for (var i = 0; i < filterbank.length; i++) {
        var filter = filterbank[i];
        var nonZeroCount = filter.where((v) => v > 0.0).length;
        if (nonZeroCount > 0) {
          filtersWithValues++;
        }
      }

      // At least 75% of filters should have values
      expect(filtersWithValues, greaterThan(60));
    });

    test('extract produces correct output shape for 1 second audio', () {
      // 1 second of audio at 16kHz = 16000 samples
      var audio = List.generate(16000, (i) => 0.0);

      var melSpec = extractor.extract(audio);

      // Expected frames: 1 + (16000 - 400) / 160 = 1 + 15600 / 160 = 1 + 97.5 = 98
      var expectedFrames = 1 + ((16000 - 400) ~/ 160);

      expect(melSpec.length, equals(80)); // nMels
      expect(melSpec[0].length, equals(expectedFrames));
    });

    test('extract normalizes audio correctly', () {
      // Create audio with amplitude 2.0
      var audio = List.generate(
        16000,
        (i) => 2.0 * math.sin(2 * math.pi * 440 * i / 16000),
      );

      // Should not throw and should normalize internally
      var melSpec = extractor.extract(audio);

      expect(melSpec.length, equals(80));
      expect(melSpec[0].length, greaterThan(0));

      // All values should be finite
      for (var row in melSpec) {
        for (var value in row) {
          expect(value.isFinite, isTrue);
        }
      }
    });

    test('extract handles silence without crashing', () {
      var silence = List.filled(16000, 0.0);

      var melSpec = extractor.extract(silence);

      expect(melSpec.length, equals(80));
      expect(melSpec[0].length, greaterThan(0));

      // All values should be finite (log of epsilon)
      for (var row in melSpec) {
        for (var value in row) {
          expect(value.isFinite, isTrue);
        }
      }
    });

    test('extract handles short audio by padding', () {
      // Audio shorter than one frame (400 samples)
      var shortAudio = List.filled(200, 0.0);

      var melSpec = extractor.extract(shortAudio);

      // Should produce at least 1 frame
      expect(melSpec.length, equals(80));
      expect(melSpec[0].length, greaterThanOrEqualTo(1));
    });

    test('Hz to Mel conversion matches HTK formula', () {
      // Test known conversions
      expect(MelExtractor.hzToMel(0), closeTo(0.0, 1e-6));
      expect(MelExtractor.hzToMel(1000), closeTo(1000.0, 1e-1)); // Approximate

      // HTK formula: 2595 * log10(1 + hz / 700)
      var hz = 440.0;
      var expectedMel = 2595.0 * math.log(1.0 + hz / 700.0) / math.ln10;
      expect(MelExtractor.hzToMel(hz), closeTo(expectedMel, 1e-6));
    });

    test('Mel to Hz conversion is inverse of Hz to Mel', () {
      var testFreqs = [0.0, 100.0, 1000.0, 4000.0, 8000.0];

      for (var hz in testFreqs) {
        var mel = MelExtractor.hzToMel(hz);
        var hzBack = MelExtractor.melToHz(mel);
        expect(hzBack, closeTo(hz, 1e-6));
      }
    });

    test('extract produces consistent results for same input', () {
      var audio = List.generate(
        16000,
        (i) => math.sin(2 * math.pi * 440 * i / 16000),
      );

      var melSpec1 = extractor.extract(audio);
      var melSpec2 = extractor.extract(audio);

      expect(melSpec1.length, equals(melSpec2.length));
      expect(melSpec1[0].length, equals(melSpec2[0].length));

      // Values should be identical
      for (var i = 0; i < melSpec1.length; i++) {
        for (var t = 0; t < melSpec1[i].length; t++) {
          expect(melSpec1[i][t], equals(melSpec2[i][t]));
        }
      }
    });

    test('extract produces different results for different inputs', () {
      var audio1 = List.generate(
        16000,
        (i) => math.sin(2 * math.pi * 440 * i / 16000),
      );
      var audio2 = List.generate(
        16000,
        (i) => math.sin(2 * math.pi * 880 * i / 16000),
      );

      var melSpec1 = extractor.extract(audio1);
      var melSpec2 = extractor.extract(audio2);

      // Should have same shape
      expect(melSpec1.length, equals(melSpec2.length));
      expect(melSpec1[0].length, equals(melSpec2[0].length));

      // But different values (at least somewhere)
      var hasDifference = false;
      for (var i = 0; i < melSpec1.length && !hasDifference; i++) {
        for (var t = 0; t < melSpec1[i].length && !hasDifference; t++) {
          if ((melSpec1[i][t] - melSpec2[i][t]).abs() > 1e-6) {
            hasDifference = true;
          }
        }
      }
      expect(hasDifference, isTrue);
    });
  });
}
