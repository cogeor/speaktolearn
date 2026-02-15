import 'dart:math' as math;
import 'package:fftea/fftea.dart';

/// Extracts mel-scale log spectrogram features from audio.
///
/// This implementation matches the Python reference in syllable_predictor_v4.py
/// to ensure numerical consistency for ML inference.
class MelExtractor {
  final int nMels;
  final int sampleRate;
  final int hopLength;
  final int winLength;
  final int nFft;
  final double fmin;
  final double fmax;

  late final List<double> _hanningWindow;
  late final List<List<double>> _melFilterbank;

  MelExtractor({
    this.nMels = 80,
    this.sampleRate = 16000,
    this.hopLength = 160,
    this.winLength = 400,
    int? nFft,
    this.fmin = 0.0,
    double? fmax,
  })  : nFft = nFft ?? winLength,
        fmax = fmax ?? sampleRate / 2.0 {
    _hanningWindow = createHanningWindow(this.nFft);
    _melFilterbank = createMelFilterbank(
      sampleRate,
      this.nFft,
      nMels,
      fmin,
      this.fmax,
    );
  }

  /// Extract mel spectrogram from audio samples.
  ///
  /// Args:
  ///   audio: Audio samples as List<double>
  ///
  /// Returns:
  ///   Mel spectrogram as List<List<double>> with shape [nMels, timeFrames]
  List<List<double>> extract(List<double> audio) {
    // Normalize audio amplitude to [-1, 1] for consistent mel features
    var normalizedAudio = _normalizeAudio(audio);

    // Compute STFT (Short-Time Fourier Transform)
    var powerSpec = _computeStft(normalizedAudio);

    // Apply mel filterbank
    var melSpec = _applyMelFilterbank(powerSpec);

    // Apply log with epsilon
    var logMelSpec = _applyLog(melSpec, epsilon: 1e-9);

    return logMelSpec;
  }

  /// Normalize audio to [-1, 1] range.
  List<double> _normalizeAudio(List<double> audio) {
    var maxAbs = 0.0;
    for (var sample in audio) {
      var abs = sample.abs();
      if (abs > maxAbs) {
        maxAbs = abs;
      }
    }

    if (maxAbs > 1e-6) {
      return audio.map((s) => s / maxAbs).toList();
    }
    return List.from(audio);
  }

  /// Compute STFT and return power spectrogram.
  ///
  /// Returns: List<List<double>> with shape [nFft/2 + 1, nFrames]
  List<List<double>> _computeStft(List<double> audio) {
    var audioData = List<double>.from(audio);

    // Calculate number of frames
    var nFrames = 1 + ((audioData.length - nFft) ~/ hopLength);

    // Handle short audio by padding
    if (nFrames < 1) {
      var padLength = nFft - audioData.length + hopLength;
      audioData.addAll(List.filled(padLength, 0.0));
      nFrames = 1;
    }

    var nFreqs = nFft ~/ 2 + 1;
    var spec = List.generate(
      nFreqs,
      (_) => List<double>.filled(nFrames, 0.0),
    );

    // Create FFT instance for real FFT
    var fft = FFT(nFft);

    for (var i = 0; i < nFrames; i++) {
      var start = i * hopLength;
      var end = start + nFft;

      // Extract frame
      var frame = <double>[];
      for (var j = start; j < end; j++) {
        if (j < audioData.length) {
          frame.add(audioData[j]);
        } else {
          frame.add(0.0);
        }
      }

      // Apply Hanning window
      var windowed = List<double>.generate(
        nFft,
        (j) => frame[j] * _hanningWindow[j],
      );

      // Perform real FFT
      var fftResult = fft.realFft(windowed);

      // Compute power spectrum: |FFT|^2
      // realFft returns first nFft/2 + 1 complex values
      // Float64x2 uses .x for real and .y for imaginary
      for (var j = 0; j < nFreqs; j++) {
        var real = fftResult[j].x;
        var imag = fftResult[j].y;
        spec[j][i] = real * real + imag * imag;
      }
    }

    return spec;
  }

  /// Apply mel filterbank to power spectrogram.
  ///
  /// Args:
  ///   powerSpec: Power spectrogram [nFreqs, nFrames]
  ///
  /// Returns:
  ///   Mel spectrogram [nMels, nFrames]
  List<List<double>> _applyMelFilterbank(List<List<double>> powerSpec) {
    var nFreqs = powerSpec.length;
    var nFrames = powerSpec[0].length;

    var melSpec = List.generate(
      nMels,
      (_) => List<double>.filled(nFrames, 0.0),
    );

    // Matrix multiplication: melFilterbank @ powerSpec
    for (var i = 0; i < nMels; i++) {
      for (var t = 0; t < nFrames; t++) {
        var sum = 0.0;
        for (var j = 0; j < nFreqs; j++) {
          sum += _melFilterbank[i][j] * powerSpec[j][t];
        }
        melSpec[i][t] = sum;
      }
    }

    return melSpec;
  }

  /// Apply logarithm with epsilon to prevent log(0).
  List<List<double>> _applyLog(
    List<List<double>> melSpec, {
    required double epsilon,
  }) {
    return melSpec.map((row) {
      return row.map((value) => math.log(value + epsilon)).toList();
    }).toList();
  }

  /// Create Hanning window of given length.
  static List<double> createHanningWindow(int length) {
    return List.generate(length, (n) {
      return 0.5 - 0.5 * math.cos(2.0 * math.pi * n / length);
    });
  }

  /// Create mel filterbank matrix.
  ///
  /// Returns: List<List<double>> with shape [nMels, nFreqs]
  static List<List<double>> createMelFilterbank(
    int sampleRate,
    int nFft,
    int nMels,
    double fmin,
    double fmax,
  ) {
    var nFreqs = nFft ~/ 2 + 1;

    // Convert Hz to mel and back
    var melMin = hzToMel(fmin);
    var melMax = hzToMel(fmax);

    // Create mel points linearly spaced in mel scale
    var melPoints = List.generate(
      nMels + 2,
      (i) => melMin + (melMax - melMin) * i / (nMels + 1),
    );

    // Convert back to Hz
    var hzPoints = melPoints.map(melToHz).toList();

    // Convert Hz to FFT bin numbers
    var binPoints = hzPoints.map((hz) {
      return ((nFft + 1) * hz / sampleRate).floor();
    }).toList();

    // Create filterbank
    var filterbank = List.generate(
      nMels,
      (_) => List<double>.filled(nFreqs, 0.0),
    );

    for (var i = 0; i < nMels; i++) {
      var left = binPoints[i];
      var center = binPoints[i + 1];
      var right = binPoints[i + 2];

      // Rising slope
      for (var j = left; j < center; j++) {
        if (center != left) {
          filterbank[i][j] = (j - left) / (center - left);
        }
      }

      // Falling slope
      for (var j = center; j < right; j++) {
        if (right != center) {
          filterbank[i][j] = (right - j) / (right - center);
        }
      }
    }

    return filterbank;
  }

  /// Convert frequency in Hz to mel scale (HTK formula).
  static double hzToMel(double hz) {
    return 2595.0 * math.log(1.0 + hz / 700.0) / math.ln10;
  }

  /// Convert mel scale to frequency in Hz (HTK formula).
  static double melToHz(double mel) {
    return 700.0 * (math.pow(10.0, mel / 2595.0) - 1.0);
  }
}
