"""Tests for audio augmentation functions (pitch shift, formant shift)."""

import numpy as np
import pytest
from numpy.typing import NDArray

from mandarin_grader.data.augmentation import (
    pitch_shift,
    formant_shift,
    random_pitch_shift,
    random_formant_shift,
)


def generate_sine_wave(
    freq_hz: float = 440.0,
    duration_s: float = 0.5,
    sr: int = 16000,
) -> NDArray[np.float32]:
    """Generate a pure sine wave for testing."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (np.sin(2 * np.pi * freq_hz * t) * 0.5).astype(np.float32)


def estimate_fundamental_freq(
    audio: NDArray[np.float32],
    sr: int = 16000,
    fmin: float = 50.0,
    fmax: float = 2000.0,
) -> float:
    """Estimate fundamental frequency using autocorrelation."""
    # Simple autocorrelation-based pitch estimation
    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)

    # Use a window of audio
    window = audio[:min(len(audio), sr // 2)]  # 0.5s max

    # Compute autocorrelation
    autocorr = np.correlate(window, window, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]  # Keep positive lags

    # Find peak in valid lag range
    search_start = min_lag
    search_end = min(max_lag, len(autocorr) - 1)

    if search_end <= search_start:
        return 0.0

    peak_lag = search_start + np.argmax(autocorr[search_start:search_end])
    return sr / peak_lag


class TestPitchShift:
    """Tests for pitch_shift function."""

    def test_no_shift_returns_same_audio(self):
        """Pitch shift of 0 semitones should return unchanged audio."""
        audio = generate_sine_wave(440.0)
        result = pitch_shift(audio, semitones=0.0)

        assert len(result) == len(audio)
        np.testing.assert_array_almost_equal(result, audio, decimal=5)

    def test_small_shift_returns_unchanged(self):
        """Very small shifts (<0.01 semitones) should skip processing."""
        audio = generate_sine_wave(440.0)
        result = pitch_shift(audio, semitones=0.005)

        assert len(result) == len(audio)
        np.testing.assert_array_almost_equal(result, audio, decimal=5)

    def test_output_same_length_as_input(self):
        """Output should have same length as input."""
        audio = generate_sine_wave(440.0, duration_s=1.0)

        for semitones in [-3.0, -1.0, 1.0, 3.0]:
            result = pitch_shift(audio, semitones)
            assert len(result) == len(audio), f"Length mismatch for {semitones} semitones"

    def test_pitch_shift_up_increases_frequency(self):
        """Shifting pitch up should increase fundamental frequency."""
        audio = generate_sine_wave(440.0, duration_s=0.5)

        # Shift up by 12 semitones (one octave)
        result = pitch_shift(audio, semitones=12.0)

        orig_freq = estimate_fundamental_freq(audio)
        shifted_freq = estimate_fundamental_freq(result)

        # Should roughly double (allow 20% tolerance for estimation error)
        assert shifted_freq > orig_freq * 1.5, f"Expected freq > {orig_freq * 1.5}, got {shifted_freq}"

    def test_pitch_shift_down_decreases_frequency(self):
        """Shifting pitch down should decrease fundamental frequency."""
        audio = generate_sine_wave(880.0, duration_s=0.5)  # Higher freq for down shift

        # Shift down by 12 semitones (one octave)
        result = pitch_shift(audio, semitones=-12.0)

        orig_freq = estimate_fundamental_freq(audio)
        shifted_freq = estimate_fundamental_freq(result)

        # Should roughly halve (allow tolerance for estimation error)
        assert shifted_freq < orig_freq * 0.7, f"Expected freq < {orig_freq * 0.7}, got {shifted_freq}"

    def test_output_dtype_is_float32(self):
        """Output should be float32."""
        audio = generate_sine_wave(440.0)
        result = pitch_shift(audio, semitones=2.0)
        assert result.dtype == np.float32


class TestFormantShift:
    """Tests for formant_shift function."""

    def test_no_shift_returns_similar_audio(self):
        """Formant shift of 1.0 should return nearly unchanged audio."""
        audio = generate_sine_wave(440.0)
        result = formant_shift(audio, shift_ratio=1.0)

        assert len(result) == len(audio)
        # Small numerical differences expected due to resampling
        np.testing.assert_array_almost_equal(result, audio, decimal=3)

    def test_small_shift_returns_unchanged(self):
        """Very small shifts (< 1% from 1.0) should skip processing."""
        audio = generate_sine_wave(440.0)
        result = formant_shift(audio, shift_ratio=1.005)

        assert len(result) == len(audio)
        np.testing.assert_array_almost_equal(result, audio, decimal=3)

    def test_output_same_length_as_input(self):
        """Output should have same length as input."""
        audio = generate_sine_wave(440.0, duration_s=1.0)

        for ratio in [0.85, 0.95, 1.05, 1.15]:
            result = formant_shift(audio, ratio)
            assert len(result) == len(audio), f"Length mismatch for ratio {ratio}"

    def test_formant_shift_preserves_pitch_approximately(self):
        """Formant shift should approximately preserve fundamental frequency."""
        audio = generate_sine_wave(440.0, duration_s=0.5)

        # Shift formants by 15%
        result = formant_shift(audio, shift_ratio=1.15)

        orig_freq = estimate_fundamental_freq(audio)
        shifted_freq = estimate_fundamental_freq(result)

        # Pitch should be within 20% of original (formant shift may have minor effects)
        assert abs(shifted_freq - orig_freq) / orig_freq < 0.25, \
            f"Pitch changed too much: {orig_freq} -> {shifted_freq}"

    def test_output_dtype_is_float32(self):
        """Output should be float32."""
        audio = generate_sine_wave(440.0)
        result = formant_shift(audio, shift_ratio=1.1)
        assert result.dtype == np.float32


class TestRandomAugmentations:
    """Tests for random augmentation functions."""

    def test_random_pitch_shift_output_length(self):
        """Random pitch shift should preserve length."""
        audio = generate_sine_wave(440.0, duration_s=1.0)

        for _ in range(5):
            result = random_pitch_shift(audio, max_semitones=3.0)
            assert len(result) == len(audio)

    def test_random_pitch_shift_different_results(self):
        """Random pitch shift should produce different results each call."""
        np.random.seed(None)  # Ensure randomness
        audio = generate_sine_wave(440.0, duration_s=0.5)

        results = [random_pitch_shift(audio, max_semitones=3.0) for _ in range(3)]

        # At least two results should be different
        all_same = all(np.allclose(results[0], r, atol=1e-3) for r in results[1:])
        assert not all_same, "Expected different random results"

    def test_random_formant_shift_output_length(self):
        """Random formant shift should preserve length."""
        audio = generate_sine_wave(440.0, duration_s=1.0)

        for _ in range(5):
            result = random_formant_shift(audio, max_shift_percent=15.0)
            assert len(result) == len(audio)

    def test_random_formant_shift_different_results(self):
        """Random formant shift should produce different results each call."""
        np.random.seed(None)  # Ensure randomness
        audio = generate_sine_wave(440.0, duration_s=0.5)

        results = [random_formant_shift(audio, max_shift_percent=15.0) for _ in range(3)]

        # At least two results should be different
        all_same = all(np.allclose(results[0], r, atol=1e-3) for r in results[1:])
        assert not all_same, "Expected different random results"


class TestAugmentationWithRealAudio:
    """Tests with more realistic audio (multi-frequency)."""

    @pytest.fixture
    def complex_audio(self) -> NDArray[np.float32]:
        """Generate complex audio with harmonics (simulates voice)."""
        sr = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)

        # Fundamental + harmonics (simulates vocal sound)
        f0 = 200.0
        audio = (
            0.5 * np.sin(2 * np.pi * f0 * t) +
            0.3 * np.sin(2 * np.pi * 2 * f0 * t) +
            0.15 * np.sin(2 * np.pi * 3 * f0 * t) +
            0.1 * np.sin(2 * np.pi * 4 * f0 * t)
        )
        return audio.astype(np.float32)

    def test_pitch_shift_complex_audio(self, complex_audio):
        """Pitch shift should work on complex audio."""
        result = pitch_shift(complex_audio, semitones=2.0)

        assert len(result) == len(complex_audio)
        assert result.dtype == np.float32
        assert not np.allclose(result, complex_audio, atol=0.01)

    def test_formant_shift_complex_audio(self, complex_audio):
        """Formant shift should work on complex audio."""
        result = formant_shift(complex_audio, shift_ratio=1.1)

        assert len(result) == len(complex_audio)
        assert result.dtype == np.float32

    def test_combined_augmentation(self, complex_audio):
        """Both augmentations can be applied sequentially."""
        # Apply pitch shift then formant shift
        step1 = pitch_shift(complex_audio, semitones=1.5)
        step2 = formant_shift(step1, shift_ratio=1.1)

        assert len(step2) == len(complex_audio)
        assert step2.dtype == np.float32


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_short_audio(self):
        """Should handle very short audio."""
        audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)

        # Should not crash
        result = pitch_shift(audio, semitones=1.0)
        assert result.dtype == np.float32

    def test_silent_audio(self):
        """Should handle silent audio."""
        audio = np.zeros(8000, dtype=np.float32)

        result = pitch_shift(audio, semitones=2.0)
        assert len(result) == len(audio)
        # Silent audio should remain approximately silent
        assert np.max(np.abs(result)) < 0.01

    def test_extreme_pitch_shift(self):
        """Should handle extreme pitch shifts without crashing."""
        audio = generate_sine_wave(440.0)

        # Extreme shifts - should not crash
        result_up = pitch_shift(audio, semitones=12.0)  # 1 octave up
        result_down = pitch_shift(audio, semitones=-12.0)  # 1 octave down

        assert len(result_up) == len(audio)
        assert len(result_down) == len(audio)

    def test_extreme_formant_shift(self):
        """Should handle extreme formant shifts without crashing."""
        audio = generate_sine_wave(440.0)

        # Extreme shifts - should not crash
        result_up = formant_shift(audio, shift_ratio=1.3)  # +30%
        result_down = formant_shift(audio, shift_ratio=0.7)  # -30%

        assert len(result_up) == len(audio)
        assert len(result_down) == len(audio)
