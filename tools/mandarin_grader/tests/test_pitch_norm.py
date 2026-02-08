"""Tests for pitch normalization module.

Tests the core invariants of speaker-independent F0 normalization:
- Hz to semitones conversion handles edge cases
- Robust statistics are computed correctly
- Normalized contours are invariant to speaker pitch range
- pYIN pitch extraction
"""

import numpy as np
import pytest

from mandarin_grader.pitch import (
    extract_f0_pyin,
    hz_to_semitones,
    normalize_f0,
    normalize_frame_track,
    robust_stats,
)
from mandarin_grader.types import FrameTrack


class TestHzToSemitones:
    """Tests for Hz to semitones conversion."""

    def test_reference_is_zero(self) -> None:
        """100 Hz with ref 100 Hz = 0 semitones."""
        f0 = np.array([100.0])
        result = hz_to_semitones(f0, ref_hz=100.0)
        assert np.isclose(result[0], 0.0)

    def test_octave_is_twelve(self) -> None:
        """200 Hz with ref 100 Hz = 12 semitones (one octave)."""
        f0 = np.array([200.0])
        result = hz_to_semitones(f0, ref_hz=100.0)
        assert np.isclose(result[0], 12.0)

    def test_half_octave_is_negative_twelve(self) -> None:
        """50 Hz with ref 100 Hz = -12 semitones."""
        f0 = np.array([50.0])
        result = hz_to_semitones(f0, ref_hz=100.0)
        assert np.isclose(result[0], -12.0)

    def test_unvoiced_is_zero(self) -> None:
        """f0 = 0 should return 0 semitones."""
        f0 = np.array([0.0, 100.0, 0.0])
        result = hz_to_semitones(f0, ref_hz=100.0)
        assert result[0] == 0.0
        assert result[2] == 0.0
        assert np.isclose(result[1], 0.0)

    def test_negative_is_zero(self) -> None:
        """Negative f0 should return 0 semitones."""
        f0 = np.array([-50.0, -100.0])
        result = hz_to_semitones(f0, ref_hz=100.0)
        assert result[0] == 0.0
        assert result[1] == 0.0

    def test_array_input(self) -> None:
        """Works with numpy arrays of various sizes."""
        f0 = np.array([100.0, 200.0, 400.0, 0.0])
        result = hz_to_semitones(f0, ref_hz=100.0)
        expected = np.array([0.0, 12.0, 24.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_default_reference(self) -> None:
        """Default reference is 100 Hz."""
        f0 = np.array([100.0])
        result = hz_to_semitones(f0)
        assert np.isclose(result[0], 0.0)


class TestRobustStats:
    """Tests for robust statistics computation."""

    def test_fully_voiced(self) -> None:
        """All voiced frames contribute to stats."""
        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        voicing = np.ones(5)
        median, mad = robust_stats(values, voicing)
        assert np.isclose(median, 30.0)  # Median of 10,20,30,40,50
        # MAD = median of |10-30|, |20-30|, |30-30|, |40-30|, |50-30|
        #     = median of 20, 10, 0, 10, 20 = 10
        assert np.isclose(mad, 10.0)

    def test_mixed_voicing(self) -> None:
        """Only voiced frames above threshold count."""
        values = np.array([100.0, 10.0, 20.0, 30.0, 200.0])
        voicing = np.array([0.3, 0.8, 0.9, 0.7, 0.2])  # Only indices 1,2,3 pass
        median, mad = robust_stats(values, voicing, voicing_threshold=0.5)
        # Voiced values: 10, 20, 30 -> median = 20
        assert np.isclose(median, 20.0)
        # MAD = median of |10-20|, |20-20|, |30-20| = median of 10, 0, 10 = 10
        assert np.isclose(mad, 10.0)

    def test_all_unvoiced(self) -> None:
        """Returns (0.0, 1.0) when no voiced frames."""
        values = np.array([10.0, 20.0, 30.0])
        voicing = np.zeros(3)
        median, mad = robust_stats(values, voicing)
        assert median == 0.0
        assert mad == 1.0

    def test_mad_calculation(self) -> None:
        """MAD is median absolute deviation."""
        # Carefully constructed example
        values = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # 100 is outlier
        voicing = np.ones(5)
        median, mad = robust_stats(values, voicing)
        # Median should be 3.0
        assert np.isclose(median, 3.0)
        # MAD = median of |1-3|, |2-3|, |3-3|, |4-3|, |100-3|
        #     = median of 2, 1, 0, 1, 97 = 1
        assert np.isclose(mad, 1.0)

    def test_constant_values_returns_unit_mad(self) -> None:
        """Constant values produce MAD of 1.0 to avoid division by zero."""
        values = np.array([5.0, 5.0, 5.0, 5.0])
        voicing = np.ones(4)
        median, mad = robust_stats(values, voicing)
        assert np.isclose(median, 5.0)
        assert mad == 1.0  # Avoid zero MAD

    def test_voicing_threshold_boundary(self) -> None:
        """Frames exactly at threshold are included."""
        values = np.array([10.0, 20.0, 30.0])
        voicing = np.array([0.5, 0.5, 0.5])
        median, mad = robust_stats(values, voicing, voicing_threshold=0.5)
        assert np.isclose(median, 20.0)


class TestNormalizeF0:
    """Tests for F0 normalization."""

    def test_scaling_invariance(self) -> None:
        """
        Key invariant: multiplying all F0 by constant
        should not change normalized contour shape.
        """
        f0_1 = np.array([100.0, 150.0, 200.0, 250.0, 200.0])
        f0_2 = f0_1 * 2  # One octave higher
        voicing = np.ones(5)

        # Convert to semitones and normalize
        st1 = hz_to_semitones(f0_1, 100.0)
        st2 = hz_to_semitones(f0_2, 100.0)

        norm1 = normalize_f0(st1, voicing)
        norm2 = normalize_f0(st2, voicing)

        # Shapes should be identical (speaker-independent)
        np.testing.assert_allclose(norm1, norm2, atol=0.01)

    def test_unvoiced_frames_excluded(self) -> None:
        """Unvoiced frames don't affect stats and don't produce NaN."""
        semitones = np.array([0.0, 12.0, 0.0, 24.0, 0.0])
        voicing = np.array([0.0, 1.0, 0.0, 1.0, 0.0])

        result = normalize_f0(semitones, voicing)

        # Unvoiced frames should be 0
        assert result[0] == 0.0
        assert result[2] == 0.0
        assert result[4] == 0.0

        # No NaN values
        assert not np.any(np.isnan(result))

    def test_all_unvoiced_returns_zeros(self) -> None:
        """All unvoiced input returns zero array."""
        semitones = np.array([12.0, 24.0, 36.0])
        voicing = np.zeros(3)

        result = normalize_f0(semitones, voicing)

        np.testing.assert_array_equal(result, np.zeros(3))

    def test_output_centered(self) -> None:
        """Output has mean near zero for voiced frames."""
        # Varying pitch contour
        semitones = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
        voicing = np.ones(5)

        result = normalize_f0(semitones, voicing)

        # Median of result should be near 0 (not mean, since we use robust stats)
        voiced_result = result[voicing >= 0.5]
        assert np.abs(np.median(voiced_result)) < 0.1

    def test_preserves_relative_shape(self) -> None:
        """Normalization preserves relative pitch movements."""
        # Rising contour
        semitones = np.array([10.0, 15.0, 20.0])
        voicing = np.ones(3)

        result = normalize_f0(semitones, voicing)

        # Should still be rising
        assert result[0] < result[1] < result[2]

    def test_different_speakers_same_utterance(self) -> None:
        """Different speakers saying same thing get similar normalized contours."""
        # Speaker 1: lower voice
        f0_speaker1 = np.array([100.0, 120.0, 140.0, 130.0, 110.0])
        # Speaker 2: higher voice (same relative pattern, shifted up)
        f0_speaker2 = np.array([200.0, 240.0, 280.0, 260.0, 220.0])

        voicing = np.ones(5)

        st1 = hz_to_semitones(f0_speaker1, 100.0)
        st2 = hz_to_semitones(f0_speaker2, 100.0)

        norm1 = normalize_f0(st1, voicing)
        norm2 = normalize_f0(st2, voicing)

        # Normalized contours should be identical
        np.testing.assert_allclose(norm1, norm2, atol=0.01)


class TestNormalizeFrameTrack:
    """Tests for FrameTrack normalization convenience function."""

    def test_with_frame_track(self) -> None:
        """Works with FrameTrack dataclass."""
        track = FrameTrack(
            frame_hz=100.0,
            f0_hz=np.array([100.0, 150.0, 200.0, 0.0, 100.0]),
            voicing=np.array([1.0, 1.0, 1.0, 0.0, 1.0]),
        )

        result = normalize_frame_track(track)

        # Should return array of same length
        assert len(result) == 5

        # Unvoiced frame should be 0
        assert result[3] == 0.0

        # No NaN values
        assert not np.any(np.isnan(result))

    def test_empty_track(self) -> None:
        """Handles empty track gracefully."""
        track = FrameTrack(
            frame_hz=100.0,
            f0_hz=np.array([]),
            voicing=np.array([]),
        )

        result = normalize_frame_track(track)

        assert len(result) == 0

    def test_all_unvoiced_track(self) -> None:
        """Handles track with no voiced frames."""
        track = FrameTrack(
            frame_hz=100.0,
            f0_hz=np.array([0.0, 0.0, 0.0]),
            voicing=np.array([0.0, 0.0, 0.0]),
        )

        result = normalize_frame_track(track)

        np.testing.assert_array_equal(result, np.zeros(3))

    def test_with_energy_field(self) -> None:
        """Works when energy field is present."""
        track = FrameTrack(
            frame_hz=100.0,
            f0_hz=np.array([100.0, 200.0, 150.0]),
            voicing=np.array([1.0, 1.0, 1.0]),
            energy=np.array([0.5, 0.8, 0.6]),
        )

        result = normalize_frame_track(track)

        # Should still work; energy is not used in pitch normalization
        assert len(result) == 3
        assert not np.any(np.isnan(result))

    def test_consistency_with_manual_pipeline(self) -> None:
        """normalize_frame_track matches manual hz->semitones->normalize pipeline."""
        track = FrameTrack(
            frame_hz=100.0,
            f0_hz=np.array([100.0, 150.0, 200.0, 175.0]),
            voicing=np.array([1.0, 1.0, 1.0, 1.0]),
        )

        # Using convenience function
        result_convenience = normalize_frame_track(track)

        # Manual pipeline
        semitones = hz_to_semitones(track.f0_hz)
        result_manual = normalize_f0(semitones, track.voicing)

        np.testing.assert_array_equal(result_convenience, result_manual)


# Check if librosa is available for pYIN tests
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


@pytest.mark.skipif(not HAS_LIBROSA, reason="librosa not installed")
class TestExtractF0Pyin:
    """Tests for pYIN-based F0 extraction."""

    def test_returns_correct_shapes(self) -> None:
        """F0 and voicing arrays have correct shapes."""
        # Generate 1 second of audio at 16kHz
        sr = 16000
        duration = 1.0
        audio = np.zeros(int(sr * duration), dtype=np.float64)

        f0, voicing = extract_f0_pyin(audio, sr=sr)

        # With hop_length=160, we expect ~100 frames per second
        expected_frames = int(sr * duration / 160)
        # Allow some tolerance for frame boundary handling
        assert abs(len(f0) - expected_frames) <= 2
        assert len(f0) == len(voicing)

    def test_silent_audio_is_unvoiced(self) -> None:
        """Silent audio should have low voicing probability."""
        sr = 16000
        audio = np.zeros(sr, dtype=np.float64)  # 1 second of silence

        f0, voicing = extract_f0_pyin(audio, sr=sr)

        # All frames should be unvoiced (voicing prob low)
        assert np.mean(voicing) < 0.5

    def test_sine_wave_is_voiced(self) -> None:
        """Pure tone should be detected as voiced with correct F0."""
        sr = 16000
        freq = 200.0  # 200 Hz sine wave
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float64)
        audio = 0.5 * np.sin(2 * np.pi * freq * t)

        f0, voicing = extract_f0_pyin(audio, sr=sr)

        # Most frames should be voiced
        voiced_mask = voicing > 0.5
        assert np.sum(voiced_mask) > len(f0) * 0.5  # At least half voiced

        # Voiced frames should have F0 near 200 Hz
        voiced_f0 = f0[voiced_mask]
        if len(voiced_f0) > 0:
            assert np.abs(np.median(voiced_f0) - freq) < 20  # Within 20 Hz

    def test_f0_no_nan_values(self) -> None:
        """F0 array should not contain NaN values."""
        sr = 16000
        audio = np.random.randn(sr).astype(np.float64) * 0.1

        f0, voicing = extract_f0_pyin(audio, sr=sr)

        assert not np.any(np.isnan(f0))
        assert not np.any(np.isnan(voicing))

    def test_custom_frequency_range(self) -> None:
        """Custom fmin/fmax parameters are respected."""
        sr = 16000
        freq = 100.0  # Low frequency
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float64)
        audio = 0.5 * np.sin(2 * np.pi * freq * t)

        # Use fmin=80, fmax=150 to include 100 Hz
        f0, voicing = extract_f0_pyin(audio, sr=sr, fmin=80.0, fmax=150.0)

        # Should detect the tone
        voiced_f0 = f0[voicing > 0.5]
        if len(voiced_f0) > 0:
            assert np.abs(np.median(voiced_f0) - freq) < 20

    def test_custom_hop_length(self) -> None:
        """Custom hop_length affects number of output frames."""
        sr = 16000
        duration = 1.0
        audio = np.zeros(int(sr * duration), dtype=np.float64)

        # Default hop_length=160 -> ~100 frames/sec
        f0_default, _ = extract_f0_pyin(audio, sr=sr, hop_length=160)

        # Double hop_length -> half the frames
        f0_double, _ = extract_f0_pyin(audio, sr=sr, hop_length=320)

        assert abs(len(f0_default) - 2 * len(f0_double)) <= 2
