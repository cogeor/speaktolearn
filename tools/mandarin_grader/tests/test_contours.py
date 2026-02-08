"""Tests for contour extraction module.

Tests the core functionality of syllable-level pitch contour extraction:
- Time conversion from milliseconds to frame indices
- Voicing ratio computation
- Contour resampling to fixed K points
- Derivative computation for shape analysis
- Full contour extraction pipeline
"""

import numpy as np
import pytest

from mandarin_grader.contour import (
    compute_derivatives,
    compute_voicing_ratio,
    extract_contour,
    ms_to_frame,
    resample_contour,
)
from mandarin_grader.types import FrameTrack, Ms, SyllableSpan


class TestMsToFrame:
    """Tests for millisecond to frame index conversion."""

    def test_basic_conversion(self) -> None:
        """100ms at 100Hz = frame 10."""
        result = ms_to_frame(100, 100.0)
        assert result == 10

    def test_zero_ms(self) -> None:
        """0 ms = frame 0."""
        result = ms_to_frame(0, 100.0)
        assert result == 0

    def test_standard_10ms_frames(self) -> None:
        """Standard 100Hz frame rate (10ms per frame)."""
        # 250ms at 100Hz = 25 frames
        result = ms_to_frame(250, 100.0)
        assert result == 25

    def test_different_frame_rates(self) -> None:
        """Works with different frame rates."""
        # 200ms at 50Hz = 10 frames
        assert ms_to_frame(200, 50.0) == 10
        # 100ms at 200Hz = 20 frames
        assert ms_to_frame(100, 200.0) == 20

    def test_fractional_result_truncated(self) -> None:
        """Fractional frame indices are truncated (not rounded)."""
        # 15ms at 100Hz = 1.5 -> 1 (truncated)
        result = ms_to_frame(15, 100.0)
        assert result == 1


class TestComputeVoicingRatio:
    """Tests for voicing ratio computation."""

    def test_all_voiced(self) -> None:
        """All voiced frames = 1.0."""
        voicing = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        result = compute_voicing_ratio(voicing)
        assert result == 1.0

    def test_all_unvoiced(self) -> None:
        """All unvoiced frames = 0.0."""
        voicing = np.array([0.0, 0.0, 0.0, 0.0])
        result = compute_voicing_ratio(voicing)
        assert result == 0.0

    def test_half_voiced(self) -> None:
        """Half voiced frames = 0.5."""
        voicing = np.array([1.0, 1.0, 0.0, 0.0])
        result = compute_voicing_ratio(voicing)
        assert result == 0.5

    def test_empty_array(self) -> None:
        """Empty array returns 0.0."""
        voicing = np.array([])
        result = compute_voicing_ratio(voicing)
        assert result == 0.0

    def test_threshold_boundary(self) -> None:
        """Frames exactly at threshold are counted as voiced."""
        voicing = np.array([0.5, 0.5, 0.5])
        result = compute_voicing_ratio(voicing, voicing_threshold=0.5)
        assert result == 1.0

    def test_below_threshold(self) -> None:
        """Frames below threshold are not counted."""
        voicing = np.array([0.4, 0.4, 0.4])
        result = compute_voicing_ratio(voicing, voicing_threshold=0.5)
        assert result == 0.0

    def test_mixed_voicing(self) -> None:
        """Mixed voicing levels above/below threshold."""
        voicing = np.array([0.3, 0.6, 0.4, 0.8, 0.5])
        # Values >= 0.5: 0.6, 0.8, 0.5 = 3 out of 5
        result = compute_voicing_ratio(voicing, voicing_threshold=0.5)
        assert np.isclose(result, 3 / 5)


class TestResampleContour:
    """Tests for contour resampling to K fixed points."""

    def test_output_shape(self) -> None:
        """Output has exactly K points."""
        f0_norm = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        voicing = np.ones(5)

        result = resample_contour(f0_norm, voicing, k=20)
        assert result.shape == (20,)

        result_10 = resample_contour(f0_norm, voicing, k=10)
        assert result_10.shape == (10,)

    def test_linear_interpolation(self) -> None:
        """Interpolation is linear between voiced points."""
        # Linear input should produce linear output
        f0_norm = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        voicing = np.ones(5)

        result = resample_contour(f0_norm, voicing, k=5)

        # Output should be linear from 0 to 4
        expected = np.linspace(0.0, 4.0, 5)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_sparse_voicing_returns_zeros(self) -> None:
        """Fewer than 3 voiced frames returns zeros."""
        f0_norm = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Only 2 voiced frames
        voicing_2 = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
        result_2 = resample_contour(f0_norm, voicing_2, k=10)
        np.testing.assert_array_equal(result_2, np.zeros(10))

        # Only 1 voiced frame
        voicing_1 = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
        result_1 = resample_contour(f0_norm, voicing_1, k=10)
        np.testing.assert_array_equal(result_1, np.zeros(10))

        # No voiced frames
        voicing_0 = np.zeros(5)
        result_0 = resample_contour(f0_norm, voicing_0, k=10)
        np.testing.assert_array_equal(result_0, np.zeros(10))

    def test_preserves_endpoints(self) -> None:
        """First and last voiced points are preserved in output."""
        f0_norm = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        voicing = np.ones(5)

        result = resample_contour(f0_norm, voicing, k=20)

        # First and last values should be preserved
        assert np.isclose(result[0], 1.0)
        assert np.isclose(result[-1], 5.0)

    def test_skips_unvoiced_frames(self) -> None:
        """Unvoiced frames are excluded from interpolation."""
        # Values with unvoiced gap in middle
        f0_norm = np.array([1.0, 999.0, 999.0, 2.0, 3.0])
        voicing = np.array([1.0, 0.0, 0.0, 1.0, 1.0])

        result = resample_contour(f0_norm, voicing, k=3)

        # Should interpolate only voiced values [1.0, 2.0, 3.0]
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_default_k_is_20(self) -> None:
        """Default K value is 20."""
        f0_norm = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        voicing = np.ones(5)

        result = resample_contour(f0_norm, voicing)
        assert result.shape == (20,)


class TestComputeDerivatives:
    """Tests for contour derivative computation."""

    def test_constant_has_zero_derivative(self) -> None:
        """Constant contour has df0 = 0 everywhere."""
        f0_norm = np.array([5.0, 5.0, 5.0, 5.0, 5.0])

        df0, ddf0 = compute_derivatives(f0_norm)

        np.testing.assert_allclose(df0, np.zeros(5), atol=1e-10)
        np.testing.assert_allclose(ddf0, np.zeros(5), atol=1e-10)

    def test_linear_has_constant_derivative(self) -> None:
        """Linear contour has constant first derivative."""
        # Linear contour: 0, 1, 2, 3, 4
        f0_norm = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        df0, ddf0 = compute_derivatives(f0_norm)

        # First derivative should be constant 1.0
        np.testing.assert_allclose(df0, np.ones(5), atol=1e-10)
        # Second derivative of linear should be 0
        np.testing.assert_allclose(ddf0, np.zeros(5), atol=1e-10)

    def test_shapes_match(self) -> None:
        """df0 and ddf0 have same shape as input."""
        f0_norm = np.array([1.0, 2.0, 4.0, 3.0, 1.0])

        df0, ddf0 = compute_derivatives(f0_norm)

        assert df0.shape == f0_norm.shape
        assert ddf0.shape == f0_norm.shape

    def test_quadratic_has_constant_second_derivative(self) -> None:
        """Quadratic contour has approximately constant second derivative."""
        # Quadratic: x^2 at x = 0, 1, 2, 3, 4
        f0_norm = np.array([0.0, 1.0, 4.0, 9.0, 16.0])

        df0, ddf0 = compute_derivatives(f0_norm)

        # Second derivative should be approximately 2
        # (numpy gradient uses central differences, so interior points are more accurate)
        assert np.isclose(ddf0[2], 2.0, atol=0.1)

    def test_output_dtype(self) -> None:
        """Output arrays are float64."""
        f0_norm = np.array([1.0, 2.0, 3.0])

        df0, ddf0 = compute_derivatives(f0_norm)

        assert df0.dtype == np.float64
        assert ddf0.dtype == np.float64


class TestExtractContour:
    """Tests for full contour extraction pipeline."""

    def test_full_pipeline(self) -> None:
        """Extract contour from FrameTrack + SyllableSpan."""
        # Create a FrameTrack with 100 frames at 100Hz (1 second of audio)
        f0_hz = np.linspace(100.0, 200.0, 100)  # Rising pitch
        voicing = np.ones(100)

        track = FrameTrack(
            frame_hz=100.0,
            f0_hz=f0_hz,
            voicing=voicing,
            energy=None,
        )

        span = SyllableSpan(
            index=0,
            start_ms=Ms(200),  # 200ms = frame 20
            end_ms=Ms(500),  # 500ms = frame 50
            confidence=1.0,
        )

        contour = extract_contour(track, span, k=20)

        # Should have valid contour
        assert contour.f0_norm.shape == (20,)
        assert contour.df0.shape == (20,)
        assert contour.ddf0.shape == (20,)

    def test_contour_fields(self) -> None:
        """Contour has all expected fields."""
        f0_hz = np.full(100, 150.0)
        voicing = np.ones(100)

        track = FrameTrack(
            frame_hz=100.0,
            f0_hz=f0_hz,
            voicing=voicing,
            energy=None,
        )

        span = SyllableSpan(
            index=0,
            start_ms=Ms(100),
            end_ms=Ms(400),
            confidence=0.9,
        )

        contour = extract_contour(track, span)

        # Check all fields exist and have correct types
        assert hasattr(contour, "f0_norm")
        assert hasattr(contour, "df0")
        assert hasattr(contour, "ddf0")
        assert hasattr(contour, "duration_ms")
        assert hasattr(contour, "voicing_ratio")

        assert isinstance(contour.f0_norm, np.ndarray)
        assert isinstance(contour.df0, np.ndarray)
        assert isinstance(contour.ddf0, np.ndarray)
        assert isinstance(contour.duration_ms, int)
        assert isinstance(contour.voicing_ratio, float)

    def test_duration_calculation(self) -> None:
        """Duration matches span boundaries."""
        track = FrameTrack(
            frame_hz=100.0,
            f0_hz=np.full(100, 150.0),
            voicing=np.ones(100),
            energy=None,
        )

        span = SyllableSpan(
            index=0,
            start_ms=Ms(150),
            end_ms=Ms(450),
            confidence=1.0,
        )

        contour = extract_contour(track, span)

        assert contour.duration_ms == 300  # 450 - 150

    def test_voicing_ratio_calculation(self) -> None:
        """Voicing ratio is computed from span frames."""
        f0_hz = np.full(100, 150.0)
        # Half voiced, half unvoiced in middle section
        voicing = np.zeros(100)
        voicing[20:35] = 1.0  # 15 voiced frames out of 30 (frames 20-49)

        track = FrameTrack(
            frame_hz=100.0,
            f0_hz=f0_hz,
            voicing=voicing,
            energy=None,
        )

        span = SyllableSpan(
            index=0,
            start_ms=Ms(200),  # frame 20
            end_ms=Ms(500),  # frame 50
            confidence=1.0,
        )

        contour = extract_contour(track, span)

        # 15 voiced frames out of 30 total frames
        assert np.isclose(contour.voicing_ratio, 0.5, atol=0.01)

    def test_empty_span_returns_zeros(self) -> None:
        """Empty or inverted span returns zero contour."""
        track = FrameTrack(
            frame_hz=100.0,
            f0_hz=np.full(100, 150.0),
            voicing=np.ones(100),
            energy=None,
        )

        # Inverted span (end before start)
        span = SyllableSpan(
            index=0,
            start_ms=Ms(500),
            end_ms=Ms(200),
            confidence=1.0,
        )

        contour = extract_contour(track, span, k=20)

        # Should return zeros for invalid span
        np.testing.assert_array_equal(contour.f0_norm, np.zeros(20))
        np.testing.assert_array_equal(contour.df0, np.zeros(20))
        np.testing.assert_array_equal(contour.ddf0, np.zeros(20))
        assert contour.voicing_ratio == 0.0

    def test_span_clamped_to_track_bounds(self) -> None:
        """Span is clamped to track boundaries."""
        track = FrameTrack(
            frame_hz=100.0,
            f0_hz=np.full(50, 150.0),  # Only 50 frames (500ms)
            voicing=np.ones(50),
            energy=None,
        )

        # Span extends beyond track
        span = SyllableSpan(
            index=0,
            start_ms=Ms(0),
            end_ms=Ms(1000),  # Beyond 500ms track
            confidence=1.0,
        )

        # Should not crash, should clamp to available frames
        contour = extract_contour(track, span)

        assert contour.f0_norm.shape == (20,)
        assert contour.duration_ms == 1000  # Duration still from span

    def test_default_k_is_20(self) -> None:
        """Default K value is 20 points."""
        track = FrameTrack(
            frame_hz=100.0,
            f0_hz=np.full(100, 150.0),
            voicing=np.ones(100),
            energy=None,
        )

        span = SyllableSpan(
            index=0,
            start_ms=Ms(0),
            end_ms=Ms(500),
            confidence=1.0,
        )

        contour = extract_contour(track, span)

        assert contour.f0_norm.shape == (20,)
        assert contour.df0.shape == (20,)
        assert contour.ddf0.shape == (20,)

    def test_sparse_voicing_span(self) -> None:
        """Span with insufficient voicing returns zero contour."""
        f0_hz = np.full(100, 150.0)
        voicing = np.zeros(100)
        voicing[25:27] = 1.0  # Only 2 voiced frames in span

        track = FrameTrack(
            frame_hz=100.0,
            f0_hz=f0_hz,
            voicing=voicing,
            energy=None,
        )

        span = SyllableSpan(
            index=0,
            start_ms=Ms(200),  # frame 20
            end_ms=Ms(500),  # frame 50
            confidence=1.0,
        )

        contour = extract_contour(track, span)

        # With < 3 voiced frames, should return zeros
        np.testing.assert_array_equal(contour.f0_norm, np.zeros(20))
