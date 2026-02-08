"""Tests for alignment module."""

import numpy as np
import pytest

from mandarin_grader.align import (
    Aligner,
    AlignmentResult,
    DTWAligner,
    UniformAligner,
)
from mandarin_grader.align.dtw import dtw_alignment, project_boundaries
from mandarin_grader.types import Ms, TargetSyllable


def make_target(pinyin: str, tone: int, index: int = 0) -> TargetSyllable:
    """Helper to create a target syllable."""
    return TargetSyllable(
        index=index,
        hanzi="测",
        pinyin=pinyin,
        initial="c",
        final="e",
        tone_underlying=tone,
        tone_surface=tone,
    )


class TestUniformAligner:
    """Tests for UniformAligner."""

    def test_aligns_single_syllable(self) -> None:
        """Single syllable gets full duration."""
        aligner = UniformAligner()
        audio = np.zeros(16000)  # 1 second at 16kHz
        targets = [make_target("ni", 3)]

        result = aligner.align(audio, targets)

        assert len(result) == 1
        assert result.syllable_spans[0].start_ms == 0
        assert result.syllable_spans[0].end_ms == 1000

    def test_aligns_multiple_syllables(self) -> None:
        """Multiple syllables divided evenly."""
        aligner = UniformAligner()
        audio = np.zeros(32000)  # 2 seconds
        targets = [make_target("ni", 3, i) for i in range(4)]

        result = aligner.align(audio, targets)

        assert len(result) == 4
        # Each syllable should be ~500ms
        for i, span in enumerate(result.syllable_spans):
            assert span.start_ms == i * 500
            expected_end = (i + 1) * 500 if i < 3 else 2000
            assert span.end_ms == expected_end

    def test_empty_targets(self) -> None:
        """Empty targets returns empty result."""
        aligner = UniformAligner()
        audio = np.zeros(16000)

        result = aligner.align(audio, [])

        assert len(result) == 0
        assert "no_targets" in result.warnings

    def test_name_property(self) -> None:
        """Aligner has correct name."""
        aligner = UniformAligner()
        assert aligner.name == "uniform"

    def test_low_confidence(self) -> None:
        """Uniform alignment has low confidence."""
        aligner = UniformAligner()
        audio = np.zeros(16000)
        targets = [make_target("ni", 3)]

        result = aligner.align(audio, targets)

        assert result.overall_confidence == 0.5
        assert result.syllable_spans[0].confidence == 0.5


class TestDTWAlignment:
    """Tests for DTW alignment functions."""

    def test_identical_sequences(self) -> None:
        """Identical sequences have zero cost diagonal path."""
        seq = np.random.randn(10, 5)
        cost, path = dtw_alignment(seq, seq)

        assert cost < 0.1  # Nearly zero
        assert len(path) == 10
        # Path should be diagonal
        for i, (q, r) in enumerate(path):
            assert q == i
            assert r == i

    def test_shifted_sequence(self) -> None:
        """Shifted sequence still aligns."""
        seq1 = np.random.randn(10, 5)
        # Shift by inserting at beginning
        seq2 = np.vstack([np.random.randn(2, 5), seq1])

        cost, path = dtw_alignment(seq1, seq2, band_fraction=0.5)

        assert len(path) > 0
        assert cost < 20  # Should still be reasonable

    def test_empty_sequence(self) -> None:
        """Empty sequence returns inf cost."""
        seq = np.random.randn(10, 5)
        empty = np.array([]).reshape(0, 5)

        cost, path = dtw_alignment(seq, empty)

        assert cost == float("inf")
        assert path == []


class TestProjectBoundaries:
    """Tests for boundary projection."""

    def test_diagonal_path(self) -> None:
        """Diagonal path preserves boundaries."""
        path = [(i, i) for i in range(100)]
        # Ref boundaries in ms, path has 100 frames at 10ms each = 1000ms
        # So boundaries at frame indices: 0, 25, 50, 75, 100
        ref_boundaries_ms = [0, 250, 500, 750, 1000]

        result = project_boundaries(ref_boundaries_ms, path, hop_length_ms=10.0, query_duration_ms=1000)

        # Should map frame 25 (ref_ms=250) to frame 25 (query_ms=250), etc.
        assert len(result) == len(ref_boundaries_ms)
        # First boundary always 0
        assert result[0] == 0
        # Boundaries should be preserved on diagonal path
        for r, ref in zip(result, ref_boundaries_ms):
            # Allow some tolerance for rounding
            assert abs(r - ref) <= 10, f"Expected ~{ref}, got {r}"

    def test_stretched_path(self) -> None:
        """Stretched query maps correctly."""
        # Query is 2x length of reference
        path = [(i, i // 2) for i in range(200)]
        ref_boundaries = [0, 500, 1000]

        result = project_boundaries(
            ref_boundaries, path, hop_length_ms=10.0, query_duration_ms=2000
        )

        # Should be stretched
        assert result[0] == 0
        assert result[-1] <= 2000

    def test_empty_path(self) -> None:
        """Empty path returns original boundaries."""
        ref_boundaries = [0, 500, 1000]

        result = project_boundaries(ref_boundaries, [])

        assert result == ref_boundaries


class TestDTWAligner:
    """Tests for DTWAligner."""

    def test_align_without_reference_falls_back(self) -> None:
        """Without reference, falls back to uniform."""
        aligner = DTWAligner()
        audio = np.zeros(16000)
        targets = [make_target("ni", 3)]

        result = aligner.align(audio, targets)

        assert len(result) == 1
        assert "dtw_fallback_to_uniform" in result.warnings

    def test_name_property(self) -> None:
        """Aligner has correct name."""
        aligner = DTWAligner()
        assert aligner.name == "dtw"


# Skip librosa-dependent tests if not available
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


@pytest.mark.skipif(not HAS_LIBROSA, reason="librosa not installed")
class TestDTWAlignerWithReference:
    """Tests for DTW aligner with reference audio."""

    def test_align_identical_audio(self) -> None:
        """Identical audio produces diagonal alignment."""
        aligner = DTWAligner()

        # Generate simple sine wave
        sr = 16000
        t = np.linspace(0, 1, sr)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        targets = [make_target("ni", 3, i) for i in range(2)]

        result = aligner.align_with_reference(
            learner_audio=audio,
            reference_audio=audio,
            targets=targets,
        )

        assert len(result) == 2
        # Should have high confidence for identical audio
        assert result.overall_confidence > 0.5

    def test_align_different_duration(self) -> None:
        """Handles audio of different durations."""
        aligner = DTWAligner()

        sr = 16000
        t1 = np.linspace(0, 1, sr)
        t2 = np.linspace(0, 1.5, int(sr * 1.5))

        # Same frequency, different duration
        learner = np.sin(2 * np.pi * 440 * t2).astype(np.float32)
        ref = np.sin(2 * np.pi * 440 * t1).astype(np.float32)

        targets = [make_target("ni", 3, i) for i in range(2)]

        result = aligner.align_with_reference(
            learner_audio=learner,
            reference_audio=ref,
            targets=targets,
        )

        assert len(result) == 2
        # Last syllable should extend to learner duration
        assert result.syllable_spans[-1].end_ms <= 1500
