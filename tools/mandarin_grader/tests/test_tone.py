"""Tests for tone classification module."""

import numpy as np
import pytest

from mandarin_grader.tone import (
    RuleBasedClassifier,
    RuleClassifierConfig,
    TemplateClassifier,
    TemplateClassifierConfig,
    ToneClassifier,
    ToneFeatures,
)
from mandarin_grader.tone.templates import dtw_distance, euclidean_distance
from mandarin_grader.types import Contour


def make_contour(
    f0_norm: list[float],
    duration_ms: int = 300,
    voicing_ratio: float = 0.9,
) -> Contour:
    """Helper to create a Contour."""
    f0 = np.array(f0_norm, dtype=np.float64)
    if len(f0) >= 2:
        df0 = np.gradient(f0)
        ddf0 = np.gradient(df0)
    else:
        df0 = np.zeros_like(f0)
        ddf0 = np.zeros_like(f0)
    return Contour(
        f0_norm=f0,
        df0=df0,
        ddf0=ddf0,
        duration_ms=duration_ms,
        voicing_ratio=voicing_ratio,
    )


class TestToneFeatures:
    """Tests for ToneFeatures extraction."""

    def test_flat_contour(self) -> None:
        """Flat contour has low slope and range."""
        contour = make_contour([0.5] * 20)
        features = ToneFeatures.from_contour(contour)

        assert abs(features.slope) < 0.01
        assert features.range_st < 0.01
        assert abs(features.start_level - 0.5) < 0.01
        assert abs(features.end_level - 0.5) < 0.01

    def test_rising_contour(self) -> None:
        """Rising contour has positive slope."""
        contour = make_contour([float(i) / 19 for i in range(20)])
        features = ToneFeatures.from_contour(contour)

        assert features.slope > 0
        assert features.end_level > features.start_level
        assert features.max_position > 0.8  # Max at end

    def test_falling_contour(self) -> None:
        """Falling contour has negative slope."""
        contour = make_contour([1.0 - float(i) / 19 for i in range(20)])
        features = ToneFeatures.from_contour(contour)

        assert features.slope < 0
        assert features.start_level > features.end_level
        assert features.max_position < 0.2  # Max at start

    def test_dipping_contour(self) -> None:
        """Dipping contour has min in middle."""
        # V-shape: start high, dip, end mid
        f0 = [0.5 - 0.5 * (1 - abs(2 * i / 19 - 1)) for i in range(20)]
        contour = make_contour(f0)
        features = ToneFeatures.from_contour(contour)

        assert 0.3 < features.min_position < 0.7
        assert features.min_level < features.start_level
        assert features.min_level < features.end_level

    def test_empty_contour(self) -> None:
        """Empty contour returns zeros."""
        contour = make_contour([])
        features = ToneFeatures.from_contour(contour)

        assert features.slope == 0.0
        assert features.range_st == 0.0

    def test_zero_contour(self) -> None:
        """All-zero contour returns zeros."""
        contour = make_contour([0.0] * 20)
        features = ToneFeatures.from_contour(contour)

        assert features.slope == 0.0
        assert features.range_st == 0.0


class TestRuleBasedClassifier:
    """Tests for RuleBasedClassifier."""

    def test_classifies_tone1_flat_high(self) -> None:
        """Flat high contour classified as Tone 1."""
        clf = RuleBasedClassifier()
        # Truly flat contour - all same value
        contour = make_contour([0.8] * 20)

        result = clf.classify(contour)

        assert result.predicted_tone == 1
        assert "flat_contour" in result.tags

    def test_classifies_tone2_rising(self) -> None:
        """Rising contour classified as Tone 2."""
        clf = RuleBasedClassifier()
        # Strong rise from -0.5 to 1.0
        contour = make_contour([-0.5 + 1.5 * i / 19 for i in range(20)])

        result = clf.classify(contour)

        assert result.predicted_tone == 2
        assert "rising_contour" in result.tags

    def test_classifies_tone3_dipping(self) -> None:
        """V-shaped contour classified as Tone 3."""
        clf = RuleBasedClassifier()
        # V-shape: 0.5 -> -0.8 -> 0.3
        f0 = []
        for i in range(20):
            if i < 10:
                f0.append(0.5 - 1.3 * i / 10)
            else:
                f0.append(-0.8 + 1.1 * (i - 10) / 10)
        contour = make_contour(f0)

        result = clf.classify(contour)

        assert result.predicted_tone == 3
        assert "dipping_contour" in result.tags

    def test_classifies_tone4_falling(self) -> None:
        """Falling contour classified as Tone 4."""
        clf = RuleBasedClassifier()
        # Strong fall from 1.0 to -0.5
        contour = make_contour([1.0 - 1.5 * i / 19 for i in range(20)])

        result = clf.classify(contour)

        assert result.predicted_tone == 4
        assert "falling_contour" in result.tags

    def test_classifies_tone0_short(self) -> None:
        """Short duration classified as Tone 0."""
        clf = RuleBasedClassifier()
        contour = make_contour([0.5] * 20, duration_ms=50)

        result = clf.classify(contour)

        assert result.predicted_tone == 0
        assert "short_duration" in result.tags

    def test_classifies_tone0_unvoiced(self) -> None:
        """Low voicing classified as Tone 0."""
        clf = RuleBasedClassifier()
        contour = make_contour([0.5] * 20, voicing_ratio=0.2)

        result = clf.classify(contour)

        assert result.predicted_tone == 0
        assert "low_voicing" in result.tags

    def test_name_property(self) -> None:
        """Classifier has correct name."""
        clf = RuleBasedClassifier()
        assert clf.name == "rule_based"

    def test_probabilities_sum_to_one(self) -> None:
        """Probabilities sum to 1."""
        clf = RuleBasedClassifier()
        contour = make_contour([0.5] * 20)

        result = clf.classify(contour)

        total = sum(result.probabilities.values())
        assert abs(total - 1.0) < 0.01


class TestTemplateClassifier:
    """Tests for TemplateClassifier."""

    def test_matches_template_exactly(self) -> None:
        """Exact template match is correct tone."""
        clf = TemplateClassifier()
        # Use Tone 1 template exactly
        contour = make_contour([0.8] * 20)

        result = clf.classify(contour)

        assert result.predicted_tone == 1
        # With 5 templates, confidence is distributed, so just check it's best
        assert result.probabilities[1] >= max(result.probabilities.values()) * 0.99

    def test_classifies_rising_as_tone2(self) -> None:
        """Rising contour matches Tone 2 template."""
        clf = TemplateClassifier()
        contour = make_contour([-0.6 + 1.5 * i / 19 for i in range(20)])

        result = clf.classify(contour)

        assert result.predicted_tone == 2

    def test_classifies_falling_as_tone4(self) -> None:
        """Falling contour matches Tone 4 template."""
        clf = TemplateClassifier()
        contour = make_contour([0.9 - 1.8 * i / 19 for i in range(20)])

        result = clf.classify(contour)

        assert result.predicted_tone == 4

    def test_name_property(self) -> None:
        """Classifier has correct name."""
        clf = TemplateClassifier()
        assert clf.name == "template"

    def test_custom_templates(self) -> None:
        """Can use custom templates."""
        custom = {
            0: [0.0] * 20,
            1: [1.0] * 20,
            2: [0.5] * 20,
            3: [-0.5] * 20,
            4: [-1.0] * 20,
        }
        clf = TemplateClassifier(templates=custom)

        contour = make_contour([1.0] * 20)
        result = clf.classify(contour)

        assert result.predicted_tone == 1

    def test_probabilities_sum_to_one(self) -> None:
        """Probabilities sum to 1."""
        clf = TemplateClassifier()
        contour = make_contour([0.5] * 20)

        result = clf.classify(contour)

        total = sum(result.probabilities.values())
        assert abs(total - 1.0) < 0.01


class TestDistanceFunctions:
    """Tests for distance functions."""

    def test_dtw_identical_sequences(self) -> None:
        """Identical sequences have zero DTW distance."""
        seq = np.array([1.0, 2.0, 3.0, 4.0])
        dist = dtw_distance(seq, seq)
        assert dist == 0.0

    def test_dtw_different_sequences(self) -> None:
        """Different sequences have positive DTW distance."""
        seq1 = np.array([1.0, 2.0, 3.0, 4.0])
        seq2 = np.array([4.0, 3.0, 2.0, 1.0])
        dist = dtw_distance(seq1, seq2)
        assert dist > 0

    def test_euclidean_identical_sequences(self) -> None:
        """Identical sequences have zero Euclidean distance."""
        seq = np.array([1.0, 2.0, 3.0, 4.0])
        dist = euclidean_distance(seq, seq)
        assert dist == 0.0

    def test_euclidean_different_lengths(self) -> None:
        """Euclidean handles different lengths via resampling."""
        seq1 = np.array([1.0, 2.0, 3.0, 4.0])
        seq2 = np.array([1.0, 2.0])
        dist = euclidean_distance(seq1, seq2)
        assert dist >= 0
