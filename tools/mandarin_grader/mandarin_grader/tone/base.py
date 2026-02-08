"""Base classes and protocols for tone classification."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..types import Contour, Tone


@dataclass
class ToneFeatures:
    """Features extracted from a pitch contour for tone classification.

    These features characterize the shape of the pitch contour
    and are used by rule-based and statistical classifiers.
    """

    slope: float  # Linear regression slope
    range_st: float  # Max - min in semitones
    start_level: float  # Mean of first 20% of contour
    end_level: float  # Mean of last 20% of contour
    min_level: float  # Minimum value
    max_level: float  # Maximum value
    min_position: float  # Position of minimum (0-1)
    max_position: float  # Position of maximum (0-1)
    duration_ms: int  # Syllable duration
    voicing_ratio: float  # Fraction of voiced frames

    @classmethod
    def from_contour(cls, contour: Contour) -> "ToneFeatures":
        """Extract ToneFeatures from a Contour.

        Args:
            contour: Normalized pitch contour.

        Returns:
            ToneFeatures instance.
        """
        f0 = contour.f0_norm
        k = len(f0)

        if k == 0 or np.all(f0 == 0):
            return cls(
                slope=0.0,
                range_st=0.0,
                start_level=0.0,
                end_level=0.0,
                min_level=0.0,
                max_level=0.0,
                min_position=0.5,
                max_position=0.5,
                duration_ms=contour.duration_ms,
                voicing_ratio=contour.voicing_ratio,
            )

        # Linear regression slope
        x = np.arange(k)
        slope = float(np.polyfit(x, f0, 1)[0])

        # Range
        min_val = float(np.min(f0))
        max_val = float(np.max(f0))
        range_st = max_val - min_val

        # Start/end levels (first/last 20%)
        segment_size = max(1, k // 5)
        start_level = float(np.mean(f0[:segment_size]))
        end_level = float(np.mean(f0[-segment_size:]))

        # Min/max positions (normalized 0-1)
        min_pos = float(np.argmin(f0) / max(1, k - 1))
        max_pos = float(np.argmax(f0) / max(1, k - 1))

        return cls(
            slope=slope,
            range_st=range_st,
            start_level=start_level,
            end_level=end_level,
            min_level=min_val,
            max_level=max_val,
            min_position=min_pos,
            max_position=max_pos,
            duration_ms=contour.duration_ms,
            voicing_ratio=contour.voicing_ratio,
        )


@dataclass
class ToneClassification:
    """Result of tone classification."""

    predicted_tone: Tone
    confidence: float  # 0-1
    probabilities: dict[int, float]  # Tone -> probability
    tags: list[str]  # Diagnostic tags


class ToneClassifier(ABC):
    """Abstract base class for tone classifiers.

    Classifiers take pitch contour features and predict the tone.
    This allows swapping between rule-based, template-matching,
    and neural classifiers.
    """

    @abstractmethod
    def classify(self, contour: Contour) -> ToneClassification:
        """Classify the tone from a pitch contour.

        Args:
            contour: Normalized pitch contour.

        Returns:
            ToneClassification with predicted tone and confidence.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the classifier name for logging/benchmarking."""
        pass
