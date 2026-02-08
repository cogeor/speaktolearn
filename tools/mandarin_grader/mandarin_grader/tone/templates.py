"""Template-based tone classifier using DTW distance."""

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
from numpy.typing import NDArray

from ..types import Contour, Tone
from .base import ToneClassification, ToneClassifier, ToneFeatures


# Default tone templates (K=20 points, normalized F0)
# These are idealized contours based on linguistic descriptions
# Should be replaced with empirically-derived templates from TTS data
DEFAULT_TEMPLATES: Dict[int, list[float]] = {
    # Tone 1: High level (55 in Chao notation)
    1: [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
        0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],

    # Tone 2: Rising (35)
    2: [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3,
        0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],

    # Tone 3: Dipping (214)
    3: [0.2, 0.1, 0.0, -0.1, -0.3, -0.5, -0.7, -0.8, -0.8, -0.7,
        -0.6, -0.4, -0.2, 0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5],

    # Tone 4: Falling (51)
    4: [0.9, 0.85, 0.75, 0.6, 0.45, 0.3, 0.15, 0.0, -0.15, -0.3,
        -0.45, -0.55, -0.65, -0.7, -0.75, -0.8, -0.82, -0.85, -0.87, -0.9],

    # Tone 0: Neutral (short, mid-level)
    0: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}


def dtw_distance(
    seq1: NDArray[np.floating],
    seq2: NDArray[np.floating],
) -> float:
    """Compute DTW distance between two sequences.

    Args:
        seq1: First sequence [N].
        seq2: Second sequence [M].

    Returns:
        DTW distance (lower = more similar).
    """
    n, m = len(seq1), len(seq2)

    if n == 0 or m == 0:
        return float("inf")

    # Initialize DP matrix
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0

    # Fill DP matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(seq1[i - 1] - seq2[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    return float(dp[n, m])


def euclidean_distance(
    seq1: NDArray[np.floating],
    seq2: NDArray[np.floating],
) -> float:
    """Compute Euclidean distance between two sequences.

    Assumes sequences have the same length.

    Args:
        seq1: First sequence [N].
        seq2: Second sequence [N].

    Returns:
        Euclidean distance.
    """
    if len(seq1) != len(seq2):
        # Resample shorter to longer
        if len(seq1) < len(seq2):
            seq1 = np.interp(
                np.linspace(0, 1, len(seq2)),
                np.linspace(0, 1, len(seq1)),
                seq1,
            )
        else:
            seq2 = np.interp(
                np.linspace(0, 1, len(seq1)),
                np.linspace(0, 1, len(seq2)),
                seq2,
            )

    return float(np.sqrt(np.mean((seq1 - seq2) ** 2)))


@dataclass
class TemplateClassifierConfig:
    """Configuration for template classifier."""

    use_dtw: bool = False  # Use DTW (slower) or Euclidean (faster)
    temperature: float = 2.0  # Softmax temperature for probabilities
    neutral_duration_max_ms: int = 100  # Short syllables -> neutral
    neutral_voicing_max: float = 0.4  # Low voicing -> neutral


class TemplateClassifier(ToneClassifier):
    """Template-based tone classifier using distance to canonical templates.

    Compares the learner's pitch contour to idealized templates for each tone
    and predicts the tone with the smallest distance.

    Advantages:
    - Simple and interpretable
    - Can use empirically-derived templates from TTS data
    - Works well when contours are clear

    Limitations:
    - Sensitive to normalization quality
    - May struggle with atypical pronunciations
    """

    def __init__(
        self,
        templates: Dict[int, list[float]] | None = None,
        config: TemplateClassifierConfig | None = None,
    ):
        """Initialize with templates and configuration.

        Args:
            templates: Tone templates {tone: contour}. Uses defaults if None.
            config: Configuration options.
        """
        self.templates = {
            k: np.array(v, dtype=np.float64)
            for k, v in (templates or DEFAULT_TEMPLATES).items()
        }
        self.config = config or TemplateClassifierConfig()

    @property
    def name(self) -> str:
        return "template"

    def classify(self, contour: Contour) -> ToneClassification:
        """Classify tone by distance to templates.

        Args:
            contour: Normalized pitch contour.

        Returns:
            ToneClassification with predicted tone and confidence.
        """
        cfg = self.config
        tags = []

        # Check for neutral tone first
        if contour.duration_ms < cfg.neutral_duration_max_ms:
            tags.append("short_duration")
            return ToneClassification(
                predicted_tone=0,
                confidence=0.7,
                probabilities={0: 0.7, 1: 0.075, 2: 0.075, 3: 0.075, 4: 0.075},
                tags=tags,
            )

        if contour.voicing_ratio < cfg.neutral_voicing_max:
            tags.append("low_voicing")
            return ToneClassification(
                predicted_tone=0,
                confidence=0.6,
                probabilities={0: 0.6, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1},
                tags=tags,
            )

        # Compute distances to each template
        learner_contour = contour.f0_norm
        distances = {}

        distance_fn = dtw_distance if cfg.use_dtw else euclidean_distance

        for tone, template in self.templates.items():
            dist = distance_fn(learner_contour, template)
            distances[tone] = dist

        # Convert distances to probabilities using softmax
        # Lower distance = higher probability
        neg_distances = {k: -v / cfg.temperature for k, v in distances.items()}
        max_neg = max(neg_distances.values())
        exp_vals = {k: np.exp(v - max_neg) for k, v in neg_distances.items()}
        total_exp = sum(exp_vals.values())
        probs = {k: v / total_exp for k, v in exp_vals.items()}

        # Find best tone
        predicted = min(distances, key=distances.get)  # type: ignore
        confidence = probs[predicted]

        # Add diagnostic tag
        min_dist = distances[predicted]
        if min_dist < 0.5:
            tags.append("strong_match")
        elif min_dist < 1.0:
            tags.append("moderate_match")
        else:
            tags.append("weak_match")

        return ToneClassification(
            predicted_tone=predicted,  # type: ignore
            confidence=confidence,
            probabilities=probs,
            tags=tags,
        )

    def update_templates(self, new_templates: Dict[int, NDArray[np.floating]]) -> None:
        """Update templates with empirically-derived contours.

        Args:
            new_templates: New templates {tone: contour array}.
        """
        self.templates = {
            k: np.array(v, dtype=np.float64) for k, v in new_templates.items()
        }
