"""Rule-based tone classifier."""

from dataclasses import dataclass

from ..types import Contour, Tone
from .base import ToneClassification, ToneClassifier, ToneFeatures


@dataclass
class RuleClassifierConfig:
    """Configuration for rule-based classifier thresholds."""

    # Neutral tone
    neutral_duration_max_ms: int = 100
    neutral_voicing_max: float = 0.4

    # Tone 1 (high level) - very strict: must be truly flat
    t1_slope_max: float = 0.02  # Nearly zero slope
    t1_range_max: float = 0.5   # Very small range

    # Tone 2 (rising)
    t2_slope_min: float = 0.03  # Positive slope
    t2_rise_min: float = 0.5    # end - start

    # Tone 3 (dipping)
    t3_min_pos_low: float = 0.25
    t3_min_pos_high: float = 0.75
    t3_range_min: float = 0.5   # Some variation

    # Tone 4 (falling)
    t4_slope_max: float = -0.03  # Negative slope
    t4_fall_min: float = 0.5     # start - end


class RuleBasedClassifier(ToneClassifier):
    """Rule-based tone classifier using contour shape features.

    Uses simple heuristics based on pitch contour characteristics:
    - Tone 1: Flat (low slope) and high level
    - Tone 2: Rising (positive slope, end > start)
    - Tone 3: Dipping (minimum in middle third)
    - Tone 4: Falling (negative slope, start > end)
    - Tone 0: Short duration or low voicing

    Advantages:
    - No training data required
    - Fully interpretable
    - Fast execution
    - Easy to tune thresholds

    Limitations:
    - May struggle with ambiguous cases
    - Thresholds may need per-speaker adjustment
    """

    def __init__(self, config: RuleClassifierConfig | None = None):
        """Initialize with configuration.

        Args:
            config: Threshold configuration. Uses defaults if None.
        """
        self.config = config or RuleClassifierConfig()

    @property
    def name(self) -> str:
        return "rule_based"

    def classify(self, contour: Contour) -> ToneClassification:
        """Classify tone using rule-based heuristics.

        Args:
            contour: Normalized pitch contour.

        Returns:
            ToneClassification with predicted tone and confidence.
        """
        features = ToneFeatures.from_contour(contour)
        cfg = self.config

        # Initialize probabilities
        probs = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        tags = []

        # Check for neutral tone first (short or unvoiced)
        # These are strong indicators - return immediately
        if features.duration_ms < cfg.neutral_duration_max_ms:
            return ToneClassification(
                predicted_tone=0,
                confidence=0.7,
                probabilities={0: 0.7, 1: 0.1, 2: 0.1, 3: 0.05, 4: 0.05},
                tags=["short_duration"],
            )
        if features.voicing_ratio < cfg.neutral_voicing_max:
            return ToneClassification(
                predicted_tone=0,
                confidence=0.6,
                probabilities={0: 0.6, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1},
                tags=["low_voicing"],
            )

        # Tone 1: High level (flat)
        if abs(features.slope) < cfg.t1_slope_max and features.range_st < cfg.t1_range_max:
            t1_score = 1.0 - abs(features.slope) / cfg.t1_slope_max
            t1_score *= 1.0 - features.range_st / cfg.t1_range_max
            probs[1] = max(probs[1], t1_score * 0.8)
            if t1_score > 0.5:
                tags.append("flat_contour")

        # Tone 2: Rising
        rise = features.end_level - features.start_level
        if features.slope > cfg.t2_slope_min and rise > cfg.t2_rise_min:
            t2_score = min(1.0, features.slope / 0.1)
            t2_score *= min(1.0, rise / 2.0)
            probs[2] = max(probs[2], t2_score * 0.85)
            if t2_score > 0.3:
                tags.append("rising_contour")

        # Tone 3: Dipping (V-shape)
        min_in_middle = cfg.t3_min_pos_low < features.min_position < cfg.t3_min_pos_high
        v_shape = (
            features.start_level > features.min_level
            and features.end_level > features.min_level
        )
        if min_in_middle and v_shape and features.range_st > cfg.t3_range_min:
            # Calculate how "V-shaped" the contour is
            dip_depth = min(
                features.start_level - features.min_level,
                features.end_level - features.min_level,
            )
            t3_score = min(1.0, dip_depth / 1.0)
            # Bonus for min being close to center
            center_dist = abs(features.min_position - 0.5)
            t3_score *= 1.0 - center_dist
            probs[3] = max(probs[3], t3_score * 0.85)
            if t3_score > 0.3:
                tags.append("dipping_contour")

        # Tone 4: Falling
        fall = features.start_level - features.end_level
        if features.slope < cfg.t4_slope_max and fall > cfg.t4_fall_min:
            t4_score = min(1.0, abs(features.slope) / 0.1)
            t4_score *= min(1.0, fall / 2.0)
            probs[4] = max(probs[4], t4_score * 0.85)
            if t4_score > 0.3:
                tags.append("falling_contour")

        # Normalize probabilities
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        else:
            # Default to uniform if all rules fail
            probs = {k: 0.2 for k in probs}
            tags.append("no_rule_matched")

        # Find best tone
        predicted = max(probs, key=probs.get)  # type: ignore
        confidence = probs[predicted]

        return ToneClassification(
            predicted_tone=predicted,  # type: ignore
            confidence=confidence,
            probabilities=probs,
            tags=tags,
        )
