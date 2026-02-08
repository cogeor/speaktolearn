"""Overall pronunciation scoring.

This module provides the DeterministicScorer class that integrates:
- Alignment (DTW or uniform)
- Tone classification (rule-based or template)
- Contour extraction
- Score fusion

The architecture supports swapping components for A/B testing.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .align import Aligner, AlignmentResult, DTWAligner, UniformAligner
from .contour import extract_contour
from .pitch import extract_f0_pyin, hz_to_semitones, normalize_f0
from .sandhi import apply_tone_sandhi
from .tone import RuleBasedClassifier, TemplateClassifier, ToneClassifier
from .types import (
    Contour,
    FrameTrack,
    Ms,
    SentenceScore,
    SyllableScores,
    SyllableSpan,
    TargetSyllable,
)


@dataclass
class ScorerConfig:
    """Configuration for the scorer."""

    contour_points: int = 20
    default_segmental_score: float = 0.7  # Placeholder until segmental scoring
    aligner_type: Literal["uniform", "dtw"] = "uniform"
    classifier_type: Literal["rule", "template"] = "rule"
    tone_weight: float = 0.7
    segmental_weight: float = 0.3


class DeterministicScorer:
    """Deterministic scorer using pluggable alignment and tone classification.

    This scorer:
    1. Applies sandhi rules to target syllables
    2. Aligns audio to get syllable spans
    3. Extracts pitch contours per syllable
    4. Classifies tones using rule-based or template classifier
    5. Computes scores based on predicted vs expected tone

    Components are swappable for A/B testing.
    """

    def __init__(
        self,
        config: ScorerConfig | None = None,
        aligner: Aligner | None = None,
        classifier: ToneClassifier | None = None,
    ):
        """Initialize scorer with optional custom components.

        Args:
            config: Scorer configuration.
            aligner: Custom aligner (overrides config.aligner_type).
            classifier: Custom classifier (overrides config.classifier_type).
        """
        self.config = config or ScorerConfig()

        # Set up aligner
        if aligner is not None:
            self.aligner = aligner
        elif self.config.aligner_type == "dtw":
            self.aligner = DTWAligner()
        else:
            self.aligner = UniformAligner()

        # Set up classifier
        if classifier is not None:
            self.classifier = classifier
        elif self.config.classifier_type == "template":
            self.classifier = TemplateClassifier()
        else:
            self.classifier = RuleBasedClassifier()

    @property
    def name(self) -> str:
        """Return scorer name for logging."""
        return f"deterministic_{self.aligner.name}_{self.classifier.name}"

    def score(
        self,
        audio: NDArray[np.floating],
        targets: list[TargetSyllable],
        sr: int = 16000,
        reference_audio: NDArray[np.floating] | None = None,
    ) -> SentenceScore:
        """Score a pronunciation attempt.

        Args:
            audio: Learner audio samples (mono, 16kHz).
            targets: Target syllables with underlying tones.
            sr: Sample rate.
            reference_audio: Optional TTS reference for DTW alignment.

        Returns:
            SentenceScore with per-syllable scores.
        """
        if not targets:
            return SentenceScore(overall=0.0, syllables=(), warnings=("no_targets",))

        # 1. Apply sandhi to get surface tones
        sandhi_targets = apply_tone_sandhi(targets)

        # 2. Align to get syllable spans
        if reference_audio is not None and isinstance(self.aligner, DTWAligner):
            alignment = self.aligner.align_with_reference(
                audio, reference_audio, sandhi_targets, sr
            )
        else:
            alignment = self.aligner.align(audio, sandhi_targets, sr)

        # 3. Extract pitch track
        track = self._extract_pitch_track(audio, sr)

        # 4. Score each syllable
        syllable_scores: list[SyllableScores] = []
        warnings = list(alignment.warnings)

        for target, span in zip(sandhi_targets, alignment.syllable_spans):
            # Extract contour for this syllable
            contour = extract_contour(track, span, k=self.config.contour_points)

            # Classify tone
            classification = self.classifier.classify(contour)

            # Compute tone score
            tone_score = self._compute_tone_score(
                predicted=classification.predicted_tone,
                expected=target.tone_surface,
                confidence=classification.confidence,
            )

            # Overall syllable score
            overall = (
                self.config.tone_weight * tone_score
                + self.config.segmental_weight * self.config.default_segmental_score
            )

            # Collect tags
            tags = list(classification.tags)
            if classification.predicted_tone != target.tone_surface:
                tags.append(
                    f"tone_{classification.predicted_tone}_vs_{target.tone_surface}"
                )

            syllable_scores.append(
                SyllableScores(
                    segmental=self.config.default_segmental_score,
                    tone=tone_score,
                    fluency=1.0,  # Placeholder
                    overall=overall,
                    tone_probs=classification.probabilities,
                    tags=tuple(tags),
                )
            )

        # 5. Compute overall sentence score
        overall = self._compute_overall(syllable_scores)

        return SentenceScore(
            overall=overall,
            syllables=tuple(syllable_scores),
            warnings=tuple(warnings),
        )

    def _extract_pitch_track(
        self, audio: NDArray[np.floating], sr: int
    ) -> FrameTrack:
        """Extract pitch track from audio."""
        f0_hz, voicing = extract_f0_pyin(audio, sr=sr)
        return FrameTrack(
            frame_hz=sr / 160,  # 10ms hop
            f0_hz=f0_hz,
            voicing=voicing,
            energy=None,
        )

    def _compute_tone_score(
        self, predicted: int, expected: int, confidence: float
    ) -> float:
        """Compute tone score based on prediction accuracy.

        Args:
            predicted: Predicted tone (0-4).
            expected: Expected surface tone (0-4).
            confidence: Classifier confidence (0-1).

        Returns:
            Tone score (0-1).
        """
        if predicted == expected:
            # Correct prediction - scale by confidence
            return 0.7 + 0.3 * confidence
        else:
            # Incorrect - apply confusion penalty
            penalty = CONFUSION_PENALTIES.get((expected, predicted), 0.5)
            return max(0.0, 0.4 * (1 - penalty) * confidence)

    def _compute_overall(self, scores: list[SyllableScores]) -> float:
        """Compute overall sentence score (0-100)."""
        if not scores:
            return 0.0

        # Weighted average with penalty for worst syllables
        sorted_scores = sorted(s.overall for s in scores)
        n = len(sorted_scores)

        # Bottom 30% weighted 1.5x (penalize worst performance)
        bottom_idx = max(1, int(n * 0.3))
        bottom_weight = 1.5
        normal_weight = 1.0

        weighted_sum = sum(s * bottom_weight for s in sorted_scores[:bottom_idx])
        weighted_sum += sum(s * normal_weight for s in sorted_scores[bottom_idx:])
        total_weight = bottom_idx * bottom_weight + (n - bottom_idx) * normal_weight

        avg = weighted_sum / total_weight
        return avg * 100


# Confusion penalties (expected, predicted) -> penalty (0 = no penalty, 1 = full penalty)
# Based on linguistic confusion patterns
CONFUSION_PENALTIES = {
    # Tone 2/3 confusion is common (both have low points)
    (2, 3): 0.3,
    (3, 2): 0.3,
    # Tone 1/4 confusion (both can sound "sharp")
    (1, 4): 0.4,
    (4, 1): 0.4,
    # Neutral tone confusions (neutral is context-dependent)
    (0, 1): 0.2,
    (0, 2): 0.3,
    (0, 3): 0.3,
    (0, 4): 0.2,
    (1, 0): 0.2,
    (2, 0): 0.3,
    (3, 0): 0.3,
    (4, 0): 0.2,
}


# Keep old MinimalScorer for backwards compatibility
@dataclass
class MinimalScorerConfig:
    """Configuration for the minimal scorer."""

    contour_points: int = 20
    default_tone_score: float = 0.5
    default_segmental_score: float = 0.5
    default_fluency_score: float = 0.5


class MinimalScorer:
    """Legacy minimal scorer with placeholder scores.

    Use DeterministicScorer instead for actual scoring.
    """

    def __init__(self, config: MinimalScorerConfig | None = None):
        self.config = config or MinimalScorerConfig()

    def score(
        self,
        targets: list[TargetSyllable],
        spans: list[SyllableSpan],
        track: FrameTrack,
    ) -> SentenceScore:
        """Score a sentence with placeholder scores."""
        sandhi_targets = apply_tone_sandhi(targets)

        contours = []
        for span in spans:
            contour = extract_contour(track, span, k=self.config.contour_points)
            contours.append(contour)

        syllable_scores = []
        for target, contour in zip(sandhi_targets, contours):
            overall = (
                self.config.default_tone_score * 0.5
                + self.config.default_segmental_score * 0.4
                + self.config.default_fluency_score * 0.1
            )
            syllable_scores.append(
                SyllableScores(
                    segmental=self.config.default_segmental_score,
                    tone=self.config.default_tone_score,
                    fluency=self.config.default_fluency_score,
                    overall=overall,
                    tone_probs={target.tone_surface: 1.0},
                    tags=(),
                )
            )

        if not syllable_scores:
            return SentenceScore(overall=0.0, syllables=(), warnings=())

        avg = sum(s.overall for s in syllable_scores) / len(syllable_scores)
        return SentenceScore(
            overall=avg * 100,
            syllables=tuple(syllable_scores),
            warnings=(),
        )
