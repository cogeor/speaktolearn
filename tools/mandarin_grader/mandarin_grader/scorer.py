"""Overall pronunciation scoring.

This module provides the MinimalScorer class that integrates sandhi rules
and contour extraction to score Mandarin pronunciation. The current
implementation returns placeholder scores, with actual ML inference
to be added later.
"""

from dataclasses import dataclass

from .contour import extract_contour
from .sandhi import apply_tone_sandhi
from .types import (
    Contour,
    FrameTrack,
    SentenceScore,
    SyllableScores,
    SyllableSpan,
    TargetSyllable,
)


@dataclass
class ScorerConfig:
    """Configuration for the scorer."""

    contour_points: int = 20
    default_tone_score: float = 0.5
    default_segmental_score: float = 0.5
    default_fluency_score: float = 0.5


class MinimalScorer:
    """
    Minimal scorer that:
    1. Applies sandhi rules to target syllables
    2. Extracts contours for each syllable span
    3. Returns placeholder scores (no ML inference yet)

    This is a scaffold for later tone and segmental scoring.
    """

    def __init__(self, config: ScorerConfig | None = None):
        self.config = config or ScorerConfig()

    def score(
        self,
        targets: list[TargetSyllable],
        spans: list[SyllableSpan],
        track: FrameTrack,
    ) -> SentenceScore:
        """
        Score a sentence.

        Args:
            targets: Target syllables with underlying tones
            spans: Aligned syllable spans from forced alignment
            track: Frame-level acoustic features

        Returns:
            SentenceScore with per-syllable scores
        """
        # 1. Apply sandhi to get surface tones
        sandhi_targets = apply_tone_sandhi(targets)

        # 2. Extract contours for each syllable
        contours: list[Contour] = []
        for span in spans:
            contour = extract_contour(track, span, k=self.config.contour_points)
            contours.append(contour)

        # 3. Generate placeholder scores
        syllable_scores: list[SyllableScores] = []
        for target, contour in zip(sandhi_targets, contours):
            score = self._score_syllable(target, contour)
            syllable_scores.append(score)

        # 4. Compute overall score
        overall = self._compute_overall(syllable_scores)

        return SentenceScore(
            overall=overall,
            syllables=tuple(syllable_scores),
            warnings=(),
        )

    def _score_syllable(
        self, target: TargetSyllable, contour: Contour
    ) -> SyllableScores:
        """Score a single syllable (placeholder implementation)."""
        # TODO: Real tone and segmental scoring
        overall = (
            self.config.default_tone_score * 0.5
            + self.config.default_segmental_score * 0.4
            + self.config.default_fluency_score * 0.1
        )
        return SyllableScores(
            segmental=self.config.default_segmental_score,
            tone=self.config.default_tone_score,
            fluency=self.config.default_fluency_score,
            overall=overall,
            tone_probs={target.tone_surface: 1.0},
            tags=(),
        )

    def _compute_overall(self, scores: list[SyllableScores]) -> float:
        """Compute overall sentence score (0-100)."""
        if not scores:
            return 0.0
        avg = sum(s.overall for s in scores) / len(scores)
        return avg * 100
