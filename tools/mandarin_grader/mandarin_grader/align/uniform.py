"""Uniform aligner - simple baseline that divides audio evenly."""

import numpy as np
from numpy.typing import NDArray

from ..types import Ms, SyllableSpan, TargetSyllable
from .base import Aligner, AlignmentResult


class UniformAligner(Aligner):
    """Aligner that divides audio uniformly among syllables.

    This is a simple baseline that assumes each syllable takes
    an equal portion of the total audio duration. Useful for:
    - Testing the pipeline without complex alignment
    - Comparing against more sophisticated aligners
    - Cases where audio closely matches expected timing

    Not suitable for:
    - Variable-speed speech
    - Audio with pauses or hesitations
    """

    @property
    def name(self) -> str:
        return "uniform"

    def align(
        self,
        audio: NDArray[np.floating],
        targets: list[TargetSyllable],
        sr: int = 16000,
    ) -> AlignmentResult:
        """Align by dividing audio uniformly among syllables.

        Args:
            audio: Audio samples.
            targets: Target syllables.
            sr: Sample rate.

        Returns:
            AlignmentResult with evenly-spaced spans.
        """
        if not targets:
            return AlignmentResult(
                syllable_spans=[],
                overall_confidence=1.0,
                warnings=["no_targets"],
            )

        # Calculate total duration in ms
        total_ms = int(len(audio) / sr * 1000)
        n_syllables = len(targets)
        ms_per_syllable = total_ms / n_syllables

        # Create evenly-spaced spans
        spans = []
        for i, target in enumerate(targets):
            start_ms = int(i * ms_per_syllable)
            end_ms = int((i + 1) * ms_per_syllable)

            # Ensure last syllable extends to end
            if i == n_syllables - 1:
                end_ms = total_ms

            spans.append(SyllableSpan(
                index=i,
                start_ms=Ms(start_ms),
                end_ms=Ms(end_ms),
                confidence=0.5,  # Low confidence for uniform assumption
                phone_spans=None,
            ))

        return AlignmentResult(
            syllable_spans=spans,
            overall_confidence=0.5,
            warnings=["uniform_alignment"],
        )
