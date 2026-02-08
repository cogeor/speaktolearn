"""Base classes and protocols for alignment."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..types import Ms, SyllableSpan, TargetSyllable


@dataclass
class AlignmentResult:
    """Result of syllable alignment."""

    syllable_spans: list[SyllableSpan]
    overall_confidence: float
    warnings: list[str]

    def __len__(self) -> int:
        return len(self.syllable_spans)


class Aligner(ABC):
    """Abstract base class for syllable aligners.

    Aligners take audio and target syllables and produce syllable spans
    with timing information. This allows swapping between different
    alignment strategies (uniform, DTW, CTC, etc.).
    """

    @abstractmethod
    def align(
        self,
        audio: NDArray[np.floating],
        targets: list[TargetSyllable],
        sr: int = 16000,
    ) -> AlignmentResult:
        """Align audio to target syllables.

        Args:
            audio: Audio samples as float array (mono, 16kHz).
            targets: Target syllables to align to.
            sr: Sample rate in Hz.

        Returns:
            AlignmentResult with syllable spans and confidence.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the aligner name for logging/benchmarking."""
        pass
