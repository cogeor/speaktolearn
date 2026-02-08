"""Type definitions and data structures for Mandarin tone grading."""

from dataclasses import dataclass
from typing import Literal, NewType

import numpy as np
from numpy.typing import NDArray

# Type aliases
Tone = Literal[0, 1, 2, 3, 4]  # 0 = neutral, 1-4 = standard tones
Ms = NewType("Ms", int)  # Milliseconds


@dataclass(frozen=True)
class TargetSyllable:
    """Represents a target syllable to be graded."""

    index: int
    hanzi: str
    pinyin: str
    initial: str
    final: str
    tone_underlying: Tone
    tone_surface: Tone
    start_expected_ms: Ms | None = None
    end_expected_ms: Ms | None = None


@dataclass(frozen=True)
class PhoneSpan:
    """Represents a phone (phoneme) with timing information."""

    phone: str
    start_ms: Ms
    end_ms: Ms
    confidence: float  # 0..1


@dataclass(frozen=True)
class SyllableSpan:
    """Represents a syllable with timing and alignment information."""

    index: int
    start_ms: Ms
    end_ms: Ms
    confidence: float  # 0..1
    phone_spans: tuple[PhoneSpan, ...] | None = None


@dataclass(frozen=True)
class FrameTrack:
    """Acoustic frame-level features extracted from audio."""

    frame_hz: float
    f0_hz: NDArray[np.floating]  # shape [T]
    voicing: NDArray[np.floating]  # shape [T], 0..1
    energy: NDArray[np.floating] | None = None


@dataclass(frozen=True)
class PosteriorTrack:
    """Posterior probabilities over tokens at each frame."""

    frame_hz: float
    token_names: tuple[str, ...]
    posteriors: NDArray[np.floating]  # shape [T, V]


@dataclass(frozen=True)
class Contour:
    """Normalized pitch contour features for a syllable."""

    f0_norm: NDArray[np.floating]  # shape [K]
    df0: NDArray[np.floating]  # shape [K]
    ddf0: NDArray[np.floating]  # shape [K]
    duration_ms: int
    voicing_ratio: float


@dataclass(frozen=True)
class ToneResult:
    """Result of tone classification for a syllable."""

    score: float
    probs: dict[int, float]  # tone -> probability
    tags: tuple[str, ...]


@dataclass(frozen=True)
class SyllableScores:
    """Detailed scores for a single syllable."""

    segmental: float
    tone: float
    fluency: float
    overall: float
    tone_probs: dict[int, float]
    tags: tuple[str, ...]


@dataclass(frozen=True)
class SentenceScore:
    """Overall scoring result for a sentence."""

    overall: float  # 0..100
    syllables: tuple[SyllableScores, ...]
    warnings: tuple[str, ...]
