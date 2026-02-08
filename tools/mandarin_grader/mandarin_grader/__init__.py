"""
Mandarin Grader - Tone pronunciation grading library for SpeakToLearn.

This library provides tools for analyzing and scoring Mandarin Chinese
tone pronunciation using pitch contour analysis and acoustic features.

Modules:
    types: Type definitions and data structures
    sandhi: Tone sandhi rule detection and handling
    pitch: Pitch extraction and normalization
    contour: Tone contour template matching
    scorer: Overall pronunciation scoring
"""

from .contour import extract_contour, resample_contour
from .pitch import hz_to_semitones, normalize_f0, robust_stats
from .sandhi import apply_tone_sandhi
from .scorer import MinimalScorer, ScorerConfig
from .types import (
    Contour,
    FrameTrack,
    Ms,
    PhoneSpan,
    PosteriorTrack,
    SentenceScore,
    SyllableScores,
    SyllableSpan,
    TargetSyllable,
    Tone,
    ToneResult,
)

__version__ = "0.1.0"

__all__ = [
    # Types
    "Tone",
    "Ms",
    "TargetSyllable",
    "PhoneSpan",
    "SyllableSpan",
    "FrameTrack",
    "PosteriorTrack",
    "Contour",
    "ToneResult",
    "SyllableScores",
    "SentenceScore",
    # Sandhi
    "apply_tone_sandhi",
    # Pitch
    "hz_to_semitones",
    "normalize_f0",
    "robust_stats",
    # Contour
    "extract_contour",
    "resample_contour",
    # Scorer
    "MinimalScorer",
    "ScorerConfig",
]
