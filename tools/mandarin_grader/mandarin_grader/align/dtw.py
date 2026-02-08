"""DTW-based aligner using TTS reference audio."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from ..types import Ms, SyllableSpan, TargetSyllable
from .base import Aligner, AlignmentResult


def extract_mfcc(
    audio: NDArray[np.floating],
    sr: int = 16000,
    n_mfcc: int = 13,
    hop_length: int = 160,
) -> NDArray[np.floating]:
    """Extract MFCC features for DTW alignment.

    Args:
        audio: Audio samples.
        sr: Sample rate.
        n_mfcc: Number of MFCC coefficients.
        hop_length: Hop length in samples (160 = 10ms at 16kHz).

    Returns:
        MFCC features, shape [T, n_mfcc].
    """
    import librosa

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        hop_length=hop_length,
    )
    return mfcc.T.astype(np.float64)  # [T, n_mfcc]


def dtw_alignment(
    query: NDArray[np.floating],
    reference: NDArray[np.floating],
    band_fraction: float = 0.3,
) -> tuple[float, list[tuple[int, int]]]:
    """Compute DTW alignment between query and reference features.

    Uses Sakoe-Chiba band constraint for efficiency.

    Args:
        query: Query features [T1, D].
        reference: Reference features [T2, D].
        band_fraction: Fraction of length for band width.

    Returns:
        Tuple of (total_cost, warping_path).
        Warping path is list of (query_idx, ref_idx) tuples.
    """
    T1, T2 = len(query), len(reference)

    if T1 == 0 or T2 == 0:
        return float("inf"), []

    band_width = max(1, int(max(T1, T2) * band_fraction))

    # Compute cost matrix (cosine distance)
    # Normalize features
    query_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
    ref_norm = reference / (np.linalg.norm(reference, axis=1, keepdims=True) + 1e-8)

    # Initialize DP matrix
    dp = np.full((T1 + 1, T2 + 1), np.inf)
    dp[0, 0] = 0

    # Fill DP matrix within band
    for i in range(1, T1 + 1):
        # Compute expected j position for diagonal
        expected_j = int(i * T2 / T1)
        j_min = max(1, expected_j - band_width)
        j_max = min(T2, expected_j + band_width)

        for j in range(j_min, j_max + 1):
            # Cosine distance: 1 - cosine_similarity
            cost = 1.0 - np.dot(query_norm[i - 1], ref_norm[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    # Backtrack to get path
    path = []
    i, j = T1, T2

    if np.isinf(dp[i, j]):
        # No valid path found within band
        return float("inf"), []

    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        candidates = [
            (dp[i - 1, j], i - 1, j),
            (dp[i, j - 1], i, j - 1),
            (dp[i - 1, j - 1], i - 1, j - 1),
        ]
        _, i, j = min(candidates, key=lambda x: x[0])

    return dp[T1, T2], path[::-1]


def project_boundaries(
    ref_boundaries_ms: list[int],
    warping_path: list[tuple[int, int]],
    hop_length_ms: float = 10.0,
    query_duration_ms: int = 0,
) -> list[int]:
    """Project reference boundaries to query through warping path.

    Args:
        ref_boundaries_ms: Reference syllable boundaries in ms.
        warping_path: DTW warping path [(query_idx, ref_idx), ...].
        hop_length_ms: Frame duration in ms.
        query_duration_ms: Total query duration in ms.

    Returns:
        Query boundaries in ms.
    """
    if not warping_path:
        return ref_boundaries_ms

    # Build mapping from ref frame to query frame
    ref_to_query = {}
    for q_idx, r_idx in warping_path:
        # Take first query frame for each ref frame
        if r_idx not in ref_to_query:
            ref_to_query[r_idx] = q_idx

    # Get max frames
    max_ref_frame = max(r for _, r in warping_path)
    max_query_frame = max(q for q, _ in warping_path)

    query_boundaries = []
    for ref_ms in ref_boundaries_ms:
        ref_frame = int(ref_ms / hop_length_ms)

        # Find closest mapped frame
        if ref_frame in ref_to_query:
            query_frame = ref_to_query[ref_frame]
        else:
            # Interpolate
            available_frames = sorted(ref_to_query.keys())
            if not available_frames:
                query_frame = int(ref_frame * max_query_frame / max(1, max_ref_frame))
            elif ref_frame <= available_frames[0]:
                query_frame = ref_to_query[available_frames[0]]
            elif ref_frame >= available_frames[-1]:
                query_frame = ref_to_query[available_frames[-1]]
            else:
                # Find bracketing frames
                for k in range(len(available_frames) - 1):
                    if available_frames[k] <= ref_frame < available_frames[k + 1]:
                        f1, f2 = available_frames[k], available_frames[k + 1]
                        q1, q2 = ref_to_query[f1], ref_to_query[f2]
                        # Linear interpolation
                        t = (ref_frame - f1) / (f2 - f1)
                        query_frame = int(q1 + t * (q2 - q1))
                        break
                else:
                    query_frame = ref_frame

        query_ms = int(query_frame * hop_length_ms)
        query_boundaries.append(min(query_ms, query_duration_ms))

    return query_boundaries


@dataclass
class DTWAlignerConfig:
    """Configuration for DTW aligner."""

    n_mfcc: int = 13
    hop_length: int = 160  # 10ms at 16kHz
    band_fraction: float = 0.3
    min_confidence: float = 0.3


class DTWAligner(Aligner):
    """Aligner using Dynamic Time Warping with TTS reference.

    This aligner compares learner audio to a reference TTS recording
    of the same sentence using DTW on MFCC features, then projects
    the known syllable boundaries from the reference to the learner.

    Assumes:
    - Reference audio has uniform syllable boundaries (TTS is consistent)
    - Learner is attempting to say the same sentence
    """

    def __init__(
        self,
        config: DTWAlignerConfig | None = None,
        feature_extractor: Callable[[NDArray, int], NDArray] | None = None,
    ):
        """Initialize DTW aligner.

        Args:
            config: Configuration options.
            feature_extractor: Optional custom feature extractor function.
        """
        self.config = config or DTWAlignerConfig()
        self._feature_extractor = feature_extractor or extract_mfcc

    @property
    def name(self) -> str:
        return "dtw"

    def align(
        self,
        audio: NDArray[np.floating],
        targets: list[TargetSyllable],
        sr: int = 16000,
    ) -> AlignmentResult:
        """Align using uniform boundaries (no reference).

        For DTW alignment with reference, use align_with_reference().
        """
        # Without reference, fall back to uniform
        return self._align_uniform(audio, targets, sr)

    def align_with_reference(
        self,
        learner_audio: NDArray[np.floating],
        reference_audio: NDArray[np.floating],
        targets: list[TargetSyllable],
        sr: int = 16000,
    ) -> AlignmentResult:
        """Align learner audio using TTS reference.

        Args:
            learner_audio: Learner's audio samples.
            reference_audio: TTS reference audio samples.
            targets: Target syllables.
            sr: Sample rate.

        Returns:
            AlignmentResult with projected syllable spans.
        """
        if not targets:
            return AlignmentResult(
                syllable_spans=[],
                overall_confidence=1.0,
                warnings=["no_targets"],
            )

        # Extract features
        hop_length_ms = self.config.hop_length / sr * 1000
        learner_features = self._feature_extractor(learner_audio, sr)
        ref_features = self._feature_extractor(reference_audio, sr)

        # Run DTW
        cost, path = dtw_alignment(
            learner_features,
            ref_features,
            band_fraction=self.config.band_fraction,
        )

        warnings = []
        if not path:
            warnings.append("dtw_failed")
            return self._align_uniform(learner_audio, targets, sr)

        # Reference boundaries (uniform for TTS)
        ref_duration_ms = int(len(reference_audio) / sr * 1000)
        learner_duration_ms = int(len(learner_audio) / sr * 1000)
        n_syllables = len(targets)
        ref_ms_per_syl = ref_duration_ms / n_syllables

        ref_boundaries = [int(i * ref_ms_per_syl) for i in range(n_syllables + 1)]

        # Project boundaries through warping path
        learner_boundaries = project_boundaries(
            ref_boundaries,
            path,
            hop_length_ms=hop_length_ms,
            query_duration_ms=learner_duration_ms,
        )

        # Ensure boundaries are valid
        learner_boundaries = sorted(set(learner_boundaries))
        if len(learner_boundaries) < n_syllables + 1:
            # Pad with uniform if projection failed
            warnings.append("boundary_projection_incomplete")
            return self._align_uniform(learner_audio, targets, sr)

        # Compute confidence from DTW cost
        # Lower cost = higher confidence
        avg_cost = cost / max(1, len(path))
        confidence = max(self.config.min_confidence, 1.0 - min(1.0, avg_cost))

        # Create spans
        spans = []
        for i, target in enumerate(targets):
            start_ms = learner_boundaries[i]
            end_ms = learner_boundaries[i + 1] if i + 1 < len(learner_boundaries) else learner_duration_ms

            spans.append(SyllableSpan(
                index=i,
                start_ms=Ms(start_ms),
                end_ms=Ms(end_ms),
                confidence=confidence,
                phone_spans=None,
            ))

        return AlignmentResult(
            syllable_spans=spans,
            overall_confidence=confidence,
            warnings=warnings,
        )

    def _align_uniform(
        self,
        audio: NDArray[np.floating],
        targets: list[TargetSyllable],
        sr: int,
    ) -> AlignmentResult:
        """Fallback to uniform alignment."""
        from .uniform import UniformAligner

        uniform = UniformAligner()
        result = uniform.align(audio, targets, sr)
        result.warnings.append("dtw_fallback_to_uniform")
        return result
