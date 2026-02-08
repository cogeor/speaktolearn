"""Contour extraction for syllable-level tone analysis.

This module provides functions for extracting and processing pitch contours
from syllable spans within audio tracks. The key operations are:
- Slicing frame tracks to syllable boundaries
- Resampling variable-length contours to fixed K points
- Computing derivatives for shape analysis
- Computing voicing statistics

The K-point representation allows comparing contours of different durations.
"""

import numpy as np
from numpy.typing import NDArray

from .pitch import normalize_frame_track
from .types import Contour, FrameTrack, SyllableSpan


def ms_to_frame(ms: int, frame_hz: float) -> int:
    """Convert milliseconds to frame index.

    Args:
        ms: Time in milliseconds.
        frame_hz: Frame rate in Hz (frames per second).

    Returns:
        Frame index (integer).
    """
    return int(ms * frame_hz / 1000)


def compute_voicing_ratio(
    voicing: NDArray[np.floating],
    voicing_threshold: float = 0.5,
) -> float:
    """Compute fraction of frames that are voiced.

    Args:
        voicing: Voicing probability per frame (0-1).
        voicing_threshold: Minimum voicing to consider a frame voiced.

    Returns:
        Fraction of voiced frames (0-1). Returns 0 if array is empty.
    """
    if len(voicing) == 0:
        return 0.0

    voiced_count = np.sum(voicing >= voicing_threshold)
    return float(voiced_count / len(voicing))


def resample_contour(
    f0_norm: NDArray[np.floating],
    voicing: NDArray[np.floating],
    k: int = 20,
    voicing_threshold: float = 0.5,
) -> NDArray[np.floating]:
    """Resample variable-length contour to fixed K points.

    The resampling process:
    1. Extract only voiced frames (voicing >= threshold)
    2. Linear interpolation to K evenly-spaced points
    3. Zero-pad if fewer than 3 voiced frames (insufficient for interpolation)

    This allows comparing contours of different durations by normalizing
    them to the same length.

    Args:
        f0_norm: Normalized F0 values per frame.
        voicing: Voicing probability per frame (0-1).
        k: Number of output points (default 20).
        voicing_threshold: Minimum voicing to consider a frame voiced.

    Returns:
        Array of shape [K] with resampled contour values.
    """
    # Find voiced frames
    voiced_mask = voicing >= voicing_threshold
    voiced_values = f0_norm[voiced_mask]

    # If fewer than 3 voiced frames, return zeros (insufficient for interpolation)
    if len(voiced_values) < 3:
        return np.zeros(k, dtype=np.float64)

    # Create interpolation points
    # Original indices normalized to [0, 1]
    original_indices = np.linspace(0, 1, len(voiced_values))
    # Target indices for K points
    target_indices = np.linspace(0, 1, k)

    # Linear interpolation
    resampled = np.interp(target_indices, original_indices, voiced_values)

    return resampled.astype(np.float64)


def compute_derivatives(
    f0_norm: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute first and second derivatives of contour.

    Uses np.gradient for differentiation, which computes second-order
    accurate central differences in the interior and first or second-order
    accurate one-sided differences at the boundaries.

    Args:
        f0_norm: Normalized F0 contour of shape [K].

    Returns:
        Tuple of (df0, ddf0) where:
        - df0: First derivative (velocity), shape [K]
        - ddf0: Second derivative (acceleration), shape [K]
    """
    # First derivative (velocity)
    df0 = np.gradient(f0_norm)

    # Second derivative (acceleration)
    ddf0 = np.gradient(df0)

    return df0.astype(np.float64), ddf0.astype(np.float64)


def extract_contour(
    track: FrameTrack,
    span: SyllableSpan,
    k: int = 20,
) -> Contour:
    """Extract Contour for a syllable span from a FrameTrack.

    The extraction process:
    1. Convert span boundaries from ms to frame indices
    2. Slice the track to span boundaries
    3. Normalize the F0 slice (Hz -> semitones -> z-score)
    4. Resample to K fixed points
    5. Compute first and second derivatives
    6. Calculate duration and voicing ratio

    Args:
        track: FrameTrack containing f0_hz and voicing arrays.
        span: SyllableSpan with start_ms and end_ms boundaries.
        k: Number of points for resampled contour (default 20).

    Returns:
        Contour dataclass with normalized features.
    """
    # Convert milliseconds to frame indices
    start_frame = ms_to_frame(span.start_ms, track.frame_hz)
    end_frame = ms_to_frame(span.end_ms, track.frame_hz)

    # Clamp to valid range
    start_frame = max(0, start_frame)
    end_frame = min(len(track.f0_hz), end_frame)

    # Handle edge case of empty or inverted span
    if end_frame <= start_frame:
        return Contour(
            f0_norm=np.zeros(k, dtype=np.float64),
            df0=np.zeros(k, dtype=np.float64),
            ddf0=np.zeros(k, dtype=np.float64),
            duration_ms=span.end_ms - span.start_ms,
            voicing_ratio=0.0,
        )

    # Slice the track to span boundaries
    f0_slice = track.f0_hz[start_frame:end_frame]
    voicing_slice = track.voicing[start_frame:end_frame]

    # Create a temporary FrameTrack for normalization
    slice_track = FrameTrack(
        frame_hz=track.frame_hz,
        f0_hz=f0_slice,
        voicing=voicing_slice,
        energy=None,
    )

    # Normalize the F0 slice
    f0_norm_full = normalize_frame_track(slice_track)

    # Resample to K points
    f0_norm = resample_contour(f0_norm_full, voicing_slice, k=k)

    # Compute derivatives
    df0, ddf0 = compute_derivatives(f0_norm)

    # Calculate duration and voicing ratio
    duration_ms = span.end_ms - span.start_ms
    voicing_ratio = compute_voicing_ratio(voicing_slice)

    return Contour(
        f0_norm=f0_norm,
        df0=df0,
        ddf0=ddf0,
        duration_ms=duration_ms,
        voicing_ratio=voicing_ratio,
    )
