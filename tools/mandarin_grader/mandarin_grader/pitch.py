"""Pitch extraction and normalization.

This module provides functions for speaker-independent F0 analysis:
- Extract F0 using pYIN algorithm
- Convert F0 from Hz to semitones
- Compute robust statistics (median, MAD) over voiced frames
- Z-score normalization using robust statistics

The key invariant is that scaling all F0 by a constant (e.g., different speakers)
should not change the normalized contour shape.
"""

import numpy as np
from numpy.typing import NDArray

from .types import FrameTrack


def extract_f0_pyin(
    audio: NDArray[np.floating],
    sr: int = 16000,
    fmin: float = 50.0,
    fmax: float = 500.0,
    hop_length: int = 160,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Extract F0 using pYIN algorithm.

    pYIN is a probabilistic variant of YIN that provides:
    - Frame-level F0 estimates in Hz
    - Voicing probability per frame

    Args:
        audio: Audio samples as float array (mono, normalized to [-1, 1]).
        sr: Sample rate in Hz (default 16000).
        fmin: Minimum F0 frequency in Hz (default 50).
        fmax: Maximum F0 frequency in Hz (default 500).
        hop_length: Hop length in samples (default 160 = 10ms at 16kHz).

    Returns:
        Tuple of (f0_hz, voicing) where:
        - f0_hz: F0 values in Hz, shape [T]. Unvoiced frames have value 0.
        - voicing: Voicing probability per frame, shape [T], range [0, 1].
    """
    import librosa

    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        hop_length=hop_length,
    )

    # Replace NaN with 0 for unvoiced frames
    f0 = np.nan_to_num(f0, nan=0.0)

    # Use voiced_probs if available, otherwise convert flag to float
    voicing = voiced_probs if voiced_probs is not None else voiced_flag.astype(float)

    return f0.astype(np.float64), voicing.astype(np.float64)


def hz_to_semitones(
    f0_hz: NDArray[np.floating],
    ref_hz: float = 100.0,
) -> NDArray[np.floating]:
    """Convert F0 from Hz to semitones relative to reference.

    Formula: st = 12 * log2(f0 / ref_hz)

    Args:
        f0_hz: Array of F0 values in Hz. Zeros and negative values
            represent unvoiced frames.
        ref_hz: Reference frequency in Hz (default 100 Hz).

    Returns:
        Array of semitone values. Unvoiced frames (f0 <= 0) get value 0.
    """
    # Create output array initialized to zero
    result = np.zeros_like(f0_hz)

    # Find voiced frames (positive F0)
    voiced_mask = f0_hz > 0

    # Convert only voiced frames
    if np.any(voiced_mask):
        result[voiced_mask] = 12.0 * np.log2(f0_hz[voiced_mask] / ref_hz)

    return result


def robust_stats(
    values: NDArray[np.floating],
    voicing: NDArray[np.floating],
    voicing_threshold: float = 0.5,
) -> tuple[float, float]:
    """Compute robust statistics (median and MAD) over voiced frames.

    Uses median and median absolute deviation (MAD) as robust estimators
    that are less sensitive to outliers than mean and standard deviation.

    Args:
        values: Array of values (e.g., semitones).
        voicing: Voicing probability per frame (0-1).
        voicing_threshold: Minimum voicing to consider a frame voiced.

    Returns:
        Tuple of (median, mad) where MAD = median absolute deviation.
        If no voiced frames, returns (0.0, 1.0) to avoid division by zero.
    """
    # Find voiced frames
    voiced_mask = voicing >= voicing_threshold
    voiced_values = values[voiced_mask]

    # Handle case with no voiced frames
    if len(voiced_values) == 0:
        return (0.0, 1.0)

    # Compute median
    median = float(np.median(voiced_values))

    # Compute MAD (median absolute deviation)
    mad = float(np.median(np.abs(voiced_values - median)))

    # Avoid zero MAD (can happen with constant values)
    if mad == 0.0:
        mad = 1.0

    return (median, mad)


def normalize_f0(
    semitones: NDArray[np.floating],
    voicing: NDArray[np.floating],
    voicing_threshold: float = 0.5,
) -> NDArray[np.floating]:
    """Speaker-normalize F0 using z-score with robust statistics.

    Normalization process:
    1. Compute median and MAD over voiced frames
    2. Apply z-score: (semitones - median) / (1.4826 * MAD)
       The factor 1.4826 scales MAD to be consistent with standard
       deviation for a normal distribution.
    3. Unvoiced frames get value 0

    Key invariant: Scaling all F0 by a constant (different speakers)
    should not change the normalized contour shape.

    Args:
        semitones: Array of F0 values in semitones.
        voicing: Voicing probability per frame (0-1).
        voicing_threshold: Minimum voicing to consider a frame voiced.

    Returns:
        Normalized F0 array. Unvoiced frames have value 0.
    """
    # Get robust statistics
    median, mad = robust_stats(semitones, voicing, voicing_threshold)

    # Scale factor to make MAD consistent with std dev for normal distribution
    scale = 1.4826 * mad

    # Create output array initialized to zero (for unvoiced frames)
    result = np.zeros_like(semitones)

    # Find voiced frames
    voiced_mask = voicing >= voicing_threshold

    # Normalize only voiced frames
    if np.any(voiced_mask):
        result[voiced_mask] = (semitones[voiced_mask] - median) / scale

    return result


def normalize_frame_track(track: FrameTrack) -> NDArray[np.floating]:
    """Convert FrameTrack to normalized F0 array.

    This is a convenience function that applies the full normalization
    pipeline to a FrameTrack:
    1. Convert Hz to semitones
    2. Apply speaker normalization using robust statistics

    Args:
        track: FrameTrack containing f0_hz and voicing arrays.

    Returns:
        Normalized F0 array with the same length as the input track.
    """
    # Convert to semitones
    semitones = hz_to_semitones(track.f0_hz)

    # Normalize using robust statistics
    normalized = normalize_f0(semitones, track.voicing)

    return normalized
