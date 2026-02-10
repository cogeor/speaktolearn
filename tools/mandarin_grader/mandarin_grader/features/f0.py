"""F0 (fundamental frequency) extraction for tone classification.

F0 is the pitch of the voice - the vibration rate of vocal cords.
Mandarin tones are defined by F0 contour (how pitch changes over time).

This module provides F0 extraction using multiple methods:
1. Autocorrelation (simple, fast)
2. YIN algorithm (more accurate)
3. PYIN (probabilistic, handles unvoiced regions)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def extract_f0_autocorr(
    audio: NDArray[np.float32],
    sr: int = 16000,
    hop_length: int = 160,
    fmin: float = 75.0,
    fmax: float = 500.0,
) -> NDArray[np.float32]:
    """Extract F0 using autocorrelation method.

    Simple and fast, but less accurate than YIN.

    Args:
        audio: Audio samples [-1, 1]
        sr: Sample rate
        hop_length: Samples between frames (160 = 10ms at 16kHz)
        fmin: Minimum F0 in Hz
        fmax: Maximum F0 in Hz

    Returns:
        F0 contour [n_frames], 0 for unvoiced frames
    """
    # Frame parameters
    frame_length = int(sr / fmin)  # Need at least one period
    n_frames = 1 + (len(audio) - frame_length) // hop_length

    # Period range in samples
    min_period = int(sr / fmax)
    max_period = int(sr / fmin)

    f0 = np.zeros(n_frames, dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start:start + frame_length]

        if len(frame) < frame_length:
            continue

        # Normalize frame
        frame = frame - frame.mean()
        if frame.std() < 1e-6:
            continue

        # Autocorrelation
        autocorr = np.correlate(frame, frame, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]

        # Find peak in valid period range
        search_region = autocorr[min_period:max_period]
        if len(search_region) == 0:
            continue

        peak_idx = np.argmax(search_region) + min_period

        # Check if peak is significant (voiced)
        if autocorr[peak_idx] > 0.3 * autocorr[0]:
            f0[i] = sr / peak_idx

    return f0


def extract_f0_yin(
    audio: NDArray[np.float32],
    sr: int = 16000,
    hop_length: int = 160,
    frame_length: int = 1024,
    fmin: float = 75.0,
    fmax: float = 500.0,
    threshold: float = 0.1,
) -> NDArray[np.float32]:
    """Extract F0 using YIN algorithm.

    More accurate than autocorrelation, handles harmonics better.

    Args:
        audio: Audio samples [-1, 1]
        sr: Sample rate
        hop_length: Samples between frames
        frame_length: Analysis window size
        fmin: Minimum F0 in Hz
        fmax: Maximum F0 in Hz
        threshold: Aperiodicity threshold (lower = stricter voicing)

    Returns:
        F0 contour [n_frames], 0 for unvoiced frames
    """
    n_frames = 1 + (len(audio) - frame_length) // hop_length

    # Period range in samples
    tau_min = int(sr / fmax)
    tau_max = int(sr / fmin)

    f0 = np.zeros(n_frames, dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start:start + frame_length]

        if len(frame) < frame_length:
            continue

        # Step 1: Difference function
        # d(tau) = sum((x[j] - x[j+tau])^2) for j in window
        W = frame_length // 2
        d = np.zeros(tau_max + 1)

        for tau in range(1, tau_max + 1):
            diff = frame[:W] - frame[tau:tau + W]
            d[tau] = np.sum(diff ** 2)

        # Step 2: Cumulative mean normalized difference
        # d'(tau) = d(tau) / ((1/tau) * sum(d[1:tau]))
        d_prime = np.ones(tau_max + 1)
        cumsum = 0
        for tau in range(1, tau_max + 1):
            cumsum += d[tau]
            if cumsum > 0:
                d_prime[tau] = d[tau] * tau / cumsum

        # Step 3: Absolute threshold
        # Find first tau where d'(tau) < threshold
        best_tau = 0
        for tau in range(tau_min, tau_max + 1):
            if d_prime[tau] < threshold:
                # Step 4: Parabolic interpolation for subframe accuracy
                if tau > 0 and tau < len(d_prime) - 1:
                    alpha = d_prime[tau - 1]
                    beta = d_prime[tau]
                    gamma = d_prime[tau + 1]
                    denom = 2 * (2 * beta - alpha - gamma)
                    if abs(denom) > 1e-9:
                        delta = (alpha - gamma) / denom
                        best_tau = tau + delta
                    else:
                        best_tau = tau
                else:
                    best_tau = tau
                break

        if best_tau > 0:
            f0[i] = sr / best_tau

    return f0


def normalize_f0(
    f0: NDArray[np.float32],
    method: str = "log_zscore",
) -> NDArray[np.float32]:
    """Normalize F0 contour for neural network input.

    Args:
        f0: Raw F0 in Hz, 0 for unvoiced
        method: Normalization method
            - "log_zscore": Log transform then z-score (recommended)
            - "semitone": Convert to semitones relative to mean
            - "minmax": Scale to [0, 1]

    Returns:
        Normalized F0 contour
    """
    # Make a copy
    f0_norm = f0.copy()

    # Get voiced frames
    voiced_mask = f0 > 0
    if not voiced_mask.any():
        return np.zeros_like(f0)

    voiced_f0 = f0[voiced_mask]

    if method == "log_zscore":
        # Log transform (pitch perception is logarithmic)
        log_f0 = np.log(voiced_f0 + 1e-8)
        mean = log_f0.mean()
        std = log_f0.std() + 1e-8

        # Normalize voiced frames
        f0_norm[voiced_mask] = (np.log(f0[voiced_mask] + 1e-8) - mean) / std
        # Mark unvoiced as special value
        f0_norm[~voiced_mask] = 0

    elif method == "semitone":
        # Convert to semitones relative to mean F0
        ref_f0 = np.median(voiced_f0)
        f0_norm[voiced_mask] = 12 * np.log2(f0[voiced_mask] / ref_f0 + 1e-8)
        f0_norm[~voiced_mask] = 0

    elif method == "minmax":
        # Scale to [0, 1]
        f0_min = voiced_f0.min()
        f0_max = voiced_f0.max()
        if f0_max > f0_min:
            f0_norm[voiced_mask] = (f0[voiced_mask] - f0_min) / (f0_max - f0_min)
        else:
            f0_norm[voiced_mask] = 0.5
        f0_norm[~voiced_mask] = 0

    return f0_norm.astype(np.float32)


def extract_f0_features(
    audio: NDArray[np.float32],
    sr: int = 16000,
    hop_length: int = 160,
    frame_length: int = 1024,
    method: str = "yin",
    normalize: str = "log_zscore",
) -> NDArray[np.float32]:
    """Extract normalized F0 features for tone classification.

    Args:
        audio: Audio samples
        sr: Sample rate
        hop_length: Hop length in samples
        frame_length: Frame length for analysis
        method: F0 extraction method ("yin" or "autocorr")
        normalize: Normalization method

    Returns:
        Normalized F0 contour [n_frames]
    """
    if method == "yin":
        f0 = extract_f0_yin(audio, sr, hop_length, frame_length)
    else:
        f0 = extract_f0_autocorr(audio, sr, hop_length)

    return normalize_f0(f0, method=normalize)


def extract_delta_f0(f0: NDArray[np.float32]) -> NDArray[np.float32]:
    """Extract F0 delta (velocity) features.

    Delta F0 captures the direction and rate of pitch change,
    which is crucial for distinguishing rising vs falling tones.

    Args:
        f0: F0 contour [n_frames]

    Returns:
        Delta F0 [n_frames]
    """
    # Simple first difference with padding
    delta = np.zeros_like(f0)
    delta[1:] = f0[1:] - f0[:-1]
    return delta.astype(np.float32)


def extract_full_f0_features(
    audio: NDArray[np.float32],
    sr: int = 16000,
    hop_length: int = 160,
) -> NDArray[np.float32]:
    """Extract F0 and delta-F0 features stacked.

    Returns:
        Features [2, n_frames] - row 0 is F0, row 1 is delta-F0
    """
    f0 = extract_f0_features(audio, sr, hop_length)
    delta = extract_delta_f0(f0)

    return np.stack([f0, delta], axis=0).astype(np.float32)
