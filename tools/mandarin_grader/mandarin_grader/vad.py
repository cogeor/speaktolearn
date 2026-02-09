"""Voice Activity Detection and energy-based syllable segmentation.

This module provides energy-based methods for detecting syllable boundaries
without requiring forced alignment or neural models.
"""

import numpy as np
from numpy.typing import NDArray

from .types import SyllableSpan


def _medfilt(x: NDArray, kernel_size: int = 5) -> NDArray:
    """Simple median filter (pure numpy, no scipy dependency)."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    padded = np.pad(x, pad, mode='edge')
    result = np.zeros_like(x)
    for i in range(len(x)):
        result[i] = np.median(padded[i:i + kernel_size])
    return result


def compute_rms_energy(
    audio: NDArray[np.floating],
    frame_length: int = 400,  # 25ms at 16kHz
    hop_length: int = 160,    # 10ms at 16kHz
) -> NDArray[np.floating]:
    """Compute RMS energy per frame.

    Args:
        audio: Audio samples.
        frame_length: Analysis window length in samples.
        hop_length: Hop between frames in samples.

    Returns:
        RMS energy per frame.
    """
    n_frames = 1 + (len(audio) - frame_length) // hop_length
    if n_frames <= 0:
        return np.array([0.0])

    energy = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        frame = audio[start:end]
        energy[i] = np.sqrt(np.mean(frame ** 2))

    return energy


def find_energy_valleys(
    energy: NDArray[np.floating],
    n_syllables: int,
    min_syllable_frames: int = 5,
    smooth_window: int = 3,
) -> list[int]:
    """Find valley points in energy contour for syllable boundaries.

    Uses a search-window approach: divides audio into expected regions
    and finds the minimum energy point in each boundary region.

    Args:
        energy: RMS energy per frame.
        n_syllables: Expected number of syllables.
        min_syllable_frames: Minimum frames per syllable.
        smooth_window: Median filter window for smoothing.

    Returns:
        List of frame indices for boundaries (n_syllables - 1 boundaries).
    """
    if n_syllables <= 1:
        return []

    n_boundaries = n_syllables - 1
    n_frames = len(energy)

    # Light smoothing to reduce noise (smaller window)
    if smooth_window > 1 and len(energy) >= smooth_window:
        smoothed = _medfilt(energy, kernel_size=smooth_window)
    else:
        smoothed = energy.copy()

    # Estimate syllable length and search window
    avg_syl_frames = n_frames // n_syllables
    search_radius = max(3, avg_syl_frames // 3)  # Search +/- 33% around expected

    boundaries = []
    for i in range(n_boundaries):
        # Expected boundary position (between syllable i and i+1)
        expected = int(n_frames * (i + 1) / n_syllables)

        # Search window around expected position
        search_start = max(min_syllable_frames, expected - search_radius)
        search_end = min(n_frames - min_syllable_frames, expected + search_radius)

        if search_start >= search_end:
            boundaries.append(expected)
            continue

        # Find minimum energy in search window
        search_region = smoothed[search_start:search_end]
        min_idx = search_start + int(np.argmin(search_region))
        boundaries.append(min_idx)

    return boundaries


def find_voicing_gaps(
    voicing: NDArray[np.floating],
    n_syllables: int,
    voicing_threshold: float = 0.5,
    min_gap_frames: int = 2,
    hop_length_ms: int = 10,
) -> list[int]:
    """Find syllable boundaries using voicing gaps.

    Args:
        voicing: Voicing confidence per frame (0-1).
        n_syllables: Expected number of syllables.
        voicing_threshold: Below this = unvoiced.
        min_gap_frames: Minimum gap length to consider.
        hop_length_ms: Frame hop in ms.

    Returns:
        List of boundary frame indices (midpoints of gaps).
    """
    if n_syllables <= 1:
        return []

    n_boundaries = n_syllables - 1

    # Find all gaps (contiguous unvoiced regions)
    gaps = []
    in_gap = False
    gap_start = 0

    for i, v in enumerate(voicing):
        if v < voicing_threshold:
            if not in_gap:
                gap_start = i
                in_gap = True
        else:
            if in_gap:
                gap_len = i - gap_start
                if gap_len >= min_gap_frames:
                    # Store gap midpoint and length
                    midpoint = gap_start + gap_len // 2
                    gaps.append((midpoint, gap_len))
                in_gap = False

    # Handle gap at end
    if in_gap:
        gap_len = len(voicing) - gap_start
        if gap_len >= min_gap_frames:
            midpoint = gap_start + gap_len // 2
            gaps.append((midpoint, gap_len))

    if not gaps:
        # No gaps found, fall back to uniform
        return [int(len(voicing) * (i + 1) / n_syllables)
                for i in range(n_boundaries)]

    # Sort gaps by length (longest first - most likely real boundaries)
    gaps.sort(key=lambda x: x[1], reverse=True)

    # Select top n_boundaries gaps, then sort by position
    boundaries = [g[0] for g in gaps[:n_boundaries]]
    boundaries.sort()

    return boundaries


def segment_by_voicing(
    audio: NDArray[np.floating],
    n_syllables: int,
    sr: int = 16000,
    hop_length_ms: int = 10,
) -> list[SyllableSpan]:
    """Segment audio using voicing gaps from pitch tracking.

    This works better than energy for TTS audio where syllables
    flow together but have brief unvoiced transitions.

    Args:
        audio: Audio samples.
        n_syllables: Expected number of syllables.
        sr: Sample rate.
        hop_length_ms: Hop length in milliseconds.

    Returns:
        List of SyllableSpan with detected boundaries.
    """
    from .pitch import extract_f0_yin

    if n_syllables <= 0:
        return []

    # Extract pitch and voicing
    hop_length = int(sr * hop_length_ms / 1000)
    _, voicing = extract_f0_yin(audio, sr=sr, hop_length=hop_length)

    # Find boundaries at voicing gaps
    boundary_frames = find_voicing_gaps(voicing, n_syllables, hop_length_ms=hop_length_ms)

    # Convert frames to milliseconds
    def frame_to_ms(frame: int) -> int:
        return int(frame * hop_length_ms)

    total_ms = int(len(audio) / sr * 1000)

    # Build spans
    spans = []
    prev_ms = 0

    for i, bf in enumerate(boundary_frames):
        end_ms = frame_to_ms(bf)
        spans.append(SyllableSpan(
            index=i,
            start_ms=prev_ms,
            end_ms=end_ms,
            confidence=0.7,  # Voicing-based has higher confidence
        ))
        prev_ms = end_ms

    # Last syllable
    spans.append(SyllableSpan(
        index=len(boundary_frames),
        start_ms=prev_ms,
        end_ms=total_ms,
        confidence=0.7,
    ))

    return spans


def segment_by_energy(
    audio: NDArray[np.floating],
    n_syllables: int,
    sr: int = 16000,
    frame_length_ms: int = 25,
    hop_length_ms: int = 10,
) -> list[SyllableSpan]:
    """Segment audio into syllables using energy-based boundary detection.

    Args:
        audio: Audio samples.
        n_syllables: Expected number of syllables.
        sr: Sample rate.
        frame_length_ms: Frame length in milliseconds.
        hop_length_ms: Hop length in milliseconds.

    Returns:
        List of SyllableSpan with detected boundaries.
    """
    if n_syllables <= 0:
        return []

    frame_length = int(sr * frame_length_ms / 1000)
    hop_length = int(sr * hop_length_ms / 1000)

    # Compute energy
    energy = compute_rms_energy(audio, frame_length, hop_length)

    # Find boundaries
    boundary_frames = find_energy_valleys(energy, n_syllables)

    # Convert frames to milliseconds
    def frame_to_ms(frame: int) -> int:
        return int(frame * hop_length_ms)

    total_ms = int(len(audio) / sr * 1000)

    # Build spans
    spans = []
    prev_ms = 0

    for i, bf in enumerate(boundary_frames):
        end_ms = frame_to_ms(bf)
        spans.append(SyllableSpan(
            index=i,
            start_ms=prev_ms,
            end_ms=end_ms,
            confidence=0.5,  # Energy-based segmentation has moderate confidence
        ))
        prev_ms = end_ms

    # Last syllable
    spans.append(SyllableSpan(
        index=len(boundary_frames),
        start_ms=prev_ms,
        end_ms=total_ms,
        confidence=0.5,
    ))

    return spans


def trim_silence(
    audio: NDArray[np.floating],
    sr: int = 16000,
    threshold_db: float = -40,
    frame_length_ms: int = 25,
    hop_length_ms: int = 10,
) -> tuple[int, int]:
    """Find start and end of speech, trimming silence.

    Args:
        audio: Audio samples.
        sr: Sample rate.
        threshold_db: Energy threshold in dB below peak.
        frame_length_ms: Frame length in milliseconds.
        hop_length_ms: Hop length in milliseconds.

    Returns:
        Tuple of (start_ms, end_ms) for speech region.
    """
    frame_length = int(sr * frame_length_ms / 1000)
    hop_length = int(sr * hop_length_ms / 1000)

    energy = compute_rms_energy(audio, frame_length, hop_length)

    if len(energy) == 0 or np.max(energy) == 0:
        return (0, int(len(audio) / sr * 1000))

    # Convert to dB
    energy_db = 20 * np.log10(energy / np.max(energy) + 1e-10)

    # Find first and last frame above threshold
    above_threshold = energy_db > threshold_db

    if not np.any(above_threshold):
        return (0, int(len(audio) / sr * 1000))

    first_frame = np.argmax(above_threshold)
    last_frame = len(above_threshold) - 1 - np.argmax(above_threshold[::-1])

    start_ms = int(first_frame * hop_length_ms)
    end_ms = int((last_frame + 1) * hop_length_ms)

    return (start_ms, end_ms)
