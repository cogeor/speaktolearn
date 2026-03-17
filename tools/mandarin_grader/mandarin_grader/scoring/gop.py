"""CTC-GOP (Goodness of Pronunciation) proxy scoring for V7 CTC model.

GOP scoring estimates pronunciation quality by measuring how well
the acoustic evidence at aligned frames supports the target phoneme/syllable.

For CTC models, we use a proxy GOP:
    GOP(s) = mean(log P(s | frame_t)) for frames aligned to syllable s

Where P(s | frame_t) = softmax(logits[t])[s].
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def ctc_greedy_decode_with_alignment(
    logits: NDArray[np.float32],
    blank_index: int = 0,
) -> Tuple[List[int], List[int]]:
    """Greedy CTC decoding that also returns per-token frame alignment.

    For each non-blank token in the collapsed sequence, records the frame
    index where it first appeared (the transition frame).

    Args:
        logits: Frame-level logits [time, n_classes]. A single sequence
                (not batched).
        blank_index: Index of the CTC blank token (default 0).

    Returns:
        Tuple of:
        - decoded_ids: List of token IDs after CTC collapse (no blanks).
        - aligned_frames: List of frame indices, one per decoded token,
          indicating the frame where that token's run began.
    """
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D logits [time, n_classes], got shape {logits.shape}")

    # Greedy argmax per frame
    frame_tokens: NDArray[np.int64] = np.argmax(logits, axis=-1)

    decoded_ids: List[int] = []
    aligned_frames: List[int] = []
    prev_token: Optional[int] = None

    for t, token in enumerate(frame_tokens.tolist()):
        if token != prev_token:
            if token != blank_index:
                decoded_ids.append(token)
                aligned_frames.append(t)
            prev_token = token

    return decoded_ids, aligned_frames


def _log_softmax(logits: NDArray[np.float32]) -> NDArray[np.float32]:
    """Numerically stable log-softmax along the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    return shifted - log_sum_exp


def compute_ctc_gop(
    syllable_logits: NDArray[np.float32],
    tone_logits: NDArray[np.float32],
    target_syllable_ids: List[int],
    target_tone_ids: List[int],
    alignment: Optional[List[Tuple[int, int]]] = None,
    blank_index: int = 0,
    min_frames_per_syllable: int = 1,
) -> Tuple[List[float], List[float]]:
    """Compute CTC-GOP proxy scores for syllables and tones.

    For each target syllable/tone, finds the frames aligned to it and
    computes the average log posterior probability of the target class
    over those frames.

    If no alignment is provided, frames are divided evenly among targets.

    Args:
        syllable_logits: Per-frame syllable logits [time, n_syllables+1].
        tone_logits: Per-frame tone logits [time, n_tones+1].
        target_syllable_ids: Target syllable class IDs, one per syllable
                             (1-indexed; index 0 is the CTC blank).
        target_tone_ids: Target tone class IDs, one per syllable.
        alignment: Optional list of (start_frame, end_frame) tuples, one
                   per syllable.  end_frame is exclusive.  When None,
                   frames are split evenly across syllables.
        blank_index: Index of the CTC blank class.
        min_frames_per_syllable: Minimum number of frames to assign per
                                 syllable when building a uniform alignment.

    Returns:
        Tuple of:
        - syllable_gop_scores: Per-syllable GOP score (average log posterior).
          Higher (closer to 0) means better pronunciation.
        - tone_gop_scores: Per-syllable tone GOP score.
    """
    if syllable_logits.ndim != 2:
        raise ValueError(
            f"syllable_logits must be 2D [time, n_classes], got {syllable_logits.shape}"
        )
    if tone_logits.ndim != 2:
        raise ValueError(
            f"tone_logits must be 2D [time, n_classes], got {tone_logits.shape}"
        )

    n_frames = syllable_logits.shape[0]
    n_targets = len(target_syllable_ids)

    if n_targets == 0:
        return [], []

    # Build alignment if not provided
    if alignment is None:
        alignment = _build_uniform_alignment(n_frames, n_targets, min_frames_per_syllable)

    # Pre-compute log posteriors
    syl_log_post = _log_softmax(syllable_logits)   # [time, n_syl+1]
    tone_log_post = _log_softmax(tone_logits)       # [time, n_tone+1]

    syllable_gop: List[float] = []
    tone_gop: List[float] = []

    for i, (syl_id, tone_id) in enumerate(zip(target_syllable_ids, target_tone_ids)):
        start_frame, end_frame = alignment[i]

        # Clamp to valid range
        start_frame = max(0, start_frame)
        end_frame = min(n_frames, end_frame)

        if start_frame >= end_frame:
            # Degenerate segment — assign a very low score
            syllable_gop.append(float(syl_log_post[min(start_frame, n_frames - 1), syl_id]))
            tone_gop.append(float(tone_log_post[min(start_frame, n_frames - 1), tone_id]))
            continue

        syl_score = float(np.mean(syl_log_post[start_frame:end_frame, syl_id]))
        tone_score = float(np.mean(tone_log_post[start_frame:end_frame, tone_id]))

        syllable_gop.append(syl_score)
        tone_gop.append(tone_score)

    return syllable_gop, tone_gop


def _build_uniform_alignment(
    n_frames: int,
    n_targets: int,
    min_frames: int = 1,
) -> List[Tuple[int, int]]:
    """Divide n_frames evenly across n_targets segments.

    Args:
        n_frames: Total number of frames.
        n_targets: Number of target segments.
        min_frames: Minimum frames per segment.

    Returns:
        List of (start, end) frame index tuples (end is exclusive).
    """
    # Ensure we have enough frames
    frames_per_target = max(min_frames, n_frames // n_targets)

    segments: List[Tuple[int, int]] = []
    start = 0
    for i in range(n_targets):
        if i == n_targets - 1:
            # Last segment gets any remainder
            end = n_frames
        else:
            end = min(start + frames_per_target, n_frames)
        segments.append((start, end))
        start = end

    return segments
