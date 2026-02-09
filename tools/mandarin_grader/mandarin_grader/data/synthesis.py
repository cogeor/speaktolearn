"""Sentence synthesis via syllable concatenation for synthetic data generation.

This module provides functions to create synthetic training sentences by
concatenating individual syllable audio files with known, exact boundaries.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..types import Ms, TargetSyllable, SyllableSpan, Tone
from .lexicon import SyllableLexicon, SyllableEntry, _remove_tone_marks


@dataclass
class AugmentationConfig:
    """Configuration for audio augmentations."""

    speed_variation: float = 0.0  # ±fraction, e.g., 0.1 = ±10%
    pitch_shift_semitones: float = 0.0  # Global pitch shift
    noise_snr_db: float | None = None  # SNR in dB, None = no noise
    gap_ms: int = 0  # Silence gap between syllables
    crossfade_ms: int = 0  # Crossfade overlap between syllables


@dataclass
class SyntheticSample:
    """A synthesized sentence with ground-truth labels."""

    id: str
    audio: NDArray[np.int16]  # Audio samples (int16, 16kHz mono)
    sample_rate: int
    syllables: list[TargetSyllable]
    ground_truth_spans: list[SyllableSpan]
    augmentations: AugmentationConfig = field(default_factory=AugmentationConfig)
    voice_id: str = "female"


def load_syllable_audio(
    lexicon: SyllableLexicon,
    pinyin: str,
    tone: Tone,
    voice: str = "female",
) -> NDArray[np.float32] | None:
    """Load a syllable audio file from the lexicon.

    Args:
        lexicon: SyllableLexicon instance
        pinyin: Base pinyin (without tone mark)
        tone: Tone number 0-4
        voice: Voice identifier

    Returns:
        Audio as float32 array normalized to [-1, 1], or None if not found
    """
    audio_path = lexicon.get_audio_path(pinyin, tone, voice)
    if audio_path is None or not audio_path.exists():
        return None

    # Read WAV file
    with wave.open(str(audio_path), 'rb') as wf:
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)
        sample_width = wf.getsampwidth()

    # Convert to float32
    if sample_width == 2:
        import struct
        audio = np.array(struct.unpack(f'{n_frames}h', raw_data), dtype=np.float32)
        audio = audio / 32768.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    return audio


def change_speed(audio: NDArray[np.float32], factor: float) -> NDArray[np.float32]:
    """Change audio speed without pitch change (simple resampling).

    Args:
        audio: Input audio array
        factor: Speed factor (>1 = faster, <1 = slower)

    Returns:
        Resampled audio
    """
    if abs(factor - 1.0) < 0.001:
        return audio

    # Simple linear interpolation resampling
    original_length = len(audio)
    new_length = int(original_length / factor)

    if new_length < 2:
        return audio

    indices = np.linspace(0, original_length - 1, new_length)
    resampled = np.interp(indices, np.arange(original_length), audio)

    return resampled.astype(np.float32)


def add_noise(audio: NDArray[np.float32], snr_db: float) -> NDArray[np.float32]:
    """Add Gaussian noise to audio at specified SNR.

    Args:
        audio: Input audio array
        snr_db: Signal-to-noise ratio in dB

    Returns:
        Audio with added noise
    """
    # Calculate signal power
    signal_power = np.mean(audio ** 2)
    if signal_power < 1e-10:
        return audio

    # Calculate noise power from SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    # Generate noise
    noise = np.random.randn(len(audio)).astype(np.float32) * np.sqrt(noise_power)

    return audio + noise


def crossfade(
    audio1: NDArray[np.float32],
    audio2: NDArray[np.float32],
    overlap_samples: int,
) -> NDArray[np.float32]:
    """Crossfade two audio segments.

    Args:
        audio1: First audio segment
        audio2: Second audio segment
        overlap_samples: Number of samples to overlap

    Returns:
        Combined audio with crossfade
    """
    if overlap_samples <= 0 or len(audio1) < overlap_samples or len(audio2) < overlap_samples:
        return np.concatenate([audio1, audio2])

    # Create fade curves
    fade_out = np.linspace(1, 0, overlap_samples).astype(np.float32)
    fade_in = np.linspace(0, 1, overlap_samples).astype(np.float32)

    # Apply crossfade
    result_length = len(audio1) + len(audio2) - overlap_samples
    result = np.zeros(result_length, dtype=np.float32)

    # Copy first segment
    result[:len(audio1) - overlap_samples] = audio1[:-overlap_samples]

    # Crossfade region
    cf_start = len(audio1) - overlap_samples
    result[cf_start:cf_start + overlap_samples] = (
        audio1[-overlap_samples:] * fade_out + audio2[:overlap_samples] * fade_in
    )

    # Copy second segment
    result[cf_start + overlap_samples:] = audio2[overlap_samples:]

    return result


def synthesize_sentence(
    syllables: list[TargetSyllable],
    lexicon: SyllableLexicon,
    voice: str = "female",
    augmentations: AugmentationConfig | None = None,
    sample_rate: int = 16000,
) -> tuple[NDArray[np.float32], list[SyllableSpan]]:
    """Synthesize a sentence by concatenating syllable audio files.

    Args:
        syllables: List of target syllables with pinyin and tone
        lexicon: SyllableLexicon with audio files
        voice: Voice identifier ("female" or "male")
        augmentations: Optional augmentation configuration
        sample_rate: Sample rate (default 16kHz)

    Returns:
        Tuple of (audio_array, list[SyllableSpan])
        The spans have exact ground-truth boundaries from concatenation.
    """
    if augmentations is None:
        augmentations = AugmentationConfig()

    audio_parts: list[NDArray[np.float32]] = []
    spans: list[SyllableSpan] = []
    current_sample = 0

    # Calculate crossfade and gap in samples
    crossfade_samples = int(augmentations.crossfade_ms * sample_rate / 1000)
    gap_samples = int(augmentations.gap_ms * sample_rate / 1000)

    for i, syl in enumerate(syllables):
        # Get base pinyin (remove tone marks)
        base_pinyin = _remove_tone_marks(syl.pinyin)
        tone = syl.tone_surface  # Use surface tone after sandhi

        # Load syllable audio
        syl_audio = load_syllable_audio(lexicon, base_pinyin, tone, voice)

        if syl_audio is None:
            # Try with underlying tone if surface tone not found
            syl_audio = load_syllable_audio(lexicon, base_pinyin, syl.tone_underlying, voice)

        if syl_audio is None:
            # Skip missing syllables (or could raise error)
            continue

        # Apply per-syllable speed variation
        if augmentations.speed_variation > 0:
            speed_factor = 1.0 + np.random.uniform(
                -augmentations.speed_variation,
                augmentations.speed_variation
            )
            syl_audio = change_speed(syl_audio, speed_factor)

        # Record span start
        start_sample = current_sample

        # Handle crossfade with previous segment
        if i > 0 and crossfade_samples > 0 and len(audio_parts) > 0:
            # Crossfade this syllable with the previous one
            prev_audio = audio_parts.pop()
            combined = crossfade(prev_audio, syl_audio, crossfade_samples)
            audio_parts.append(combined)

            # Adjust current position
            current_sample = sum(len(a) for a in audio_parts)
            end_sample = current_sample

            # Update span start to account for crossfade
            start_sample = start_sample - crossfade_samples
        else:
            # Add gap before syllable (except first)
            if i > 0 and gap_samples > 0:
                gap = np.zeros(gap_samples, dtype=np.float32)
                audio_parts.append(gap)
                current_sample += gap_samples
                start_sample = current_sample

            audio_parts.append(syl_audio)
            current_sample += len(syl_audio)
            end_sample = current_sample

        # Create span with exact boundaries
        span = SyllableSpan(
            index=i,
            start_ms=Ms(int(start_sample / sample_rate * 1000)),
            end_ms=Ms(int(end_sample / sample_rate * 1000)),
            confidence=1.0,  # Ground truth has perfect confidence
        )
        spans.append(span)

    # Concatenate all parts
    if not audio_parts:
        return np.array([], dtype=np.float32), []

    full_audio = np.concatenate(audio_parts)

    # Apply global augmentations
    if augmentations.pitch_shift_semitones != 0:
        # Simple pitch shift via resampling (changes duration slightly)
        # For proper pitch shift, would need more sophisticated algorithm
        pitch_factor = 2 ** (augmentations.pitch_shift_semitones / 12)
        full_audio = change_speed(full_audio, pitch_factor)

    if augmentations.noise_snr_db is not None:
        full_audio = add_noise(full_audio, augmentations.noise_snr_db)

    return full_audio, spans


def create_synthetic_sample(
    sample_id: str,
    syllables: list[TargetSyllable],
    lexicon: SyllableLexicon,
    voice: str = "female",
    augmentations: AugmentationConfig | None = None,
    sample_rate: int = 16000,
) -> SyntheticSample:
    """Create a complete synthetic sample with audio and labels.

    Args:
        sample_id: Unique identifier for this sample
        syllables: List of target syllables
        lexicon: SyllableLexicon with audio files
        voice: Voice identifier
        augmentations: Optional augmentation configuration
        sample_rate: Sample rate

    Returns:
        SyntheticSample with audio, syllables, and ground-truth spans
    """
    if augmentations is None:
        augmentations = AugmentationConfig()

    audio_float, spans = synthesize_sentence(
        syllables=syllables,
        lexicon=lexicon,
        voice=voice,
        augmentations=augmentations,
        sample_rate=sample_rate,
    )

    # Convert to int16
    audio_int16 = (audio_float * 32767).clip(-32768, 32767).astype(np.int16)

    return SyntheticSample(
        id=sample_id,
        audio=audio_int16,
        sample_rate=sample_rate,
        syllables=syllables,
        ground_truth_spans=spans,
        augmentations=augmentations,
        voice_id=voice,
    )


def save_synthetic_sample(sample: SyntheticSample, output_path: Path) -> None:
    """Save a synthetic sample to WAV file.

    Args:
        sample: SyntheticSample to save
        output_path: Path for output WAV file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(output_path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample.sample_rate)
        wf.writeframes(sample.audio.tobytes())
