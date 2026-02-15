"""Full-sentence dataset for V5 syllable+tone prediction training.

This module provides a dataset class that creates training samples for the
SyllablePredictorV5 model. Each sample consists of:
- Full sentence audio (mel spectrogram, up to 10s)
- Position index indicating which syllable to predict
- Target syllable and tone

Unlike V4's AutoregressiveDataset which extracts 1s chunks, this dataset
uses the full sentence audio and relies on position embeddings to tell
the model which syllable to predict.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..types import Tone
from .lexicon import _remove_tone_marks
from .autoregressive_dataset import (
    SyntheticSentenceInfo,
    load_audio_wav,
    spec_augment,
)
from .augmentation import pitch_shift, formant_shift


@dataclass
class FullSentenceSample:
    """A single training sample for V5 full-sentence prediction."""

    sample_id: str
    audio_full: NDArray[np.float32] | None  # [n_samples] raw audio for full sentence
    mel_full: NDArray[np.float32] | None  # [n_mels, time] precomputed mel
    sample_rate: int
    position: int  # Which syllable position to predict (0-indexed)
    target_syllable: str  # Base pinyin of target syllable
    target_tone: Tone  # Tone of target syllable
    n_syllables: int  # Total syllables in sentence


class FullSentenceDataset:
    """Dataset for V5 full-sentence syllable+tone prediction.

    Creates training samples by using full sentence audio with position
    indices indicating which syllable to predict.

    Each sample includes:
    - Full sentence mel spectrogram
    - Position index (0 to n_syllables-1)
    - Target: syllable ID + tone

    This matches V5 inference where we have full audio and want to
    predict each syllable by position.
    """

    def __init__(
        self,
        sentences: list[SyntheticSentenceInfo],
        sample_rate: int = 16000,
        max_duration_s: float = 10.0,
        augment: bool = True,
        noise_snr_db: float | None = 30.0,
        speed_variation: float = 0.1,
        volume_variation_db: float = 12.0,
        pitch_shift_semitones: float = 0.0,
        formant_shift_percent: float = 0.0,
    ):
        """Initialize dataset.

        Args:
            sentences: List of sentence info
            sample_rate: Audio sample rate
            max_duration_s: Maximum audio duration in seconds
            augment: Whether to apply augmentation
            noise_snr_db: SNR for noise augmentation (None = no noise)
            speed_variation: Speed variation fraction (0.1 = ±10%)
            volume_variation_db: Volume variation in dB
            pitch_shift_semitones: Max pitch shift in semitones (±)
            formant_shift_percent: Max formant shift in percent (±)
        """
        self.sentences = sentences
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_s * sample_rate)
        self.augment = augment
        self.noise_snr_db = noise_snr_db
        self.speed_variation = speed_variation
        self.volume_variation_db = volume_variation_db
        self.pitch_shift_semitones = pitch_shift_semitones
        self.formant_shift_percent = formant_shift_percent

        # Build index: list of (sentence_idx, syllable_idx) pairs
        self._index = []
        for sent_idx, sent in enumerate(sentences):
            for syl_idx in range(len(sent.syllables)):
                self._index.append((sent_idx, syl_idx))

        # Audio cache
        self._audio_cache: dict[str, NDArray[np.float32]] = {}
        # Optional precomputed mel cache
        self._mel_cache: dict[str, NDArray[np.float32]] = {}

    def __len__(self) -> int:
        return len(self._index)

    def preload_audio(self, progress_callback=None) -> None:
        """Pre-load all audio files into memory cache."""
        total = len(self.sentences)
        for i, sent in enumerate(self.sentences):
            key = str(sent.audio_path)
            if key not in self._audio_cache:
                self._audio_cache[key] = load_audio_wav(sent.audio_path, self.sample_rate)
            if progress_callback and (i + 1) % 1000 == 0:
                progress_callback(i + 1, total)
        if progress_callback:
            progress_callback(total, total)

    def _load_audio(self, path: Path) -> NDArray[np.float32]:
        """Load audio with caching."""
        key = str(path)
        if key not in self._audio_cache:
            self._audio_cache[key] = load_audio_wav(path, self.sample_rate)
        return self._audio_cache[key]

    def _apply_augmentation(self, audio: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply audio augmentation."""
        if not self.augment:
            return audio

        # 1. Pitch shift
        if self.pitch_shift_semitones > 0:
            semitones = np.random.uniform(
                -self.pitch_shift_semitones, self.pitch_shift_semitones
            )
            if abs(semitones) > 0.1:
                audio = pitch_shift(audio, semitones, sr=self.sample_rate)

        # 2. Formant shift
        if self.formant_shift_percent > 0:
            shift_ratio = 1.0 + np.random.uniform(
                -self.formant_shift_percent, self.formant_shift_percent
            ) / 100.0
            if abs(shift_ratio - 1.0) > 0.01:
                audio = formant_shift(audio, shift_ratio, sr=self.sample_rate)

        # 3. Speed variation
        if self.speed_variation > 0:
            factor = 1.0 + np.random.uniform(-self.speed_variation, self.speed_variation)
            if abs(factor - 1.0) > 0.01:
                new_length = int(len(audio) / factor)
                if new_length > 1:
                    indices = np.linspace(0, len(audio) - 1, new_length)
                    audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

        # 4. Volume variation
        if self.volume_variation_db > 0:
            gain_db = np.random.uniform(-self.volume_variation_db, self.volume_variation_db / 2)
            gain_linear = 10 ** (gain_db / 20)
            audio = audio * gain_linear

        # 5. Additive noise
        if self.noise_snr_db is not None:
            signal_power = np.mean(audio ** 2)
            if signal_power > 1e-10:
                snr_linear = 10 ** (self.noise_snr_db / 10)
                noise_power = signal_power / snr_linear
                noise = np.random.randn(len(audio)).astype(np.float32) * np.sqrt(noise_power)
                audio = audio + noise

        return audio

    def __getitem__(self, idx: int) -> FullSentenceSample:
        """Get a training sample.

        Args:
            idx: Sample index

        Returns:
            FullSentenceSample with full audio and position
        """
        sent_idx, syl_idx = self._index[idx]
        sentence = self.sentences[sent_idx]

        key = str(sentence.audio_path)
        mel = self._mel_cache.get(key)
        audio = None if mel is not None else self._load_audio(sentence.audio_path).copy()

        # Get target syllable info
        target_syl = sentence.syllables[syl_idx]
        target_pinyin = _remove_tone_marks(target_syl.pinyin)
        target_tone = target_syl.tone_surface

        if mel is None:
            # Apply augmentation
            audio = self._apply_augmentation(audio)

            # Truncate to max duration
            if len(audio) > self.max_samples:
                audio = audio[:self.max_samples]

            mel_full = None
        else:
            audio = None
            # Apply SpecAugment to mel if augmenting
            if self.augment:
                mel = spec_augment(mel.copy())
            mel_full = mel

        return FullSentenceSample(
            sample_id=f"{sentence.id}_{syl_idx}",
            audio_full=audio,
            mel_full=mel_full,
            sample_rate=self.sample_rate,
            position=syl_idx,
            target_syllable=target_pinyin,
            target_tone=target_tone,
            n_syllables=len(sentence.syllables),
        )
