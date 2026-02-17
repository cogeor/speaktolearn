"""Autoregressive dataset for syllable+tone prediction training.

This module provides a dataset class that creates training samples for the
SyllablePredictorV3 model. Each sample consists of:
- Audio chunk (1s, ~100 frames at 10ms hop)
- Pinyin context (syllables before the target)
- Target syllable and tone

For synthetic data, we have exact syllable boundaries, making it easy to
create training samples where the target syllable is within the audio chunk.

Data augmentation:
- Random start position within the sentence
- Audio augmentations (noise, speed variation)
"""

from __future__ import annotations

import json
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
from numpy.typing import NDArray

from ..types import Tone, TargetSyllable
from ..pitch import extract_f0_pyin, normalize_f0, hz_to_semitones
from .lexicon import SyllableLexicon, _remove_tone_marks
from .dataloader import AudioSample, SentenceDataset, parse_romanization
from .augmentation import pitch_shift, formant_shift


@dataclass
class AutoregressiveSample:
    """A single training sample for autoregressive prediction."""

    sample_id: str
    audio_chunk: NDArray[np.float32] | None  # [n_samples] raw audio for 1s chunk
    mel_chunk: NDArray[np.float32] | None  # [n_mels, time] precomputed mel chunk
    f0_chunk: NDArray[np.float32] | None  # [time] normalized F0 for the chunk
    sample_rate: int
    pinyin_context: list[str]  # Previous syllables (base pinyin, no tone)
    target_syllable: str  # Base pinyin of target syllable
    target_tone: Tone  # Tone of target syllable
    syllable_idx: int  # Index of target syllable in sentence


@dataclass
class SyntheticSentenceInfo:
    """Information about a synthetic sentence."""

    id: str
    audio_path: Path
    text: str
    syllables: list[TargetSyllable]
    # Boundaries in samples (from synthesis metadata or uniform split)
    syllable_boundaries: list[tuple[int, int]]  # [(start_sample, end_sample), ...]
    sample_rate: int = 16000
    total_samples: int | None = None


def load_synthetic_metadata(data_dir: Path) -> list[SyntheticSentenceInfo]:
    """Load synthetic sentence metadata from directory.

    Supports two formats:
    1. New format with "sentences" key
    2. Legacy format with "samples" key (from synthetic_train)

    Expected structure:
        data_dir/
            metadata.json  (with sentence info)
            audio/
                sentence_001.wav
                ...

    Args:
        data_dir: Directory containing synthetic data

    Returns:
        List of SyntheticSentenceInfo
    """
    metadata_path = data_dir / "metadata.json"
    audio_dir = data_dir / "audio"

    if not metadata_path.exists():
        return []

    with open(metadata_path, encoding="utf-8") as f:
        data = json.load(f)

    sentences = []
    sample_rate = data.get("sample_rate", 16000)

    # Support both "sentences" and "samples" keys (legacy format)
    items = data.get("sentences", data.get("samples", []))

    for item in items:
        # Handle audio path - could be relative with backslashes (Windows)
        if "audio_path" in item:
            audio_rel = item["audio_path"].replace("\\", "/")
            audio_path = data_dir / audio_rel
        else:
            audio_path = audio_dir / f"{item['id']}.wav"

        if not audio_path.exists():
            continue

        # Parse syllables from romanization if not pre-parsed
        if "syllables" in item:
            syllables = []
            for i, syl_data in enumerate(item["syllables"]):
                syllables.append(TargetSyllable(
                    index=i,
                    hanzi=syl_data.get("hanzi", ""),
                    pinyin=syl_data.get("pinyin", ""),
                    initial=syl_data.get("initial", ""),
                    final=syl_data.get("final", ""),
                    tone_underlying=syl_data.get("tone", 0),
                    tone_surface=syl_data.get("tone", 0),
                ))
        else:
            syllables = parse_romanization(
                item.get("romanization", ""),
                item.get("text", "")
            )

        # Parse boundaries - handle multiple formats
        boundaries = []
        if "boundaries" in item:
            for b in item["boundaries"]:
                if isinstance(b, dict):
                    # Object format: {"start_ms": 0, "end_ms": 100}
                    start_ms = b.get("start_ms", 0)
                    end_ms = b.get("end_ms", 0)
                    start_sample = int(start_ms * sample_rate / 1000)
                    end_sample = int(end_ms * sample_rate / 1000)
                    boundaries.append((start_sample, end_sample))
                elif isinstance(b, (list, tuple)) and len(b) >= 2:
                    # Array format: [start_sample, end_sample]
                    boundaries.append((b[0], b[1]))
        elif "ground_truth_spans" in item:
            # Convert from ms to samples
            for span in item["ground_truth_spans"]:
                start_ms = span.get("start_ms", 0)
                end_ms = span.get("end_ms", 0)
                start_sample = int(start_ms * sample_rate / 1000)
                end_sample = int(end_ms * sample_rate / 1000)
                boundaries.append((start_sample, end_sample))
        # else: boundaries stays empty, will estimate from audio length

        sentences.append(SyntheticSentenceInfo(
            id=item["id"],
            audio_path=audio_path,
            text=item.get("text", ""),
            syllables=syllables,
            syllable_boundaries=boundaries,
            sample_rate=sample_rate,
            total_samples=None,
        ))

    return sentences


def load_audio_wav(path: Path, target_sr: int = 16000) -> NDArray[np.float32]:
    """Load audio from WAV file.

    Args:
        path: Path to WAV file
        target_sr: Target sample rate

    Returns:
        Audio as float32 array normalized to [-1, 1]
    """
    with wave.open(str(path), 'rb') as wf:
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)
        sample_width = wf.getsampwidth()
        sr = wf.getframerate()

    if sample_width == 2:
        audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    # Resample if needed (simple linear interpolation)
    if sr != target_sr:
        duration = len(audio) / sr
        new_length = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, new_length)
        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    return audio


def spec_augment(
    mel: NDArray[np.float32],
    freq_masks: int = 2,
    freq_width: int = 10,
    time_masks: int = 2,
    time_width: int = 40,
) -> NDArray[np.float32]:
    """Apply SpecAugment: frequency and time masking to mel spectrogram.

    Args:
        mel: Mel spectrogram [n_mels, time]
        freq_masks: Number of frequency masks
        freq_width: Max width of each frequency mask
        time_masks: Number of time masks
        time_width: Max width of each time mask

    Returns:
        Augmented mel spectrogram
    """
    mel = mel.copy()
    n_mels, n_time = mel.shape

    # Frequency masking
    for _ in range(freq_masks):
        f = np.random.randint(0, freq_width + 1)
        f0 = np.random.randint(0, max(1, n_mels - f))
        mel[f0:f0 + f, :] = 0.0

    # Time masking
    for _ in range(time_masks):
        t = np.random.randint(0, time_width + 1)
        t0 = np.random.randint(0, max(1, n_time - t))
        mel[:, t0:t0 + t] = 0.0

    return mel


class AutoregressiveDataset:
    """Dataset for autoregressive syllable+tone prediction.

    Creates training samples by iterating through synthetic sentences
    and generating samples for each syllable position.

    Each sample includes:
    - 1s audio chunk containing the target syllable (position randomized)
    - Context: all previous syllables' base pinyin
    - Target: syllable index + tone

    IMPORTANT - Inference Strategy:
    ==============================
    This dataset trains the model to find a target syllable ANYWHERE within
    a 1s audio chunk (not centered). This matches the inference scenario:

    1. Run VAD on full audio → get active speech regions (remove silence)
    2. Estimate syllable positions: total_active_duration / n_syllables
    3. For syllable i at estimated position P:
       - Take a 1s chunk that CONTAINS position P
       - Chunk start can vary as long as target is inside with margin
    4. Model predicts what syllable+tone is in the chunk given context
    5. Advance to next syllable position, repeat

    The margin parameter ensures the target syllable is not cut off at
    chunk boundaries. During training, chunk position is randomized within
    valid range to match inference variability.

    Data augmentation:
    - Randomized chunk position (target anywhere in chunk, not centered)
    - Audio noise and speed variation
    """

    def __init__(
        self,
        sentences: list[SyntheticSentenceInfo],
        sample_rate: int = 16000,
        chunk_duration_s: float = 1.0,
        margin_s: float = 0.1,
        augment: bool = True,
        noise_snr_db: float | None = 30.0,
        speed_variation: float = 0.1,
        volume_variation_db: float = 12.0,
        pitch_shift_semitones: float = 0.0,
        formant_shift_percent: float = 0.0,
    ):
        """Initialize dataset.

        Args:
            sentences: List of synthetic sentence info
            sample_rate: Audio sample rate
            chunk_duration_s: Duration of audio chunks in seconds
            margin_s: Minimum margin (seconds) between target syllable and chunk edge.
                      Ensures target is not cut off. Default 100ms.
            augment: Whether to apply augmentation (includes chunk position randomization)
            noise_snr_db: SNR for noise augmentation (None = no noise)
            speed_variation: Speed variation fraction (0.1 = ±10%, 0.05 = ±5%)
            volume_variation_db: Volume variation in dB (12 = -12dB to +6dB range)
            pitch_shift_semitones: Max pitch shift in semitones (±), 0 to disable
            formant_shift_percent: Max formant shift in percent (±), 0 to disable
        """
        self.sentences = sentences
        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_duration_s * sample_rate)
        self.margin_samples = int(margin_s * sample_rate)
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
        # Optional precomputed mel cache (keyed by audio_path string)
        self._mel_cache: dict[str, NDArray[np.float32]] = {}
        self._mel_hop_length = 160
        self._mel_win_length = 400

    def _extract_f0_chunk(
        self,
        audio: NDArray[np.float32],
        sr: int = 16000,
        hop_length: int = 160,
    ) -> NDArray[np.float32]:
        """Extract normalized F0 features from audio chunk.

        Args:
            audio: Audio samples [n_samples]
            sr: Sample rate
            hop_length: Hop length for frame extraction (should match mel)

        Returns:
            Normalized F0 [n_frames], 0 for unvoiced frames
        """
        f0_hz, voicing = extract_f0_pyin(
            audio, sr=sr, fmin=50.0, fmax=500.0, hop_length=hop_length
        )
        semitones = hz_to_semitones(f0_hz, ref_hz=100.0)
        normalized = normalize_f0(semitones, voicing)
        return normalized.astype(np.float32)

    def __len__(self) -> int:
        return len(self._index)

    def preload_audio(self, progress_callback=None) -> None:
        """Pre-load all audio files into memory cache.

        This significantly speeds up training by avoiding disk I/O during batching.
        Call this before training starts.

        Args:
            progress_callback: Optional callback(loaded, total) for progress reporting
        """
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
        """Apply audio augmentation.

        Order of operations:
        1. Pitch shift (preserves duration, high-quality phase vocoder)
        2. Formant shift (preserves pitch, simulates vocal tract variation)
        3. Speed variation (simple resampling - fast but changes pitch)
        4. Volume variation
        5. Additive noise
        """
        if not self.augment:
            return audio

        # 1. Pitch shift (high-quality, duration-preserving)
        if self.pitch_shift_semitones > 0:
            semitones = np.random.uniform(
                -self.pitch_shift_semitones, self.pitch_shift_semitones
            )
            if abs(semitones) > 0.1:  # Skip very small shifts
                audio = pitch_shift(audio, semitones, sr=self.sample_rate)

        # 2. Formant shift (simulates different vocal tract lengths)
        if self.formant_shift_percent > 0:
            shift_ratio = 1.0 + np.random.uniform(
                -self.formant_shift_percent, self.formant_shift_percent
            ) / 100.0
            if abs(shift_ratio - 1.0) > 0.01:  # Skip very small shifts
                audio = formant_shift(audio, shift_ratio, sr=self.sample_rate)

        # 3. Speed variation (simple linear interpolation - fast)
        if self.speed_variation > 0:
            factor = 1.0 + np.random.uniform(-self.speed_variation, self.speed_variation)
            if abs(factor - 1.0) > 0.01:
                new_length = int(len(audio) / factor)
                if new_length > 1:
                    indices = np.linspace(0, len(audio) - 1, new_length)
                    audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

        # 4. Volume variation (critical for robustness to different recording levels)
        # Random gain between -volume_db and +volume_db/2
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

    def _samples_to_frame(self, sample_idx: int) -> int:
        return max(0, sample_idx // self._mel_hop_length)

    def _frames_for_samples(self, n_samples: int) -> int:
        if n_samples < self._mel_win_length:
            return 1
        return 1 + (n_samples - self._mel_win_length) // self._mel_hop_length

    def _extract_mel_chunk(
        self,
        mel: NDArray[np.float32],
        sentence: SyntheticSentenceInfo,
        syl_idx: int,
    ) -> NDArray[np.float32]:
        """Extract a fixed-duration mel chunk using the same chunking logic as audio."""
        if sentence.syllable_boundaries and syl_idx < len(sentence.syllable_boundaries):
            syl_start, syl_end = sentence.syllable_boundaries[syl_idx]
        else:
            total_samples = sentence.total_samples
            if total_samples is None:
                total_samples = max(0, (mel.shape[1] - 1) * self._mel_hop_length + self._mel_win_length)
            n_syllables = len(sentence.syllables)
            samples_per_syl = total_samples // max(n_syllables, 1)
            syl_start = syl_idx * samples_per_syl
            syl_end = (syl_idx + 1) * samples_per_syl

        total_samples = sentence.total_samples
        if total_samples is None:
            total_samples = max(0, (mel.shape[1] - 1) * self._mel_hop_length + self._mel_win_length)

        min_chunk_start = syl_end + self.margin_samples - self.chunk_samples
        max_chunk_start = syl_start - self.margin_samples
        min_chunk_start = max(0, min_chunk_start)
        max_chunk_start = max(0, min(max_chunk_start, total_samples - self.chunk_samples))

        if min_chunk_start > max_chunk_start:
            syl_mid = (syl_start + syl_end) // 2
            chunk_start = max(0, min(syl_mid - self.chunk_samples // 2, total_samples - self.chunk_samples))
        elif self.augment:
            chunk_start = int(np.random.randint(min_chunk_start, max_chunk_start + 1))
        else:
            chunk_start = (min_chunk_start + max_chunk_start) // 2

        target_frames = self._frames_for_samples(self.chunk_samples)
        start_frame = self._samples_to_frame(chunk_start)
        end_frame = start_frame + target_frames

        mel_chunk = mel[:, start_frame:end_frame]
        if mel_chunk.shape[1] < target_frames:
            pad = np.zeros((mel.shape[0], target_frames - mel_chunk.shape[1]), dtype=np.float32)
            mel_chunk = np.concatenate([mel_chunk, pad], axis=1)

        # Apply SpecAugment when augmenting
        if self.augment:
            mel_chunk = spec_augment(mel_chunk)

        return mel_chunk.astype(np.float32, copy=False)

    def __getitem__(self, idx: int) -> AutoregressiveSample:
        """Get a training sample.

        The chunk position is randomized so the target syllable can appear
        anywhere within the chunk (with margin from edges). This matches
        inference behavior where we don't know exact syllable positions.

        Args:
            idx: Sample index

        Returns:
            AutoregressiveSample
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
            # Determine syllable boundaries (exact or estimated)
            if sentence.syllable_boundaries and syl_idx < len(sentence.syllable_boundaries):
                syl_start, syl_end = sentence.syllable_boundaries[syl_idx]
            else:
                n_syllables = len(sentence.syllables)
                samples_per_syl = len(audio) // max(n_syllables, 1)
                syl_start = syl_idx * samples_per_syl
                syl_end = (syl_idx + 1) * samples_per_syl

            # Calculate valid chunk start range
            min_chunk_start = syl_end + self.margin_samples - self.chunk_samples
            max_chunk_start = syl_start - self.margin_samples

            min_chunk_start = max(0, min_chunk_start)
            max_chunk_start = max(0, min(max_chunk_start, len(audio) - self.chunk_samples))

            if min_chunk_start > max_chunk_start:
                syl_mid = (syl_start + syl_end) // 2
                chunk_start = max(0, min(syl_mid - self.chunk_samples // 2,
                                         len(audio) - self.chunk_samples))
            elif self.augment:
                chunk_start = int(np.random.randint(min_chunk_start, max_chunk_start + 1))
            else:
                chunk_start = (min_chunk_start + max_chunk_start) // 2

            chunk_end = chunk_start + self.chunk_samples

            if chunk_end > len(audio):
                chunk_end = len(audio)
                chunk_start = max(0, chunk_end - self.chunk_samples)

            chunk = audio[chunk_start:chunk_end]
            if len(chunk) < self.chunk_samples:
                chunk = np.pad(chunk, (0, self.chunk_samples - len(chunk)))

            # Apply audio augmentation only on waveform path.
            chunk = self._apply_augmentation(chunk)
            mel_chunk = None
            # Extract F0 from audio chunk
            f0_chunk = self._extract_f0_chunk(chunk, self.sample_rate, self._mel_hop_length)
        else:
            chunk = None
            mel_chunk = self._extract_mel_chunk(mel, sentence, syl_idx)
            # F0 not available when using precomputed mel (would need separate cache)
            f0_chunk = None

        # Build pinyin context (all syllables before target)
        context = []
        for i in range(syl_idx):
            context.append(_remove_tone_marks(sentence.syllables[i].pinyin))

        return AutoregressiveSample(
            sample_id=f"{sentence.id}_{syl_idx}",
            audio_chunk=chunk,
            mel_chunk=mel_chunk,
            f0_chunk=f0_chunk,
            sample_rate=self.sample_rate,
            pinyin_context=context,
            target_syllable=target_pinyin,
            target_tone=target_tone,
            syllable_idx=syl_idx,
        )

    def get_sample_for_sentence(
        self,
        sentence_idx: int,
        target_syl_idx: int,
    ) -> AutoregressiveSample:
        """Get a specific sample by sentence and syllable index."""
        # Find the index
        for idx, (s_idx, syl_idx) in enumerate(self._index):
            if s_idx == sentence_idx and syl_idx == target_syl_idx:
                return self[idx]
        raise ValueError(f"Sample not found: sentence={sentence_idx}, syllable={target_syl_idx}")


def estimate_syllable_positions(
    audio_samples: int,
    n_syllables: int,
    sample_rate: int = 16000,
    active_regions: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Estimate syllable positions using proportional mapping.

    This is the inference-time heuristic for determining where each syllable
    is located in the audio. Used when we know the expected text but not
    the exact alignment.

    Args:
        audio_samples: Total audio length in samples
        n_syllables: Number of expected syllables
        sample_rate: Audio sample rate
        active_regions: Optional list of (start, end) sample indices from VAD.
                       If provided, syllables are mapped only to active regions.
                       If None, syllables are distributed across entire audio.

    Returns:
        List of (start_sample, end_sample) for each syllable
    """
    if n_syllables == 0:
        return []

    if active_regions:
        # Map syllables to active speech regions only
        total_active = sum(end - start for start, end in active_regions)
        samples_per_syl = total_active // n_syllables

        positions = []
        current_syl = 0
        accumulated = 0

        for region_start, region_end in active_regions:
            region_len = region_end - region_start
            region_pos = 0

            while current_syl < n_syllables and region_pos < region_len:
                syl_start = region_start + region_pos
                remaining_in_region = region_len - region_pos
                syl_len = min(samples_per_syl, remaining_in_region)
                syl_end = syl_start + syl_len

                positions.append((syl_start, syl_end))
                current_syl += 1
                region_pos += syl_len

        # Handle any remaining syllables (shouldn't happen with good VAD)
        while len(positions) < n_syllables:
            positions.append(positions[-1] if positions else (0, audio_samples))

        return positions
    else:
        # Simple uniform distribution across entire audio
        samples_per_syl = audio_samples // n_syllables
        return [
            (i * samples_per_syl, (i + 1) * samples_per_syl)
            for i in range(n_syllables)
        ]


def get_inference_chunk(
    audio: NDArray[np.float32],
    syllable_position: tuple[int, int],
    chunk_samples: int = 16000,
    margin_samples: int = 1600,
) -> NDArray[np.float32]:
    """Extract audio chunk for inference given estimated syllable position.

    This function extracts a chunk that contains the target syllable with
    appropriate margin, matching the training distribution.

    Args:
        audio: Full audio array
        syllable_position: (start_sample, end_sample) of target syllable
        chunk_samples: Chunk size in samples (default 1s at 16kHz)
        margin_samples: Margin from chunk edges (default 100ms at 16kHz)

    Returns:
        Audio chunk of length chunk_samples

    Example (mobile inference pseudocode):
        ```
        # 1. Load audio and run VAD
        audio = load_audio(recording_path)
        active_regions = run_vad(audio)

        # 2. Estimate syllable positions
        positions = estimate_syllable_positions(
            len(audio), len(expected_pinyin), active_regions=active_regions
        )

        # 3. For each syllable, extract chunk and predict
        context = []
        for i, (expected_syl, position) in enumerate(zip(expected_pinyin, positions)):
            chunk = get_inference_chunk(audio, position)
            mel = extract_mel(chunk)
            pinyin_ids = encode_context(context)

            pred_syl, pred_tone = model.predict(mel, pinyin_ids)

            # Compare pred_syl to expected_syl for grading
            is_correct = (pred_syl == expected_syl)

            context.append(expected_syl)  # Use expected for next iteration
        ```
    """
    syl_start, syl_end = syllable_position
    audio_len = len(audio)

    # Calculate valid chunk start range (same logic as training)
    min_chunk_start = syl_end + margin_samples - chunk_samples
    max_chunk_start = syl_start - margin_samples

    min_chunk_start = max(0, min_chunk_start)
    max_chunk_start = max(0, min(max_chunk_start, audio_len - chunk_samples))

    if min_chunk_start > max_chunk_start:
        # Edge case - center on syllable
        syl_mid = (syl_start + syl_end) // 2
        chunk_start = max(0, min(syl_mid - chunk_samples // 2,
                                 audio_len - chunk_samples))
    else:
        # Use middle of valid range (deterministic for inference)
        chunk_start = (min_chunk_start + max_chunk_start) // 2

    chunk_end = chunk_start + chunk_samples

    # Extract and pad if needed
    if chunk_end > audio_len:
        chunk = np.pad(audio[chunk_start:], (0, chunk_end - audio_len))
    else:
        chunk = audio[chunk_start:chunk_end]

    return chunk


def create_synthetic_training_data(
    lexicon: SyllableLexicon,
    sentences_json: Path,
    output_dir: Path,
    voice: str = "female1",
    sample_rate: int = 16000,
    max_sentences: int | None = None,
) -> Path:
    """Create synthetic training data from syllable lexicon.

    This function creates synthetic sentences by concatenating syllable audio
    files from the lexicon, with exact ground-truth boundaries.

    Args:
        lexicon: SyllableLexicon with individual syllable audio
        sentences_json: Path to sentences.zh.json with sentence definitions
        output_dir: Output directory for synthetic data
        voice: Voice ID to use
        sample_rate: Sample rate
        max_sentences: Maximum number of sentences (None = all)

    Returns:
        Path to output directory
    """
    from .synthesis import synthesize_sentence, AugmentationConfig

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    # Load sentences
    with open(sentences_json, encoding="utf-8") as f:
        data = json.load(f)

    sentences_meta = []
    count = 0

    for item in data.get("items", []):
        if max_sentences and count >= max_sentences:
            break

        # Parse syllables
        syllables = parse_romanization(
            item.get("romanization", ""),
            item.get("text", "")
        )
        if not syllables:
            continue

        # Synthesize
        try:
            audio, spans = synthesize_sentence(
                syllables=syllables,
                lexicon=lexicon,
                voice=voice,
                augmentations=AugmentationConfig(),  # No augmentation for base data
                sample_rate=sample_rate,
            )
        except Exception as e:
            print(f"Failed to synthesize {item['id']}: {e}")
            continue

        if len(audio) == 0:
            continue

        # Save audio
        audio_path = audio_dir / f"{item['id']}.wav"
        with wave.open(str(audio_path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
            wf.writeframes(audio_int16.tobytes())

        # Convert spans to boundaries
        boundaries = []
        for span in spans:
            start_sample = int(span.start_ms * sample_rate / 1000)
            end_sample = int(span.end_ms * sample_rate / 1000)
            boundaries.append([start_sample, end_sample])

        # Add to metadata
        sentences_meta.append({
            "id": item["id"],
            "text": item.get("text", ""),
            "romanization": item.get("romanization", ""),
            "syllables": [
                {
                    "hanzi": s.hanzi,
                    "pinyin": s.pinyin,
                    "tone": s.tone_surface,
                }
                for s in syllables
            ],
            "boundaries": boundaries,
        })
        count += 1

    # Save metadata
    metadata = {
        "sample_rate": sample_rate,
        "voice": voice,
        "n_sentences": len(sentences_meta),
        "sentences": sentences_meta,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return output_dir
