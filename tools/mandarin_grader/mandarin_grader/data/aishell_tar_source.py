"""AISHELL-3 TAR data source - pre-processed tar archives.

This data source reads from tar archives created by convert_aishell3_to_tar.py.
Audio is pre-resampled to 16kHz and metadata is included as JSON files.

Dataset structure:
    aishell3_tar/
        train/
            shard_0000.tar
            shard_0001.tar
            ...
        test/
            shard_0000.tar
            ...

Each shard contains pairs:
    {utt_id}.json  - metadata
    {utt_id}.wav   - 16kHz mono int16 audio

This is much faster than the original AISHELL-3 source because:
1. Sequential tar reading vs random file access
2. No resampling needed (pre-done during conversion)
3. Metadata pre-parsed in JSON
"""

from __future__ import annotations

import io
import json
import tarfile
import wave
from pathlib import Path

import numpy as np

from .data_source import DataSource, SentenceInfo
from ..types import TargetSyllable


def load_wav_from_bytes(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Load wav audio from bytes.

    Args:
        wav_bytes: Raw wav file bytes

    Returns:
        Tuple of (audio array float32, sample_rate)
    """
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, 'rb') as wf:
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)
        sample_rate = wf.getframerate()

    # Use fast numpy.frombuffer
    audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sample_rate


class AISHELL3TarDataSource(DataSource):
    """Data source for pre-processed AISHELL-3 tar archives.

    Reads from tar shards created by convert_aishell3_to_tar.py.
    Much faster than the original AISHELL-3 source because:
    - Sequential tar reading
    - No resampling (pre-done)
    - Metadata pre-parsed

    Note: Unlike the original source, this stores actual audio data
    in SentenceInfo (via a custom audio_data field) rather than audio_path.
    """

    name = "aishell3_tar"
    description = "AISHELL-3 pre-processed tar archives (16kHz)"

    def load(
        self,
        data_dir: Path,
        split: str = "train",
        max_sentences: int | None = None,
    ) -> list[SentenceInfo]:
        """Load sentences from tar archives.

        Args:
            data_dir: Directory containing train/test subdirectories with shards
            split: "train" or "test"
            max_sentences: Maximum sentences to load (None = all)

        Returns:
            List of SentenceInfo objects
        """
        split_dir = data_dir / split
        if not split_dir.exists():
            return []

        # Find all shard files
        shard_files = sorted(split_dir.glob("shard_*.tar"))
        if not shard_files:
            return []

        sentences = []
        count = 0

        for shard_path in shard_files:
            if max_sentences and count >= max_sentences:
                break

            # Load all entries from this shard
            shard_sentences = self._load_shard(shard_path)

            for sent in shard_sentences:
                if max_sentences and count >= max_sentences:
                    break
                sentences.append(sent)
                count += 1

        return sentences

    def _load_shard(self, shard_path: Path) -> list[SentenceInfo]:
        """Load all sentences from a single shard.

        Args:
            shard_path: Path to shard tar file

        Returns:
            List of SentenceInfo objects
        """
        sentences = []

        # Read all members into memory first for efficiency
        metadata_map = {}  # utt_id -> metadata dict
        audio_map = {}  # utt_id -> audio bytes

        with tarfile.open(shard_path, 'r') as tar:
            for member in tar.getmembers():
                if member.name.endswith('.json'):
                    utt_id = member.name[:-5]  # Remove .json
                    f = tar.extractfile(member)
                    if f:
                        metadata_map[utt_id] = json.load(f)
                elif member.name.endswith('.wav'):
                    utt_id = member.name[:-4]  # Remove .wav
                    f = tar.extractfile(member)
                    if f:
                        audio_map[utt_id] = f.read()

        # Create SentenceInfo for each complete pair
        for utt_id, metadata in metadata_map.items():
            if utt_id not in audio_map:
                continue

            # Parse syllables from metadata
            syllables = []
            for i, syl in enumerate(metadata.get("syllables", [])):
                syllables.append(TargetSyllable(
                    index=i,
                    hanzi=syl.get("hanzi", ""),
                    pinyin=syl.get("pinyin", ""),
                    initial="",
                    final="",
                    tone_underlying=syl.get("tone", 0),
                    tone_surface=syl.get("tone", 0),
                ))

            if not syllables:
                continue

            # Load audio to get duration for boundary estimation
            audio, sample_rate = load_wav_from_bytes(audio_map[utt_id])
            audio_samples = len(audio)

            # Estimate uniform boundaries
            n_syllables = len(syllables)
            samples_per_syl = audio_samples // max(n_syllables, 1)
            boundaries = [
                (i * samples_per_syl, (i + 1) * samples_per_syl)
                for i in range(n_syllables)
            ]
            if boundaries:
                boundaries[-1] = (boundaries[-1][0], audio_samples)

            # Create a temporary file path marker (the actual audio will be cached)
            # We use the shard path + utt_id as a virtual path
            virtual_path = shard_path.parent / f"_cache_{utt_id}.wav"

            sentences.append(SentenceInfo(
                id=utt_id,
                audio_path=virtual_path,
                text=metadata.get("hanzi", ""),
                syllables=syllables,
                syllable_boundaries=boundaries,
                sample_rate=sample_rate,
            ))

            # Store audio in a cache that can be accessed by the dataset
            # This is a bit hacky but avoids changing SentenceInfo
            if not hasattr(self, '_audio_cache'):
                self._audio_cache = {}
            self._audio_cache[str(virtual_path)] = audio

        return sentences

    def get_audio_cache(self) -> dict[str, np.ndarray]:
        """Get the audio cache (for use by dataset).

        Returns:
            Dict mapping virtual path strings to audio arrays
        """
        return getattr(self, '_audio_cache', {})

    def is_available(self, data_dir: Path) -> bool:
        """Check if tar archives exist."""
        train_dir = data_dir / "train"
        return train_dir.exists() and any(train_dir.glob("shard_*.tar"))

    def get_speakers(self, data_dir: Path, split: str = "train") -> list[str]:
        """Get list of speaker IDs (requires loading metadata)."""
        # Would need to scan all shards - not implemented for efficiency
        return []
