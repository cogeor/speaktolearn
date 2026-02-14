"""AISHELL-3 TAR data source with precomputed mel features.

This data source reads tar shards created by convert_aishell3_to_tar.py where
mel spectrograms are precomputed offline. That removes per-batch mel extraction
from the training hot path.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np

from .data_source import DataSource, SentenceInfo
from ..types import TargetSyllable


class AISHELL3TarDataSource(DataSource):
    """Data source for mel-first AISHELL-3 tar archives."""

    name = "aishell3_tar"
    description = "AISHELL-3 pre-processed tar archives with precomputed mel"

    def __init__(self) -> None:
        self._mel_cache: dict[str, np.ndarray] = {}

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

        shard_files = sorted(split_dir.glob("shard_*.tar"))
        if not shard_files:
            return []

        self._mel_cache = {}
        sentences: list[SentenceInfo] = []
        count = 0

        for shard_path in shard_files:
            if max_sentences is not None and count >= max_sentences:
                break

            for sent in self._load_shard(shard_path):
                if max_sentences is not None and count >= max_sentences:
                    break
                sentences.append(sent)
                count += 1

        return sentences

    def _load_shard(self, shard_path: Path) -> list[SentenceInfo]:
        """Load all sentences from a single shard."""
        sentences: list[SentenceInfo] = []

        metadata_map: dict[str, dict] = {}
        mel_map: dict[str, np.ndarray] = {}

        with tarfile.open(shard_path, "r") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".json"):
                    utt_id = member.name[:-5]
                    f = tar.extractfile(member)
                    if f is not None:
                        metadata_map[utt_id] = json.load(f)
                elif member.name.endswith(".mel.npy"):
                    utt_id = member.name[:-8]
                    f = tar.extractfile(member)
                    if f is not None:
                        mel_map[utt_id] = np.load(io.BytesIO(f.read()))

        for utt_id, metadata in metadata_map.items():
            mel = mel_map.get(utt_id)
            if mel is None:
                continue

            syllables = []
            for i, syl in enumerate(metadata.get("syllables", [])):
                tone = int(syl.get("tone", 0))
                syllables.append(TargetSyllable(
                    index=i,
                    hanzi=syl.get("hanzi", ""),
                    pinyin=syl.get("pinyin", ""),
                    initial="",
                    final="",
                    tone_underlying=tone,
                    tone_surface=tone,
                ))

            if not syllables:
                continue

            total_samples = int(metadata.get("audio_num_samples", 0))
            if total_samples <= 0:
                n_frames = int(mel.shape[1]) if mel.ndim == 2 else 0
                hop = int(metadata.get("mel_hop_length", 160))
                win = int(metadata.get("mel_win_length", 400))
                total_samples = max(0, (n_frames - 1) * hop + win)

            n_syllables = len(syllables)
            samples_per_syl = total_samples // max(n_syllables, 1)
            boundaries = [
                (i * samples_per_syl, (i + 1) * samples_per_syl)
                for i in range(n_syllables)
            ]
            if boundaries:
                boundaries[-1] = (boundaries[-1][0], total_samples)

            virtual_path = shard_path.parent / f"_cache_{utt_id}.wav"
            virtual_key = str(virtual_path)
            self._mel_cache[virtual_key] = mel.astype(np.float32, copy=False)

            sentences.append(SentenceInfo(
                id=utt_id,
                audio_path=virtual_path,
                text=metadata.get("hanzi", ""),
                syllables=syllables,
                syllable_boundaries=boundaries,
                sample_rate=int(metadata.get("sample_rate", 16000)),
                total_samples=total_samples,
            ))

        return sentences

    def get_mel_cache(self) -> dict[str, np.ndarray]:
        """Get preloaded mel cache keyed by virtual audio path."""
        return self._mel_cache

    def is_available(self, data_dir: Path) -> bool:
        """Check if tar archives exist."""
        train_dir = data_dir / "train"
        return train_dir.exists() and any(train_dir.glob("shard_*.tar"))

    def get_speakers(self, data_dir: Path, split: str = "train") -> list[str]:
        """Get list of speaker IDs (not implemented)."""
        return []
