"""Tests for mel-first AISHELL tar loading and dataset mel chunks."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np

from mandarin_grader.data.aishell_tar_source import AISHELL3TarDataSource
from mandarin_grader.data.autoregressive_dataset import AutoregressiveDataset, SyntheticSentenceInfo
from mandarin_grader.types import TargetSyllable


def _write_tar_sample(shard_path: Path, utt_id: str, metadata: dict, mel: np.ndarray) -> None:
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(shard_path, "w") as tar:
        meta_bytes = json.dumps(metadata).encode("utf-8")
        meta_info = tarfile.TarInfo(name=f"{utt_id}.json")
        meta_info.size = len(meta_bytes)
        tar.addfile(meta_info, io.BytesIO(meta_bytes))

        mel_buf = io.BytesIO()
        np.save(mel_buf, mel.astype(np.float32), allow_pickle=False)
        mel_bytes = mel_buf.getvalue()
        mel_info = tarfile.TarInfo(name=f"{utt_id}.mel.npy")
        mel_info.size = len(mel_bytes)
        tar.addfile(mel_info, io.BytesIO(mel_bytes))


def test_aishell_tar_source_loads_mel_cache(tmp_path: Path) -> None:
    source = AISHELL3TarDataSource()

    metadata = {
        "id": "utt_001",
        "speaker": "SPK001",
        "hanzi": "??",
        "syllables": [
            {"hanzi": "?", "pinyin": "ni", "tone": 3},
            {"hanzi": "?", "pinyin": "hao", "tone": 3},
        ],
        "sample_rate": 16000,
        "audio_num_samples": 32000,
        "mel_hop_length": 160,
        "mel_win_length": 400,
        "mel_num_frames": 198,
    }
    mel = np.random.randn(80, 198).astype(np.float32)
    _write_tar_sample(tmp_path / "train" / "shard_0000.tar", "utt_001", metadata, mel)

    sentences = source.load(tmp_path, split="train")
    assert len(sentences) == 1

    sent = sentences[0]
    assert sent.id == "utt_001"
    assert sent.total_samples == 32000
    assert sent.syllable_boundaries[-1][1] == 32000

    mel_cache = source.get_mel_cache()
    assert str(sent.audio_path) in mel_cache
    assert mel_cache[str(sent.audio_path)].shape == (80, 198)


def test_autoregressive_dataset_uses_precomputed_mel(tmp_path: Path) -> None:
    sent = SyntheticSentenceInfo(
        id="utt_001",
        audio_path=tmp_path / "virtual.wav",
        text="??",
        syllables=[
            TargetSyllable(0, "?", "ni", "n", "i", 3, 3),
            TargetSyllable(1, "?", "hao", "h", "ao", 3, 3),
        ],
        syllable_boundaries=[(0, 16000), (16000, 32000)],
        sample_rate=16000,
        total_samples=32000,
    )

    dataset = AutoregressiveDataset([sent], sample_rate=16000, augment=False)
    dataset._mel_cache[str(sent.audio_path)] = np.random.randn(80, 220).astype(np.float32)

    sample = dataset[0]
    assert sample.audio_chunk is None
    assert sample.mel_chunk is not None
    expected_frames = 1 + (dataset.chunk_samples - dataset._mel_win_length) // dataset._mel_hop_length
    assert sample.mel_chunk.shape == (80, expected_frames)
