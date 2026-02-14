#!/usr/bin/env python3
"""Convert AISHELL-3 dataset to mel-first tar archives.

This script converts AISHELL-3 wav files into tar shards where each utterance
contains metadata JSON and a precomputed mel spectrogram.

Output format:
    output_dir/
        train/
            shard_0000.tar
            ...
        test/
            shard_0000.tar
            ...

Each shard contains:
    {utt_id}.json      metadata (id, speaker, hanzi, pinyin, syllables, mel config)
    {utt_id}.mel.npy   mel spectrogram [n_mels, time] float32
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import sys
import tarfile
import wave
from pathlib import Path

import numpy as np

# Allow local package imports when running this script directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

from mandarin_grader.model.syllable_predictor_v3 import (  # noqa: E402
    SyllablePredictorConfig,
    extract_mel_spectrogram,
)


def parse_aishell3_pinyin(pinyin_str: str) -> list[dict]:
    """Parse AISHELL-3 pinyin format to syllable dicts."""
    syllables = []
    for token in pinyin_str.strip().split():
        if not token:
            continue
        match = re.match(r"^([a-zA-ZuUvV\u00fc\u00dc]+)(\d)?$", token)
        if match:
            base = match.group(1).lower().replace("v", "\u00fc")
            tone = int(match.group(2)) if match.group(2) else 0
            if tone == 5 or tone > 4:
                tone = 0
            syllables.append({"pinyin": base, "tone": tone})
        else:
            syllables.append({"pinyin": token.lower(), "tone": 0})
    return syllables


def load_audio_resample(wav_path: Path, target_sr: int = 16000) -> np.ndarray:
    """Load wav file, resample to target_sr, return float32 audio in [-1, 1]."""
    with wave.open(str(wav_path), "rb") as wf:
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)
        sample_width = wf.getsampwidth()
        sr = wf.getframerate()
        n_channels = wf.getnchannels()

    if sample_width == 2:
        audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    if sr != target_sr:
        duration = len(audio) / sr
        new_length = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, new_length)
        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    return audio


def load_transcripts(content_file: Path) -> dict[str, tuple[str, str, list[dict]]]:
    """Load AISHELL-3 transcripts."""
    transcripts = {}
    with open(content_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue

            utt_id = parts[0].replace(".wav", "")
            content = parts[1]

            tokens = content.split()
            hanzi_chars = []
            pinyin_parts = []

            for i, token in enumerate(tokens):
                if i % 2 == 0:
                    hanzi_chars.append(token)
                else:
                    pinyin_parts.append(token)

            hanzi = "".join(hanzi_chars)
            pinyin = " ".join(pinyin_parts)
            syllables = parse_aishell3_pinyin(pinyin)

            for i, syl in enumerate(syllables):
                syl["hanzi"] = hanzi[i] if i < len(hanzi) else ""

            transcripts[utt_id] = (hanzi, pinyin, syllables)

    return transcripts


def convert_split(
    input_dir: Path,
    output_dir: Path,
    split: str,
    mel_config: SyllablePredictorConfig,
    shard_size: int = 1000,
) -> int:
    """Convert one split (train or test) to mel tar shards."""
    split_dir = input_dir / split
    content_file = split_dir / "content.txt"
    wav_dir = split_dir / "wav"

    if not content_file.exists():
        print(f"Skipping {split}: content.txt not found")
        return 0

    out_split_dir = output_dir / split
    if out_split_dir.exists():
        shutil.rmtree(out_split_dir)
    out_split_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading transcripts for {split}...")
    transcripts = load_transcripts(content_file)
    print(f"  Found {len(transcripts)} transcripts")

    items = []
    for speaker_dir in sorted(wav_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker_id = speaker_dir.name

        for wav_file in sorted(speaker_dir.glob("*.wav")):
            utt_id = wav_file.stem
            if utt_id not in transcripts:
                continue

            hanzi, pinyin, syllables = transcripts[utt_id]
            if not syllables:
                continue

            items.append({
                "wav_path": wav_file,
                "utt_id": utt_id,
                "speaker": speaker_id,
                "hanzi": hanzi,
                "pinyin": pinyin,
                "syllables": syllables,
            })

    print(f"  Found {len(items)} wav files with transcripts")

    total_converted = 0
    shard_idx = 0
    current_shard: list[dict] = []

    for i, item in enumerate(items, 1):
        try:
            audio = load_audio_resample(item["wav_path"], mel_config.sample_rate)
            mel = extract_mel_spectrogram(audio, mel_config)
        except Exception as e:
            print(f"  Error processing {item['wav_path']}: {e}")
            continue

        metadata = {
            "id": item["utt_id"],
            "speaker": item["speaker"],
            "hanzi": item["hanzi"],
            "pinyin": item["pinyin"],
            "syllables": item["syllables"],
            "sample_rate": mel_config.sample_rate,
            "audio_num_samples": int(len(audio)),
            "mel_n_mels": mel_config.n_mels,
            "mel_hop_length": mel_config.hop_length,
            "mel_win_length": mel_config.win_length,
            "mel_num_frames": int(mel.shape[1]),
        }

        current_shard.append({
            "utt_id": item["utt_id"],
            "metadata": metadata,
            "mel": mel,
        })

        if len(current_shard) >= shard_size:
            shard_path = out_split_dir / f"shard_{shard_idx:04d}.tar"
            write_shard(shard_path, current_shard)
            total_converted += len(current_shard)
            print(
                f"  Written {shard_path.name}: {len(current_shard)} utterances "
                f"({total_converted}/{len(items)})"
            )
            shard_idx += 1
            current_shard = []

        if i % 5000 == 0:
            print(f"  Processed {i}/{len(items)} utterances")

    if current_shard:
        shard_path = out_split_dir / f"shard_{shard_idx:04d}.tar"
        write_shard(shard_path, current_shard)
        total_converted += len(current_shard)
        print(f"  Written {shard_path.name}: {len(current_shard)} utterances ({total_converted}/{len(items)})")

    return total_converted


def write_shard(shard_path: Path, items: list[dict]) -> None:
    """Write a shard tar file with JSON + mel npy entries."""
    with tarfile.open(shard_path, "w") as tar:
        for item in items:
            utt_id = item["utt_id"]

            json_bytes = json.dumps(item["metadata"], ensure_ascii=False).encode("utf-8")
            json_info = tarfile.TarInfo(name=f"{utt_id}.json")
            json_info.size = len(json_bytes)
            tar.addfile(json_info, io.BytesIO(json_bytes))

            mel_buf = io.BytesIO()
            np.save(mel_buf, item["mel"].astype(np.float32), allow_pickle=False)
            mel_bytes = mel_buf.getvalue()
            mel_info = tarfile.TarInfo(name=f"{utt_id}.mel.npy")
            mel_info.size = len(mel_bytes)
            tar.addfile(mel_info, io.BytesIO(mel_bytes))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert AISHELL-3 to mel tar shards")
    parser.add_argument("input_dir", type=Path, help="AISHELL-3 root directory")
    parser.add_argument("output_dir", type=Path, help="Output directory for tar files")
    parser.add_argument("--shard-size", type=int, default=1000, help="Utterances per shard")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate")
    parser.add_argument("--n-mels", type=int, default=80, help="Mel bins")
    parser.add_argument("--hop-length", type=int, default=160, help="Mel hop length")
    parser.add_argument("--win-length", type=int, default=400, help="Mel window length")
    parser.add_argument("--splits", nargs="+", default=["train", "test"], help="Splits to convert")
    args = parser.parse_args()

    mel_config = SyllablePredictorConfig(
        n_mels=args.n_mels,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        win_length=args.win_length,
    )

    print(f"Converting AISHELL-3 from {args.input_dir} to {args.output_dir}")
    print(
        f"Shard size: {args.shard_size}, sample rate: {args.sample_rate}Hz, "
        f"mel: n_mels={args.n_mels}, hop={args.hop_length}, win={args.win_length}"
    )

    total = 0
    for split in args.splits:
        print(f"\nProcessing {split}...")
        count = convert_split(
            args.input_dir,
            args.output_dir,
            split,
            mel_config,
            args.shard_size,
        )
        total += count
        print(f"  {split}: {count} utterances")

    print(f"\nDone. Total: {total} utterances")


if __name__ == "__main__":
    main()
