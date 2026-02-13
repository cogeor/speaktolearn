#!/usr/bin/env python3
"""Convert AISHELL-3 dataset to tar archives with pre-resampled 16kHz audio.

This script converts the scattered AISHELL-3 wav files into tar shards for
efficient sequential loading. Audio is resampled from 44.1kHz to 16kHz during
conversion to avoid resampling overhead during training.

Output format:
    output_dir/
        train/
            shard_0000.tar  (contains ~1000 utterances)
            shard_0001.tar
            ...
        test/
            shard_0000.tar
            ...

Each shard contains pairs of files:
    {utt_id}.json  - metadata (id, speaker, hanzi, pinyin, syllables)
    {utt_id}.wav   - 16kHz mono int16 audio

Usage:
    python convert_aishell3_to_tar.py datasets/aishell3 datasets/aishell3_tar
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tarfile
import wave
from pathlib import Path

import numpy as np


def parse_aishell3_pinyin(pinyin_str: str) -> list[dict]:
    """Parse AISHELL-3 pinyin format to syllable dicts.

    Args:
        pinyin_str: Space-separated pinyin with tone numbers (e.g., "ni3 hao3")

    Returns:
        List of {"pinyin": str, "tone": int} dicts
    """
    syllables = []
    for token in pinyin_str.strip().split():
        if not token:
            continue
        match = re.match(r'^([a-zA-ZüÜ]+)(\d)?$', token)
        if match:
            base = match.group(1).lower()
            tone = int(match.group(2)) if match.group(2) else 0
            if tone == 5:  # neutral tone
                tone = 0
            elif tone > 4:
                tone = 0
            syllables.append({"pinyin": base, "tone": tone})
        else:
            syllables.append({"pinyin": token.lower(), "tone": 0})
    return syllables


def load_audio_resample(wav_path: Path, target_sr: int = 16000) -> bytes:
    """Load wav file, resample to target_sr, return as int16 bytes.

    Uses numpy.frombuffer for fast byte conversion.

    Args:
        wav_path: Path to input wav file
        target_sr: Target sample rate (default 16kHz)

    Returns:
        Audio as int16 bytes (little-endian)
    """
    with wave.open(str(wav_path), 'rb') as wf:
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)
        sample_width = wf.getsampwidth()
        sr = wf.getframerate()

    # Convert to float32 (using fast numpy.frombuffer)
    if sample_width == 2:
        audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    # Resample if needed (linear interpolation)
    if sr != target_sr:
        duration = len(audio) / sr
        new_length = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, new_length)
        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    # Convert back to int16 bytes
    audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    return audio_int16.tobytes()


def make_wav_bytes(audio_bytes: bytes, sample_rate: int = 16000) -> bytes:
    """Create a complete wav file in memory.

    Args:
        audio_bytes: Raw int16 audio bytes
        sample_rate: Sample rate

    Returns:
        Complete wav file as bytes
    """
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)
    return buf.getvalue()


def load_transcripts(content_file: Path) -> dict[str, tuple[str, str, list[dict]]]:
    """Load AISHELL-3 transcripts.

    Args:
        content_file: Path to content.txt

    Returns:
        Dict: utt_id -> (hanzi, pinyin_str, syllables)
    """
    transcripts = {}
    with open(content_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) < 2:
                continue

            utt_id = parts[0].replace('.wav', '')
            content = parts[1]

            # Parse interleaved hanzi + pinyin format
            tokens = content.split()
            hanzi_chars = []
            pinyin_parts = []

            for i, token in enumerate(tokens):
                if i % 2 == 0:
                    hanzi_chars.append(token)
                else:
                    pinyin_parts.append(token)

            hanzi = ''.join(hanzi_chars)
            pinyin = ' '.join(pinyin_parts)
            syllables = parse_aishell3_pinyin(pinyin)

            # Add hanzi to syllables
            for i, syl in enumerate(syllables):
                syl["hanzi"] = hanzi[i] if i < len(hanzi) else ""

            transcripts[utt_id] = (hanzi, pinyin, syllables)

    return transcripts


def convert_split(
    input_dir: Path,
    output_dir: Path,
    split: str,
    shard_size: int = 1000,
    target_sr: int = 16000,
) -> int:
    """Convert one split (train or test) to tar shards.

    Args:
        input_dir: AISHELL-3 root directory
        output_dir: Output directory for tar files
        split: "train" or "test"
        shard_size: Number of utterances per shard
        target_sr: Target sample rate

    Returns:
        Number of utterances converted
    """
    split_dir = input_dir / split
    content_file = split_dir / "content.txt"
    wav_dir = split_dir / "wav"

    if not content_file.exists():
        print(f"Skipping {split}: content.txt not found")
        return 0

    # Create output directory
    out_split_dir = output_dir / split
    out_split_dir.mkdir(parents=True, exist_ok=True)

    # Load all transcripts
    print(f"Loading transcripts for {split}...")
    transcripts = load_transcripts(content_file)
    print(f"  Found {len(transcripts)} transcripts")

    # Collect all wav files with their metadata
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

    # Convert to tar shards
    total_converted = 0
    shard_idx = 0
    current_shard = []

    for i, item in enumerate(items):
        # Load and resample audio
        try:
            audio_bytes = load_audio_resample(item["wav_path"], target_sr)
            wav_bytes = make_wav_bytes(audio_bytes, target_sr)
        except Exception as e:
            print(f"  Error processing {item['wav_path']}: {e}")
            continue

        # Create metadata JSON
        metadata = {
            "id": item["utt_id"],
            "speaker": item["speaker"],
            "hanzi": item["hanzi"],
            "pinyin": item["pinyin"],
            "syllables": item["syllables"],
            "sample_rate": target_sr,
        }

        current_shard.append({
            "utt_id": item["utt_id"],
            "metadata": metadata,
            "wav_bytes": wav_bytes,
        })

        # Write shard when full
        if len(current_shard) >= shard_size:
            shard_path = out_split_dir / f"shard_{shard_idx:04d}.tar"
            write_shard(shard_path, current_shard)
            total_converted += len(current_shard)
            print(f"  Written {shard_path.name}: {len(current_shard)} utterances ({total_converted}/{len(items)})")
            shard_idx += 1
            current_shard = []

    # Write remaining items
    if current_shard:
        shard_path = out_split_dir / f"shard_{shard_idx:04d}.tar"
        write_shard(shard_path, current_shard)
        total_converted += len(current_shard)
        print(f"  Written {shard_path.name}: {len(current_shard)} utterances ({total_converted}/{len(items)})")

    return total_converted


def write_shard(shard_path: Path, items: list[dict]) -> None:
    """Write a shard tar file.

    Args:
        shard_path: Output tar file path
        items: List of {"utt_id", "metadata", "wav_bytes"} dicts
    """
    with tarfile.open(shard_path, 'w') as tar:
        for item in items:
            utt_id = item["utt_id"]

            # Add JSON metadata
            json_bytes = json.dumps(item["metadata"], ensure_ascii=False).encode('utf-8')
            json_info = tarfile.TarInfo(name=f"{utt_id}.json")
            json_info.size = len(json_bytes)
            tar.addfile(json_info, io.BytesIO(json_bytes))

            # Add wav audio
            wav_info = tarfile.TarInfo(name=f"{utt_id}.wav")
            wav_info.size = len(item["wav_bytes"])
            tar.addfile(wav_info, io.BytesIO(item["wav_bytes"]))


def main():
    parser = argparse.ArgumentParser(description="Convert AISHELL-3 to tar shards")
    parser.add_argument("input_dir", type=Path, help="AISHELL-3 root directory")
    parser.add_argument("output_dir", type=Path, help="Output directory for tar files")
    parser.add_argument("--shard-size", type=int, default=1000, help="Utterances per shard")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate")
    parser.add_argument("--splits", nargs="+", default=["train", "test"], help="Splits to convert")
    args = parser.parse_args()

    print(f"Converting AISHELL-3 from {args.input_dir} to {args.output_dir}")
    print(f"Shard size: {args.shard_size}, Sample rate: {args.sample_rate}Hz")

    total = 0
    for split in args.splits:
        print(f"\nProcessing {split}...")
        count = convert_split(
            args.input_dir,
            args.output_dir,
            split,
            args.shard_size,
            args.sample_rate,
        )
        total += count
        print(f"  {split}: {count} utterances")

    print(f"\nDone! Total: {total} utterances")


if __name__ == "__main__":
    main()
