#!/usr/bin/env python3
"""Split audio files into syllables using energy-based segmentation.

This script segments TTS audio files and saves individual syllables
for manual verification of boundary detection quality.

Usage:
    cd tools/mandarin_grader
    python scripts/split_audio.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

# Fix Windows encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mandarin_grader.data import SentenceDataset
from mandarin_grader.vad import segment_by_energy, segment_by_voicing, trim_silence, compute_rms_energy


def load_audio_pydub(path: Path, sr: int = 16000):
    """Load audio using pydub."""
    from pydub import AudioSegment

    audio_seg = AudioSegment.from_file(str(path))
    audio_seg = audio_seg.set_channels(1).set_frame_rate(sr)
    samples = np.array(audio_seg.get_array_of_samples())
    max_val = float(2 ** (audio_seg.sample_width * 8 - 1))
    return samples.astype(np.float32) / max_val, audio_seg


def save_audio_segment(audio_seg, start_ms: int, end_ms: int, output_path: Path):
    """Save a segment of audio to file."""
    segment = audio_seg[start_ms:end_ms]
    segment.export(str(output_path), format="wav")


def main():
    # Paths
    project_root = Path(__file__).parent.parent.parent.parent
    assets_dir = project_root / "apps" / "mobile_flutter" / "assets"
    sentences_json = assets_dir / "datasets" / "sentences.zh.json"
    audio_dir = assets_dir / "examples"
    output_dir = Path(__file__).parent.parent / "data" / "splits"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Load dataset
    dataset = SentenceDataset.from_app_assets(sentences_json, audio_dir, voice="female")
    print(f"Loaded {len(dataset)} samples")

    # Select a few samples with different syllable counts
    sample_ids = [
        "ts_000001",  # ni hao (2 syllables)
        "ts_000003",  # wo ai ni (3 syllables)
        "ts_000005",  # zai jian (2 syllables)
        "ts_000010",  # 4+ syllables
        "ts_000015",  # longer
    ]

    for sample in dataset:
        if sample.id not in sample_ids:
            continue

        print(f"\n{'='*60}")
        print(f"Sample: {sample.id}")
        print(f"Romanization: {sample.romanization}")
        print(f"Syllables: {len(sample.syllables)}")

        # Load audio
        audio, audio_seg = load_audio_pydub(sample.audio_path)
        duration_ms = int(len(audio) / 16000 * 1000)
        print(f"Duration: {duration_ms}ms")

        # Trim silence
        speech_start, speech_end = trim_silence(audio)
        print(f"Speech region: {speech_start}ms - {speech_end}ms")

        # Segment by voicing gaps (works better for TTS than energy)
        n_syllables = len(sample.syllables)
        spans = segment_by_voicing(audio, n_syllables)

        print(f"\nDetected boundaries:")
        for i, span in enumerate(spans):
            pinyin = sample.syllables[i].pinyin if i < len(sample.syllables) else "?"
            print(f"  {i}: {span.start_ms:4d}ms - {span.end_ms:4d}ms  ({pinyin})")

        # Save full audio
        sample_dir = output_dir / sample.id
        sample_dir.mkdir(exist_ok=True)

        full_path = sample_dir / "full.wav"
        audio_seg.export(str(full_path), format="wav")
        print(f"\nSaved: {full_path}")

        # Save individual syllables
        for i, span in enumerate(spans):
            pinyin = sample.syllables[i].pinyin if i < len(sample.syllables) else f"syl{i}"
            # Clean pinyin for filename
            clean_pinyin = pinyin.replace("ǐ", "i3").replace("ǎ", "a3").replace("ò", "o4")
            clean_pinyin = clean_pinyin.replace("ǜ", "v4").replace("ā", "a1").replace("é", "e2")
            clean_pinyin = clean_pinyin.replace("è", "e4").replace("ì", "i4").replace("ù", "u4")
            clean_pinyin = clean_pinyin.replace("á", "a2").replace("ú", "u2").replace("ó", "o2")
            clean_pinyin = clean_pinyin.replace("ī", "i1").replace("ū", "u1").replace("ē", "e1")
            clean_pinyin = clean_pinyin.replace("ō", "o1").replace("ǐ", "i3").replace("ǔ", "u3")
            clean_pinyin = clean_pinyin.replace("ě", "e3").replace("ǒ", "o3")

            syl_path = sample_dir / f"{i:02d}_{clean_pinyin}.wav"
            save_audio_segment(audio_seg, span.start_ms, span.end_ms, syl_path)
            print(f"Saved: {syl_path.name} ({span.end_ms - span.start_ms}ms)")

        # Save metadata
        meta = {
            "id": sample.id,
            "romanization": sample.romanization,
            "duration_ms": duration_ms,
            "speech_region": {"start_ms": speech_start, "end_ms": speech_end},
            "syllables": [
                {
                    "index": i,
                    "pinyin": sample.syllables[i].pinyin if i < len(sample.syllables) else None,
                    "tone": sample.syllables[i].tone_underlying if i < len(sample.syllables) else None,
                    "start_ms": span.start_ms,
                    "end_ms": span.end_ms,
                }
                for i, span in enumerate(spans)
            ],
        }
        meta_path = sample_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"Saved: {meta_path.name}")

    print(f"\n{'='*60}")
    print(f"Done! Check splits in: {output_dir}")


if __name__ == "__main__":
    main()
