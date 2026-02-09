#!/usr/bin/env python3
"""Generate synthetic training data from syllable lexicon.

This script creates training samples by concatenating syllable audio files,
providing ground-truth tone labels for supervised learning.

Usage:
    python generate_training_data.py --output data/synthetic
    python generate_training_data.py --voices female1 male1 --augment
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add parent package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from mandarin_grader.data.lexicon import SyllableLexicon
from mandarin_grader.data.dataloader import parse_romanization
from mandarin_grader.data.synthesis import (
    AugmentationConfig,
    SyntheticSample,
    create_synthetic_sample,
    save_synthetic_sample,
)


def load_all_sentences(sentences_json: Path) -> list[dict]:
    """Load all sentences from the dataset.

    Args:
        sentences_json: Path to sentences.zh.json

    Returns:
        List of sentence dictionaries
    """
    with open(sentences_json, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("items", [])


def generate_training_data(
    sentences_json: Path,
    lexicon_path: Path,
    output_dir: Path,
    voices: list[str],
    augment: bool = False,
    n_augmentations: int = 2,
) -> dict:
    """Generate synthetic training data.

    Args:
        sentences_json: Path to sentences.zh.json
        lexicon_path: Path to syllable lexicon
        output_dir: Output directory
        voices: List of voice IDs to generate
        augment: Whether to generate augmented variants
        n_augmentations: Number of augmented variants per sample

    Returns:
        Statistics dict
    """
    # Load lexicon
    print(f"Loading lexicon from {lexicon_path}...")
    lexicon = SyllableLexicon.load(lexicon_path)
    print(f"Loaded {len(lexicon)} entries")

    # Load sentences
    print(f"Loading sentences from {sentences_json}...")
    sentences = load_all_sentences(sentences_json)
    print(f"Loaded {len(sentences)} sentences")

    # Create output directories
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Track samples and stats
    samples_metadata = []
    stats = {
        "total_samples": 0,
        "total_syllables": 0,
        "tone_counts": {str(i): 0 for i in range(5)},
        "failed": 0,
    }

    for i, sentence in enumerate(sentences):
        sid = sentence.get("id", f"sent_{i}")
        text = sentence.get("text", "")
        romanization = sentence.get("romanization", "")

        if not romanization:
            continue

        syllables = parse_romanization(romanization, text)
        if not syllables:
            continue

        # Generate for each voice
        for voice in voices:
            try:
                # Base sample (no augmentation)
                sample = create_synthetic_sample(
                    sample_id=f"{sid}_{voice}",
                    syllables=syllables,
                    lexicon=lexicon,
                    voice=voice,
                    augmentations=AugmentationConfig(),
                )

                if len(sample.audio) == 0:
                    stats["failed"] += 1
                    continue

                # Save audio
                audio_path = audio_dir / f"{sid}_{voice}.wav"
                save_synthetic_sample(sample, audio_path)

                # Extract tone labels
                tones = [s.tone_surface for s in syllables]

                # Create metadata
                meta = {
                    "id": f"{sid}_{voice}",
                    "sentence_id": sid,
                    "audio_path": str(audio_path.relative_to(output_dir)),
                    "text": text,
                    "romanization": romanization,
                    "voice": voice,
                    "tones": tones,
                    "n_syllables": len(tones),
                    "duration_ms": len(sample.audio) / sample.sample_rate * 1000,
                    "augmented": False,
                    "boundaries": [
                        {"start_ms": s.start_ms, "end_ms": s.end_ms}
                        for s in sample.ground_truth_spans
                    ],
                }
                samples_metadata.append(meta)

                # Update stats
                stats["total_samples"] += 1
                stats["total_syllables"] += len(tones)
                for t in tones:
                    stats["tone_counts"][str(t)] += 1

                # Generate augmented variants
                if augment:
                    for aug_idx in range(n_augmentations):
                        aug_config = AugmentationConfig(
                            speed_variation=0.1,
                            gap_ms=np.random.randint(10, 30),
                            noise_snr_db=30 if aug_idx % 2 == 0 else None,
                        )

                        aug_sample = create_synthetic_sample(
                            sample_id=f"{sid}_{voice}_aug{aug_idx}",
                            syllables=syllables,
                            lexicon=lexicon,
                            voice=voice,
                            augmentations=aug_config,
                        )

                        aug_audio_path = audio_dir / f"{sid}_{voice}_aug{aug_idx}.wav"
                        save_synthetic_sample(aug_sample, aug_audio_path)

                        aug_meta = {
                            "id": f"{sid}_{voice}_aug{aug_idx}",
                            "sentence_id": sid,
                            "audio_path": str(aug_audio_path.relative_to(output_dir)),
                            "text": text,
                            "romanization": romanization,
                            "voice": voice,
                            "tones": tones,
                            "n_syllables": len(tones),
                            "duration_ms": len(aug_sample.audio) / aug_sample.sample_rate * 1000,
                            "augmented": True,
                            "boundaries": [
                                {"start_ms": s.start_ms, "end_ms": s.end_ms}
                                for s in aug_sample.ground_truth_spans
                            ],
                        }
                        samples_metadata.append(aug_meta)

                        stats["total_samples"] += 1
                        stats["total_syllables"] += len(tones)
                        for t in tones:
                            stats["tone_counts"][str(t)] += 1

            except Exception as e:
                print(f"Error processing {sid}/{voice}: {e}")
                stats["failed"] += 1

        # Progress
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(sentences)} sentences...")

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "samples": samples_metadata,
            "stats": stats,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nSaved metadata to {metadata_path}")
    print(f"\nStatistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Total syllables: {stats['total_syllables']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Tone distribution:")
    for tone, count in stats["tone_counts"].items():
        pct = count / max(stats["total_syllables"], 1) * 100
        print(f"    Tone {tone}: {count} ({pct:.1f}%)")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data from syllable lexicon"
    )
    parser.add_argument(
        "--sentences",
        type=Path,
        default=Path("apps/mobile_flutter/assets/datasets/sentences.zh.json"),
        help="Path to sentences.zh.json",
    )
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=Path("tools/mandarin_grader/data/syllables_v2"),
        help="Path to syllable lexicon",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tools/mandarin_grader/data/synthetic"),
        help="Output directory",
    )
    parser.add_argument(
        "--voices",
        nargs="+",
        default=["female1", "male1"],
        help="Voices to generate",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Generate augmented variants",
    )
    parser.add_argument(
        "--n-augmentations",
        type=int,
        default=2,
        help="Number of augmented variants per sample",
    )

    args = parser.parse_args()

    generate_training_data(
        sentences_json=args.sentences,
        lexicon_path=args.lexicon,
        output_dir=args.output,
        voices=args.voices,
        augment=args.augment,
        n_augmentations=args.n_augmentations,
    )


if __name__ == "__main__":
    main()
