#!/usr/bin/env python3
"""Generate sample reconstructed sentences for quality review.

This script synthesizes sentences from the syllable lexicon to verify
that the concatenation produces realistic audio.

Usage:
    python generate_samples.py --output data/samples
    python generate_samples.py --sentences 5 --output data/samples
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
    synthesize_sentence,
    save_synthetic_sample,
    create_synthetic_sample,
)


def load_sentences(sentences_json: Path, n_samples: int = 5) -> list[dict]:
    """Load sample sentences from the dataset.

    Args:
        sentences_json: Path to sentences.zh.json
        n_samples: Number of sentences to sample

    Returns:
        List of sentence dictionaries
    """
    with open(sentences_json, encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("items", [])

    # Sample a variety of sentence lengths
    short = [i for i in items if len(i.get("romanization", "").split()) <= 3]
    medium = [i for i in items if 3 < len(i.get("romanization", "").split()) <= 6]
    long = [i for i in items if len(i.get("romanization", "").split()) > 6]

    # Select diverse samples
    samples = []
    for category in [short, medium, long]:
        if category:
            np.random.shuffle(category)
            samples.extend(category[:max(1, n_samples // 3)])

    # Ensure we have enough
    if len(samples) < n_samples:
        remaining = [i for i in items if i not in samples]
        np.random.shuffle(remaining)
        samples.extend(remaining[:n_samples - len(samples)])

    return samples[:n_samples]


def generate_samples(
    sentences_json: Path,
    lexicon_path: Path,
    output_dir: Path,
    n_samples: int = 5,
    voices: list[str] = None,
) -> None:
    """Generate sample reconstructed sentences.

    Args:
        sentences_json: Path to sentences.zh.json
        lexicon_path: Path to syllable lexicon
        output_dir: Output directory for samples
        n_samples: Number of samples to generate
        voices: List of voices to generate (default: female and male)
    """
    if voices is None:
        voices = ["female", "male"]

    # Load lexicon
    print(f"Loading lexicon from {lexicon_path}...")
    lexicon = SyllableLexicon.load(lexicon_path)
    print(f"Loaded {len(lexicon)} entries")

    # Load sample sentences
    print(f"Loading sentences from {sentences_json}...")
    sentences = load_sentences(sentences_json, n_samples)
    print(f"Selected {len(sentences)} sentences")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate samples
    report_lines = [
        "# Sample Reconstructed Sentences",
        "",
        "These sentences were synthesized by concatenating individual syllable audio files.",
        "",
        "## Samples",
        "",
    ]

    for i, sentence in enumerate(sentences, 1):
        sid = sentence.get("id", f"sample_{i}")
        text = sentence.get("text", "")
        romanization = sentence.get("romanization", "")
        gloss = sentence.get("gloss", {}).get("en", "")

        try:
            print(f"\n[{i}/{len(sentences)}] Processing: {text} ({romanization})")
        except UnicodeEncodeError:
            print(f"\n[{i}/{len(sentences)}] Processing: {sid}")

        syllables = parse_romanization(romanization, text)
        if not syllables:
            print(f"  Warning: No syllables parsed for {text}")
            continue

        report_lines.extend([
            f"### {i}. {text}",
            "",
            f"- **Pinyin:** {romanization}",
            f"- **English:** {gloss}",
            f"- **Syllables:** {len(syllables)}",
            "",
        ])

        for voice in voices:
            # Create sample without augmentation
            sample = create_synthetic_sample(
                sample_id=f"{sid}_{voice}",
                syllables=syllables,
                lexicon=lexicon,
                voice=voice,
                augmentations=AugmentationConfig(),  # No augmentation
            )

            # Save audio
            output_path = output_dir / f"{sid}_{voice}.wav"
            save_synthetic_sample(sample, output_path)

            # Calculate duration
            duration_ms = len(sample.audio) / sample.sample_rate * 1000

            print(f"  Saved {output_path.name} ({duration_ms:.0f}ms)")

            report_lines.append(f"- **{voice.title()} voice:** `{output_path.name}` ({duration_ms:.0f}ms)")

            # Also create with augmentation
            aug_sample = create_synthetic_sample(
                sample_id=f"{sid}_{voice}_aug",
                syllables=syllables,
                lexicon=lexicon,
                voice=voice,
                augmentations=AugmentationConfig(
                    speed_variation=0.1,
                    gap_ms=20,
                ),
            )

            aug_output_path = output_dir / f"{sid}_{voice}_aug.wav"
            save_synthetic_sample(aug_sample, aug_output_path)
            aug_duration_ms = len(aug_sample.audio) / aug_sample.sample_rate * 1000

            print(f"  Saved {aug_output_path.name} ({aug_duration_ms:.0f}ms, augmented)")

        # Add boundary info
        report_lines.extend([
            "",
            "**Boundaries:**",
            "",
            "| Syllable | Start (ms) | End (ms) | Duration (ms) |",
            "|----------|------------|----------|---------------|",
        ])

        for j, (syl, span) in enumerate(zip(syllables, sample.ground_truth_spans)):
            report_lines.append(
                f"| {syl.pinyin} | {span.start_ms} | {span.end_ms} | {span.end_ms - span.start_ms} |"
            )

        report_lines.append("")

    # Save report
    report_path = output_dir / "README.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nReport saved to {report_path}")
    print(f"Total samples: {len(sentences) * len(voices) * 2} (with and without augmentation)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate sample reconstructed sentences"
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
        default=Path("tools/mandarin_grader/data/samples"),
        help="Output directory for samples",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
        help="Number of sentences to sample",
    )
    parser.add_argument(
        "--voices",
        nargs="+",
        default=["female1", "male1"],
        help="Voices to generate (female1, female2, male1, male2)",
    )

    args = parser.parse_args()

    np.random.seed(42)  # For reproducibility

    generate_samples(
        sentences_json=args.sentences,
        lexicon_path=args.lexicon,
        output_dir=args.output,
        n_samples=args.n_samples,
        voices=args.voices,
    )


if __name__ == "__main__":
    main()
