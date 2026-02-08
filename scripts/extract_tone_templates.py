#!/usr/bin/env python3
"""Extract tone templates from TTS reference audio.

This script loads TTS audio from app assets, extracts pitch contours,
and generates tone templates for use in scoring.

Usage:
    python scripts/extract_tone_templates.py
    python scripts/extract_tone_templates.py --voice male
    python scripts/extract_tone_templates.py --output templates.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Add mandarin_grader to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "mandarin_grader"))

from mandarin_grader import SentenceDataset, apply_tone_sandhi, extract_f0_pyin
from mandarin_grader.data import load_audio
from mandarin_grader.pitch import hz_to_semitones, normalize_f0

# Paths relative to repo root
REPO_ROOT = Path(__file__).parent.parent
SENTENCES_JSON = REPO_ROOT / "apps" / "mobile_flutter" / "assets" / "datasets" / "sentences.zh.json"
AUDIO_DIR = REPO_ROOT / "apps" / "mobile_flutter" / "assets" / "examples"
OUTPUT_DIR = REPO_ROOT / "tools" / "mandarin_grader" / "mandarin_grader" / "data"


def main():
    parser = argparse.ArgumentParser(description="Extract tone templates from TTS audio")
    parser.add_argument("--voice", choices=["male", "female"], default="female",
                       help="Voice to use for template extraction")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "tone_templates.json",
                       help="Output path for templates")
    parser.add_argument("--k", type=int, default=20,
                       help="Number of points in contour")
    args = parser.parse_args()

    print(f"Loading dataset from {SENTENCES_JSON}...")
    dataset = SentenceDataset.from_app_assets(
        sentences_json=SENTENCES_JSON,
        audio_dir=AUDIO_DIR,
        voice=args.voice,
    )
    print(f"Loaded {len(dataset)} samples")

    # Collect contours by tone
    contours_by_tone: dict[int, list[np.ndarray]] = defaultdict(list)

    for sample in dataset:
        try:
            # Load and process audio
            audio = load_audio(sample.audio_path)
            f0_hz, voicing = extract_f0_pyin(audio)

            # Normalize F0
            semitones = hz_to_semitones(f0_hz, ref_hz=100.0)
            f0_norm = normalize_f0(semitones, voicing)

            # Apply sandhi to get surface tones
            syllables = apply_tone_sandhi(sample.syllables)

            # Simple: divide audio evenly among syllables
            n_syllables = len(syllables)
            if n_syllables == 0:
                continue

            n_frames = len(f0_norm)
            frames_per_syllable = n_frames // n_syllables

            for i, syl in enumerate(syllables):
                start_frame = i * frames_per_syllable
                end_frame = (i + 1) * frames_per_syllable if i < n_syllables - 1 else n_frames

                syl_f0 = f0_norm[start_frame:end_frame]
                syl_voicing = voicing[start_frame:end_frame]

                if len(syl_f0) < 3:
                    continue

                # Simple resampling to K points
                voiced_mask = syl_voicing > 0.5
                if voiced_mask.sum() < 3:
                    continue

                voiced_f0 = syl_f0[voiced_mask]
                resampled = np.interp(
                    np.linspace(0, len(voiced_f0) - 1, args.k),
                    np.arange(len(voiced_f0)),
                    voiced_f0
                )

                contours_by_tone[syl.tone_surface].append(resampled)

        except Exception as e:
            print(f"Error processing {sample.id}: {e}")
            continue

    # Compute mean templates
    templates = {}
    for tone, contours in sorted(contours_by_tone.items()):
        if len(contours) < 5:
            print(f"Tone {tone}: only {len(contours)} samples, skipping")
            continue

        stacked = np.stack(contours)
        mean_contour = np.mean(stacked, axis=0)
        std_contour = np.std(stacked, axis=0)

        templates[str(tone)] = {
            "mean": mean_contour.tolist(),
            "std": std_contour.tolist(),
            "n_samples": len(contours),
        }
        print(f"Tone {tone}: {len(contours)} samples")

    # Save templates
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "version": "1.0",
            "k": args.k,
            "voice": args.voice,
            "templates": templates,
        }, f, indent=2)

    print(f"\nTemplates saved to {args.output}")


if __name__ == "__main__":
    main()
