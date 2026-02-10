#!/usr/bin/env python3
"""Evaluate SyllablePredictorV3 on real TTS data.

This script tests the model on the actual TTS audio files from the app,
using the inference strategy designed for mobile deployment.

Usage:
    python eval_tts.py --checkpoint checkpoints_v3_run1/best_model.pt
    python eval_tts.py --checkpoint checkpoints_v3_run1/best_model.pt --voice female
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SENTENCES_JSON = PROJECT_ROOT / "apps/mobile_flutter/assets/datasets/sentences.zh.json"
AUDIO_DIR = PROJECT_ROOT / "apps/mobile_flutter/assets/examples"
DEFAULT_CHECKPOINT = Path(__file__).parent.parent / "checkpoints_v3_run1" / "best_model.pt"


def load_audio_file(path: Path, target_sr: int = 16000) -> np.ndarray:
    """Load audio file (mp3 or wav) and convert to float32."""
    import librosa

    samples, _ = librosa.load(str(path), sr=target_sr, mono=True)
    return samples.astype(np.float32)


def add_inference_noise(audio: np.ndarray, snr_db: float = 30.0) -> np.ndarray:
    """Add noise to audio to match training distribution.

    Training used noise augmentation which raised the mel floor.
    Without noise, clean audio has many floor values the model hasn't seen.
    """
    signal_power = np.mean(audio ** 2)
    if signal_power < 1e-10:
        return audio
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.randn(len(audio)).astype(np.float32) * np.sqrt(noise_power)
    return audio + noise


def load_sentences(sentences_json: Path) -> list[dict]:
    """Load sentence metadata."""
    with open(sentences_json, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("items", [])


def evaluate_on_tts(
    model,
    vocab,
    config,
    sentences: list[dict],
    audio_dir: Path,
    voice: str,
    device: str,
    add_noise: bool = True,
    noise_snr_db: float = 30.0,
) -> dict:
    """Evaluate model on TTS audio files using grader mode.

    Grader mode strategy:
    1. Load full audio
    2. Estimate syllable positions (uniform distribution)
    3. For each syllable, provide CORRECT previous pinyin as context
    4. Check if probability for correct syllable/tone is high enough

    This simulates real grading where we know what the user should say.
    """
    import torch
    import torch.nn.functional as F
    from mandarin_grader.data.dataloader import parse_romanization
    from mandarin_grader.data.lexicon import _remove_tone_marks
    from mandarin_grader.data.autoregressive_dataset import (
        estimate_syllable_positions, get_inference_chunk
    )
    from mandarin_grader.model.syllable_predictor_v3 import extract_mel_spectrogram

    model.eval()

    results = {
        "total_syllables": 0,
        # Argmax accuracy (old method)
        "argmax_syl_correct": 0,
        "argmax_tone_correct": 0,
        # Probability-based metrics
        "syl_probs": [],  # P(correct syllable)
        "tone_probs": [],  # P(correct tone)
        "syl_ranks": [],  # Rank of correct syllable (1 = best)
        "tone_ranks": [],  # Rank of correct tone (1 = best)
        # Per-tone stats
        "per_tone_probs": defaultdict(list),
        "per_tone_total": defaultdict(int),
        "samples": [],
    }

    voice_dir = audio_dir / voice

    for item in sentences:
        audio_path = voice_dir / f"{item['id']}.mp3"
        if not audio_path.exists():
            continue

        # Parse expected syllables
        syllables = parse_romanization(item.get("romanization", ""), item.get("text", ""))
        if not syllables:
            continue

        # Load audio
        try:
            audio = load_audio_file(audio_path, config.sample_rate)
        except Exception as e:
            print(f"Failed to load {audio_path}: {e}")
            continue

        # Estimate syllable positions
        positions = estimate_syllable_positions(
            audio_samples=len(audio),
            n_syllables=len(syllables),
            sample_rate=config.sample_rate,
        )

        # Predict each syllable with teacher forcing (correct context)
        context = []
        sample_results = []

        for i, (syl, pos) in enumerate(zip(syllables, positions)):
            # Extract chunk
            chunk = get_inference_chunk(
                audio, pos,
                chunk_samples=int(1.0 * config.sample_rate),
                margin_samples=int(0.1 * config.sample_rate),
            )

            # Add noise to match training distribution
            if add_noise:
                chunk = add_inference_noise(chunk, noise_snr_db)

            # Extract mel
            mel = extract_mel_spectrogram(chunk, config)
            mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).to(device)

            # Encode context (CORRECT previous pinyin - teacher forcing)
            pinyin_ids = vocab.encode_sequence(context, add_bos=True)
            pinyin_tensor = torch.tensor(pinyin_ids, dtype=torch.long).unsqueeze(0).to(device)

            # Predict
            with torch.no_grad():
                syl_logits, tone_logits = model(mel_tensor, pinyin_tensor)

                # Softmax to get probabilities
                syl_probs = F.softmax(syl_logits[0], dim=0)
                tone_probs = F.softmax(tone_logits[0], dim=0)

                # Argmax predictions
                pred_syl_idx = syl_logits[0].argmax().item()
                pred_tone = tone_logits[0].argmax().item()

            # Expected
            expected_pinyin = _remove_tone_marks(syl.pinyin)
            expected_syl_idx = vocab.encode(expected_pinyin)
            expected_tone = syl.tone_surface

            # Get probability of correct answer
            correct_syl_prob = syl_probs[expected_syl_idx].item()
            correct_tone_prob = tone_probs[expected_tone].item()

            # Get rank of correct answer (1 = best)
            syl_rank = (syl_probs > syl_probs[expected_syl_idx]).sum().item() + 1
            tone_rank = (tone_probs > tone_probs[expected_tone]).sum().item() + 1

            # Argmax correctness
            syl_argmax_correct = (pred_syl_idx == expected_syl_idx)
            tone_argmax_correct = (pred_tone == expected_tone)

            # Update results
            results["total_syllables"] += 1
            if syl_argmax_correct:
                results["argmax_syl_correct"] += 1
            if tone_argmax_correct:
                results["argmax_tone_correct"] += 1

            results["syl_probs"].append(correct_syl_prob)
            results["tone_probs"].append(correct_tone_prob)
            results["syl_ranks"].append(syl_rank)
            results["tone_ranks"].append(tone_rank)

            results["per_tone_total"][expected_tone] += 1
            results["per_tone_probs"][expected_tone].append(correct_tone_prob)

            sample_results.append({
                "expected_syl": expected_pinyin,
                "predicted_syl": vocab.decode(pred_syl_idx),
                "expected_tone": expected_tone,
                "predicted_tone": pred_tone,
                "syl_prob": correct_syl_prob,
                "tone_prob": correct_tone_prob,
                "syl_rank": syl_rank,
                "tone_rank": tone_rank,
            })

            # Add to context (always use expected - teacher forcing)
            context.append(expected_pinyin)

        results["samples"].append({
            "id": item["id"],
            "text": item.get("text", ""),
            "syllables": sample_results,
        })

    # Calculate metrics
    n = max(results["total_syllables"], 1)

    # Argmax accuracy
    results["argmax_syl_accuracy"] = results["argmax_syl_correct"] / n
    results["argmax_tone_accuracy"] = results["argmax_tone_correct"] / n

    # Probability stats
    results["mean_syl_prob"] = np.mean(results["syl_probs"]) if results["syl_probs"] else 0
    results["mean_tone_prob"] = np.mean(results["tone_probs"]) if results["tone_probs"] else 0
    results["median_syl_prob"] = np.median(results["syl_probs"]) if results["syl_probs"] else 0
    results["median_tone_prob"] = np.median(results["tone_probs"]) if results["tone_probs"] else 0

    # Rank stats (how often is correct answer in top-k)
    results["syl_top1"] = sum(1 for r in results["syl_ranks"] if r == 1) / n
    results["syl_top3"] = sum(1 for r in results["syl_ranks"] if r <= 3) / n
    results["syl_top5"] = sum(1 for r in results["syl_ranks"] if r <= 5) / n
    results["tone_top1"] = sum(1 for r in results["tone_ranks"] if r == 1) / n
    results["tone_top2"] = sum(1 for r in results["tone_ranks"] if r <= 2) / n

    # Threshold-based accuracy (would we accept this as correct?)
    for thresh in [0.1, 0.2, 0.3, 0.5]:
        results[f"syl_above_{thresh}"] = sum(1 for p in results["syl_probs"] if p >= thresh) / n
        results[f"tone_above_{thresh}"] = sum(1 for p in results["tone_probs"] if p >= thresh) / n

    # Per-tone probability stats
    results["per_tone_mean_prob"] = {
        t: np.mean(results["per_tone_probs"][t]) if results["per_tone_probs"][t] else 0
        for t in range(5)
    }

    return results


def main():
    import torch

    parser = argparse.ArgumentParser(description="Evaluate on TTS data")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--voice", choices=["female", "male"], default="female")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-noise", action="store_true", help="Disable inference noise (for debugging)")
    parser.add_argument("--noise-snr", type=float, default=30.0, help="SNR for inference noise in dB")
    args = parser.parse_args()

    print("=" * 60)
    print("TTS Evaluation")
    print("=" * 60)

    # Load model
    from mandarin_grader.model.syllable_predictor_v3 import (
        SyllablePredictorV3, SyllablePredictorConfig, SyllableVocab
    )

    config = SyllablePredictorConfig()
    model = SyllablePredictorV3(config).to(args.device)
    vocab = SyllableVocab()

    if args.checkpoint.exists():
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint}")
        if "epoch" in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if "val_syl_accuracy" in checkpoint:
            print(f"  Val Syl Acc: {checkpoint['val_syl_accuracy']:.4f}")
            print(f"  Val Tone Acc: {checkpoint['val_tone_accuracy']:.4f}")
    else:
        print(f"WARNING: Checkpoint not found: {args.checkpoint}")
        print("Using untrained model")

    # Load sentences
    sentences = load_sentences(SENTENCES_JSON)
    if args.max_samples:
        sentences = sentences[:args.max_samples]
    print(f"Loaded {len(sentences)} sentences")
    print(f"Voice: {args.voice}")

    # Evaluate
    results = evaluate_on_tts(
        model, vocab, config, sentences, AUDIO_DIR, args.voice, args.device,
        add_noise=not args.no_noise, noise_snr_db=args.noise_snr
    )

    # Print results
    print()
    print("=" * 60)
    print("RESULTS (Grader Mode - Teacher Forcing)")
    print("=" * 60)
    print(f"Total syllables: {results['total_syllables']}")

    print()
    print("--- Argmax Accuracy (old method) ---")
    print(f"Syllable: {results['argmax_syl_accuracy']:.4f}")
    print(f"Tone:     {results['argmax_tone_accuracy']:.4f}")

    print()
    print("--- Probability of Correct Answer ---")
    print(f"Syllable - Mean: {results['mean_syl_prob']:.4f}, Median: {results['median_syl_prob']:.4f}")
    print(f"Tone     - Mean: {results['mean_tone_prob']:.4f}, Median: {results['median_tone_prob']:.4f}")

    print()
    print("--- Rank of Correct Answer ---")
    print(f"Syllable - Top1: {results['syl_top1']:.4f}, Top3: {results['syl_top3']:.4f}, Top5: {results['syl_top5']:.4f}")
    print(f"Tone     - Top1: {results['tone_top1']:.4f}, Top2: {results['tone_top2']:.4f}")

    print()
    print("--- Threshold-based Accuracy (P >= thresh) ---")
    print(f"{'Threshold':<12} {'Syllable':>10} {'Tone':>10}")
    for thresh in [0.1, 0.2, 0.3, 0.5]:
        syl_acc = results[f"syl_above_{thresh}"]
        tone_acc = results[f"tone_above_{thresh}"]
        print(f"{thresh:<12} {syl_acc:>10.4f} {tone_acc:>10.4f}")

    print()
    print("--- Per-Tone Stats ---")
    print(f"{'Tone':<6} {'Count':>8} {'Mean P':>10}")
    for t in range(5):
        total = results["per_tone_total"].get(t, 0)
        mean_p = results["per_tone_mean_prob"].get(t, 0)
        print(f"{t:<6} {total:>8} {mean_p:>10.4f}")

    # Show low-probability samples
    print()
    print("Lowest probability samples (first 10):")
    all_samples = []
    for sample in results["samples"]:
        for syl in sample["syllables"]:
            all_samples.append((sample["id"], syl))
    all_samples.sort(key=lambda x: x[1]["syl_prob"])
    for sid, syl in all_samples[:10]:
        print(f"  [{sid}]: {syl['expected_syl']}{syl['expected_tone']} "
              f"P(syl)={syl['syl_prob']:.3f} rank={syl['syl_rank']}, "
              f"P(tone)={syl['tone_prob']:.3f} rank={syl['tone_rank']}, "
              f"pred={syl['predicted_syl']}{syl['predicted_tone']}")


if __name__ == "__main__":
    main()
