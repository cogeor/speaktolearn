#!/usr/bin/env python3
"""Evaluate SyllablePredictorV3 with top-k accuracy metrics.

This script evaluates the model on:
1. AISHELL-3 validation data (from tar source)
2. OpenAI TTS examples (from apps/mobile_flutter/assets/examples/)

Metrics reported:
- Hanzi (syllable): Top-1, Top-2, Top-5, Top-10 accuracy
- Tone: Top-1, Top-2 accuracy

Usage:
    # Evaluate on AISHELL-3 validation data
    python eval_single_sentence.py --source aishell3

    # Evaluate on TTS examples
    python eval_single_sentence.py --source tts --voice female

    # Both sources
    python eval_single_sentence.py --source all

    # Single sentence demo
    python eval_single_sentence.py --source tts --sentence-id ts_000001 --verbose
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Fix Windows console encoding for Chinese characters
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent))

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SENTENCES_JSON = PROJECT_ROOT / "apps/mobile_flutter/assets/datasets/sentences.zh.json"
AUDIO_DIR = PROJECT_ROOT / "apps/mobile_flutter/assets/examples"
AISHELL_TAR_DIR = Path(__file__).parent.parent / "datasets" / "aishell3_tar"
DEFAULT_CHECKPOINT = Path(__file__).parent.parent / "checkpoints_v3_10M_aishell3_tar_mel_bs128" / "best_model.pt"


@dataclass
class TopKMetrics:
    """Top-k accuracy metrics."""
    total: int = 0

    # Hanzi (syllable) top-k
    hanzi_top1: int = 0
    hanzi_top2: int = 0
    hanzi_top5: int = 0
    hanzi_top10: int = 0

    # Tone top-k
    tone_top1: int = 0
    tone_top2: int = 0

    def update(self, hanzi_rank: int, tone_rank: int):
        """Update metrics with a single prediction."""
        self.total += 1
        if hanzi_rank <= 1:
            self.hanzi_top1 += 1
        if hanzi_rank <= 2:
            self.hanzi_top2 += 1
        if hanzi_rank <= 5:
            self.hanzi_top5 += 1
        if hanzi_rank <= 10:
            self.hanzi_top10 += 1
        if tone_rank <= 1:
            self.tone_top1 += 1
        if tone_rank <= 2:
            self.tone_top2 += 1

    def as_dict(self) -> dict:
        """Return metrics as dictionary with accuracies."""
        n = max(self.total, 1)
        return {
            "total_syllables": self.total,
            "hanzi_top1_acc": self.hanzi_top1 / n,
            "hanzi_top2_acc": self.hanzi_top2 / n,
            "hanzi_top5_acc": self.hanzi_top5 / n,
            "hanzi_top10_acc": self.hanzi_top10 / n,
            "tone_top1_acc": self.tone_top1 / n,
            "tone_top2_acc": self.tone_top2 / n,
        }

    def print_summary(self, title: str = "Results"):
        """Print formatted summary."""
        d = self.as_dict()
        print()
        print("=" * 60)
        print(title)
        print("=" * 60)
        print(f"Total syllables: {d['total_syllables']}")
        print()
        print("Hanzi (Syllable) Accuracy:")
        print(f"  Top-1:  {d['hanzi_top1_acc']:.4f} ({self.hanzi_top1}/{self.total})")
        print(f"  Top-2:  {d['hanzi_top2_acc']:.4f} ({self.hanzi_top2}/{self.total})")
        print(f"  Top-5:  {d['hanzi_top5_acc']:.4f} ({self.hanzi_top5}/{self.total})")
        print(f"  Top-10: {d['hanzi_top10_acc']:.4f} ({self.hanzi_top10}/{self.total})")
        print()
        print("Tone Accuracy:")
        print(f"  Top-1: {d['tone_top1_acc']:.4f} ({self.tone_top1}/{self.total})")
        print(f"  Top-2: {d['tone_top2_acc']:.4f} ({self.tone_top2}/{self.total})")


def load_audio_file(path: Path, target_sr: int = 16000) -> np.ndarray:
    """Load audio file (mp3 or wav) and convert to float32."""
    import librosa
    samples, _ = librosa.load(str(path), sr=target_sr, mono=True)
    return samples.astype(np.float32)


def add_inference_noise(audio: np.ndarray, snr_db: float = 30.0) -> np.ndarray:
    """Add noise to match training distribution."""
    signal_power = np.mean(audio ** 2)
    if signal_power < 1e-10:
        return audio
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.randn(len(audio)).astype(np.float32) * np.sqrt(noise_power)
    return audio + noise


def get_rank(probs: np.ndarray, target_idx: int) -> int:
    """Get rank of target index (1 = best)."""
    return int((probs > probs[target_idx]).sum() + 1)


def evaluate_single_sentence(
    model,
    vocab,
    config,
    audio: np.ndarray,
    syllables: list,
    device: str,
    context_mode: str = "position",
    add_noise: bool = True,
    verbose: bool = False,
) -> list[dict]:
    """Evaluate a single sentence.

    Args:
        model: SyllablePredictorV3 model
        vocab: SyllableVocab
        config: SyllablePredictorConfig
        audio: Audio samples (float32)
        syllables: List of TargetSyllable
        device: torch device
        context_mode: "position" or "pinyin"
        add_noise: Whether to add inference noise
        verbose: Print per-syllable details

    Returns:
        List of per-syllable result dicts
    """
    import torch
    import torch.nn.functional as F
    from mandarin_grader.data.lexicon import _remove_tone_marks
    from mandarin_grader.data.autoregressive_dataset import (
        estimate_syllable_positions, get_inference_chunk
    )
    from mandarin_grader.model.syllable_predictor_v3 import extract_mel_spectrogram

    model.eval()
    results = []

    # Estimate syllable positions
    positions = estimate_syllable_positions(
        audio_samples=len(audio),
        n_syllables=len(syllables),
        sample_rate=config.sample_rate,
    )

    context = []

    for i, (syl, pos) in enumerate(zip(syllables, positions)):
        # Extract chunk
        chunk = get_inference_chunk(
            audio, pos,
            chunk_samples=int(1.0 * config.sample_rate),
            margin_samples=int(0.1 * config.sample_rate),
        )

        if add_noise:
            chunk = add_inference_noise(chunk)

        # Extract mel
        mel = extract_mel_spectrogram(chunk, config)
        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).to(device)

        # Encode context based on mode
        if context_mode == "pinyin":
            pinyin_ids = vocab.encode_sequence(context, add_bos=True)
        else:
            # Position mode: [BOS, position_index]
            position_token = 2 + i
            pinyin_ids = [vocab.bos_token, position_token]

        pinyin_tensor = torch.tensor(pinyin_ids, dtype=torch.long).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            syl_logits, tone_logits = model(mel_tensor, pinyin_tensor)
            syl_probs = F.softmax(syl_logits[0], dim=0).cpu().numpy()
            tone_probs = F.softmax(tone_logits[0], dim=0).cpu().numpy()

        # Expected values
        expected_pinyin = _remove_tone_marks(syl.pinyin)
        expected_syl_idx = vocab.encode(expected_pinyin)
        expected_tone = syl.tone_surface

        # Get ranks
        hanzi_rank = get_rank(syl_probs, expected_syl_idx)
        tone_rank = get_rank(tone_probs, expected_tone)

        # Get top-k predictions
        top10_indices = np.argsort(syl_probs)[-10:][::-1]
        top10_syls = [vocab.decode(idx) for idx in top10_indices]

        result = {
            "syllable_idx": i,
            "expected_hanzi": syl.hanzi,
            "expected_pinyin": expected_pinyin,
            "expected_tone": expected_tone,
            "hanzi_rank": hanzi_rank,
            "tone_rank": tone_rank,
            "hanzi_prob": float(syl_probs[expected_syl_idx]),
            "tone_prob": float(tone_probs[expected_tone]),
            "top10_predictions": top10_syls,
            "predicted_tone": int(tone_probs.argmax()),
        }
        results.append(result)

        if verbose:
            status = "✓" if hanzi_rank == 1 else f"rank={hanzi_rank}"
            tone_status = "✓" if tone_rank == 1 else f"rank={tone_rank}"
            print(f"  [{i}] {syl.hanzi} ({expected_pinyin}{expected_tone}): "
                  f"hanzi {status}, tone {tone_status}, "
                  f"top3=[{', '.join(top10_syls[:3])}]")

        # Update context (teacher forcing)
        context.append(expected_pinyin)

    return results


def evaluate_on_aishell3_val(
    model, vocab, config, device: str,
    context_mode: str = "position",
    max_sentences: int | None = None,
    verbose: bool = False,
) -> TopKMetrics:
    """Evaluate on AISHELL-3 validation split."""
    from mandarin_grader.data.aishell_tar_source import AISHELL3TarDataSource
    from mandarin_grader.model.syllable_predictor_v3 import extract_mel_spectrogram
    from mandarin_grader.data.autoregressive_dataset import get_inference_chunk, estimate_syllable_positions
    from mandarin_grader.data.lexicon import _remove_tone_marks
    import torch
    import torch.nn.functional as F

    print("\nLoading AISHELL-3 validation data...")
    source = AISHELL3TarDataSource()

    if not source.is_available(AISHELL_TAR_DIR):
        print(f"AISHELL-3 tar data not found at {AISHELL_TAR_DIR}")
        return TopKMetrics()

    # Load test split (used as validation)
    sentences = source.load(AISHELL_TAR_DIR, split="test", max_sentences=max_sentences)
    mel_cache = source.get_mel_cache()

    print(f"Loaded {len(sentences)} validation sentences")

    metrics = TopKMetrics()
    model.eval()

    for sent_idx, sent in enumerate(sentences):
        if verbose:
            print(f"\n[{sent.id}] {sent.text}")

        # Get precomputed mel
        mel_key = str(sent.audio_path)
        full_mel = mel_cache.get(mel_key)

        if full_mel is None:
            continue

        # Estimate positions based on total samples
        positions = estimate_syllable_positions(
            audio_samples=sent.total_samples,
            n_syllables=len(sent.syllables),
            sample_rate=sent.sample_rate,
        )

        context = []

        for syl_idx, (syl, pos) in enumerate(zip(sent.syllables, positions)):
            # Extract mel chunk (similar to training)
            # Use position to get frame range
            hop_length = 160
            win_length = 400
            chunk_samples = int(1.0 * sent.sample_rate)
            margin_samples = int(0.1 * sent.sample_rate)

            syl_start, syl_end = pos

            # Calculate valid chunk start range
            min_chunk_start = syl_end + margin_samples - chunk_samples
            max_chunk_start = syl_start - margin_samples
            min_chunk_start = max(0, min_chunk_start)
            max_chunk_start = max(0, min(max_chunk_start, sent.total_samples - chunk_samples))

            if min_chunk_start > max_chunk_start:
                syl_mid = (syl_start + syl_end) // 2
                chunk_start = max(0, min(syl_mid - chunk_samples // 2, sent.total_samples - chunk_samples))
            else:
                chunk_start = (min_chunk_start + max_chunk_start) // 2

            # Convert to frames
            target_frames = max(1, 1 + (chunk_samples - win_length) // hop_length)
            start_frame = max(0, chunk_start // hop_length)
            end_frame = start_frame + target_frames

            mel_chunk = full_mel[:, start_frame:end_frame]
            if mel_chunk.shape[1] < target_frames:
                pad = np.zeros((full_mel.shape[0], target_frames - mel_chunk.shape[1]), dtype=np.float32)
                mel_chunk = np.concatenate([mel_chunk, pad], axis=1)

            mel_tensor = torch.tensor(mel_chunk, dtype=torch.float32).unsqueeze(0).to(device)

            # Encode context
            if context_mode == "pinyin":
                pinyin_ids = vocab.encode_sequence(context, add_bos=True)
            else:
                position_token = 2 + syl_idx
                pinyin_ids = [vocab.bos_token, position_token]

            pinyin_tensor = torch.tensor(pinyin_ids, dtype=torch.long).unsqueeze(0).to(device)

            # Predict
            with torch.no_grad():
                syl_logits, tone_logits = model(mel_tensor, pinyin_tensor)
                syl_probs = F.softmax(syl_logits[0], dim=0).cpu().numpy()
                tone_probs = F.softmax(tone_logits[0], dim=0).cpu().numpy()

            # Expected values
            expected_pinyin = _remove_tone_marks(syl.pinyin)
            expected_syl_idx = vocab.encode(expected_pinyin)
            expected_tone = syl.tone_surface

            # Get ranks
            hanzi_rank = get_rank(syl_probs, expected_syl_idx)
            tone_rank = get_rank(tone_probs, expected_tone)

            metrics.update(hanzi_rank, tone_rank)

            if verbose:
                status = "✓" if hanzi_rank == 1 else f"rank={hanzi_rank}"
                tone_status = "✓" if tone_rank == 1 else f"rank={tone_rank}"
                print(f"  [{syl_idx}] {syl.hanzi} ({expected_pinyin}{expected_tone}): "
                      f"hanzi {status}, tone {tone_status}")

            # Update context
            context.append(expected_pinyin)

        # Progress
        if (sent_idx + 1) % 100 == 0:
            print(f"  Processed {sent_idx + 1}/{len(sentences)} sentences...")

    return metrics


def evaluate_on_tts(
    model, vocab, config, device: str,
    voice: str = "female",
    context_mode: str = "position",
    max_sentences: int | None = None,
    sentence_id: str | None = None,
    verbose: bool = False,
) -> TopKMetrics:
    """Evaluate on OpenAI TTS examples."""
    from mandarin_grader.data.dataloader import parse_romanization

    print(f"\nLoading TTS data (voice: {voice})...")

    # Load sentences metadata
    with open(SENTENCES_JSON, encoding="utf-8") as f:
        data = json.load(f)

    sentences = data.get("items", [])

    if sentence_id:
        sentences = [s for s in sentences if s["id"] == sentence_id]
        if not sentences:
            print(f"Sentence {sentence_id} not found")
            return TopKMetrics()
    elif max_sentences:
        sentences = sentences[:max_sentences]

    print(f"Processing {len(sentences)} sentences")

    voice_dir = AUDIO_DIR / voice
    metrics = TopKMetrics()

    for sent_idx, item in enumerate(sentences):
        audio_path = voice_dir / f"{item['id']}.mp3"
        if not audio_path.exists():
            continue

        # Parse syllables
        syllables = parse_romanization(
            item.get("romanization", ""),
            item.get("text", "")
        )
        if not syllables:
            continue

        # Load audio
        try:
            audio = load_audio_file(audio_path, config.sample_rate)
        except Exception as e:
            print(f"Failed to load {audio_path}: {e}")
            continue

        if verbose:
            print(f"\n[{item['id']}] {item.get('text', '')}")

        results = evaluate_single_sentence(
            model, vocab, config, audio, syllables, device,
            context_mode=context_mode,
            add_noise=True,
            verbose=verbose,
        )

        for r in results:
            metrics.update(r["hanzi_rank"], r["tone_rank"])

        # Progress
        if not verbose and (sent_idx + 1) % 50 == 0:
            print(f"  Processed {sent_idx + 1}/{len(sentences)} sentences...")

    return metrics


def main():
    import torch

    parser = argparse.ArgumentParser(
        description="Evaluate SyllablePredictorV3 with top-k accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--source", choices=["aishell3", "tts", "all"], default="all",
                        help="Data source to evaluate on")
    parser.add_argument("--voice", choices=["female", "male"], default="female",
                        help="TTS voice (for --source tts or all)")
    parser.add_argument("--max-sentences", type=int, default=None,
                        help="Maximum sentences to evaluate")
    parser.add_argument("--sentence-id", type=str, default=None,
                        help="Evaluate single sentence by ID (TTS only)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-syllable details")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON file")

    # Model architecture (must match training)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--context-mode", choices=["position", "pinyin"], default="position")

    args = parser.parse_args()

    print("=" * 60)
    print("SyllablePredictorV3 Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Source: {args.source}")
    print(f"Context mode: {args.context_mode}")
    print(f"Device: {args.device}")

    # Load model
    from mandarin_grader.model.syllable_predictor_v3 import (
        SyllablePredictorV3, SyllablePredictorConfig, SyllableVocab
    )

    config = SyllablePredictorConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dim_feedforward=args.dim_feedforward,
    )
    model = SyllablePredictorV3(config).to(args.device)
    vocab = SyllableVocab()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} params")

    if args.checkpoint.exists():
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint (epoch {checkpoint.get('epoch', '?')})")
        if "val_syl_accuracy" in checkpoint:
            print(f"  Training val acc: syl={checkpoint['val_syl_accuracy']:.4f}, "
                  f"tone={checkpoint['val_tone_accuracy']:.4f}")
    else:
        print(f"WARNING: Checkpoint not found: {args.checkpoint}")
        return

    results = {}

    # Evaluate on AISHELL-3 validation
    if args.source in ["aishell3", "all"]:
        aishell_metrics = evaluate_on_aishell3_val(
            model, vocab, config, args.device,
            context_mode=args.context_mode,
            max_sentences=args.max_sentences,
            verbose=args.verbose,
        )
        aishell_metrics.print_summary("AISHELL-3 Validation Results")
        results["aishell3"] = aishell_metrics.as_dict()

    # Evaluate on TTS
    if args.source in ["tts", "all"]:
        tts_metrics = evaluate_on_tts(
            model, vocab, config, args.device,
            voice=args.voice,
            context_mode=args.context_mode,
            max_sentences=args.max_sentences,
            sentence_id=args.sentence_id,
            verbose=args.verbose,
        )
        tts_metrics.print_summary(f"TTS ({args.voice}) Results")
        results["tts"] = tts_metrics.as_dict()

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
