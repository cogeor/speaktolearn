#!/usr/bin/env python3
"""Training script for SyllableToneModel.

This script trains the syllable-tone model on TTS sentence data.
It supports checkpointing, best model tracking, and accuracy reporting.

Usage:
    python train.py --epochs 50 --checkpoint-every 100
    python train.py --checkpoint-dir models/checkpoints --lr 0.001
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SENTENCES_JSON = PROJECT_ROOT / "apps/mobile_flutter/assets/datasets/sentences.zh.json"
AUDIO_DIR = PROJECT_ROOT / "apps/mobile_flutter/assets/examples"
DEFAULT_CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 50
    batch_size: int = 32  # Increased from 8
    learning_rate: float = 0.001
    val_split: float = 0.2
    checkpoint_every: int = 100  # steps
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
    device: str = "cuda"  # or "cpu"


def load_training_data(
    sentences_json: Path,
    audio_dir: Path,
) -> tuple[list[dict], list[dict]]:
    """Load training data from app assets.

    Returns:
        Tuple of (train_samples, val_samples)
    """
    from mandarin_grader.data import SentenceDataset
    from mandarin_grader.data.audio import load_audio

    samples = []

    # Load both voices
    for voice in ["female", "male"]:
        try:
            dataset = SentenceDataset.from_app_assets(
                sentences_json=sentences_json,
                audio_dir=audio_dir,
                voice=voice,
            )
            logger.info(f"Loaded {len(dataset)} {voice} samples")

            for sample in dataset:
                # Extract tones from syllables
                tones = [s.tone_surface for s in sample.syllables]
                if not tones:
                    continue

                samples.append({
                    "id": sample.id,
                    "audio_path": sample.audio_path,
                    "text": sample.text,
                    "tones": tones,
                    "voice": voice,
                })
        except Exception as e:
            logger.warning(f"Failed to load {voice} data: {e}")

    # Shuffle and split
    np.random.seed(42)
    np.random.shuffle(samples)

    split_idx = int(len(samples) * (1 - 0.2))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    logger.info(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    return train_samples, val_samples


def compute_class_weights(samples: list[dict], n_classes: int = 5) -> np.ndarray:
    """Compute class weights based on inverse frequency.

    Args:
        samples: Training samples with 'tones' field
        n_classes: Number of tone classes (0-4)

    Returns:
        Array of class weights [n_classes]
    """
    # Count tones
    counts = np.zeros(n_classes, dtype=np.float32)
    for sample in samples:
        for tone in sample["tones"]:
            if 0 <= tone < n_classes:
                counts[tone] += 1

    # Avoid division by zero
    counts = np.maximum(counts, 1.0)

    # Inverse frequency weighting
    total = counts.sum()
    weights = total / (n_classes * counts)

    # Normalize so mean weight is 1.0
    weights = weights / weights.mean()

    logger.info(f"Class counts: {counts.astype(int).tolist()}")
    logger.info(f"Class weights: {[f'{w:.2f}' for w in weights]}")

    return weights


def extract_features(audio_path: Path, config) -> np.ndarray | None:
    """Extract mel spectrogram features from audio.

    Args:
        audio_path: Path to audio file
        config: ModelConfig with mel parameters

    Returns:
        Mel spectrogram [n_mels, time] or None if failed
    """
    from mandarin_grader.data.audio import load_audio
    from mandarin_grader.model.syllable_tone_model import extract_mel_spectrogram

    try:
        audio = load_audio(audio_path, target_sr=config.sample_rate)
        mel = extract_mel_spectrogram(audio, config)
        return mel
    except Exception as e:
        logger.warning(f"Failed to extract features from {audio_path}: {e}")
        return None


class ToneDataset:
    """PyTorch-compatible dataset for tone training."""

    def __init__(self, samples: list[dict], model_config):
        self.samples = samples
        self.model_config = model_config
        self._cache = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        # Check cache
        cache_key = str(sample["audio_path"])
        if cache_key not in self._cache:
            mel = extract_features(sample["audio_path"], self.model_config)
            self._cache[cache_key] = mel

        mel = self._cache[cache_key]
        tones = sample["tones"]

        return {
            "mel": mel,
            "tones": tones,
            "id": sample["id"],
        }


def collate_fn(batch: list[dict]) -> dict:
    """Collate batch with padding.

    Returns dict with:
        - mel: [batch, n_mels, max_time]
        - tones: [batch, max_syllables]
        - lengths: [batch] - actual time lengths
        - tone_lengths: [batch] - actual syllable counts
    """
    import torch

    # Filter out failed samples
    batch = [b for b in batch if b["mel"] is not None]
    if not batch:
        return None

    # Pad mels to max length
    max_time = max(b["mel"].shape[1] for b in batch)
    n_mels = batch[0]["mel"].shape[0]

    mels = []
    lengths = []
    for b in batch:
        mel = b["mel"]
        pad_len = max_time - mel.shape[1]
        if pad_len > 0:
            mel = np.pad(mel, ((0, 0), (0, pad_len)), mode="constant")
        mels.append(mel)
        lengths.append(b["mel"].shape[1])

    # Pad tones to max syllables
    max_syls = max(len(b["tones"]) for b in batch)
    tones = []
    tone_lengths = []
    for b in batch:
        t = b["tones"]
        pad_len = max_syls - len(t)
        if pad_len > 0:
            t = t + [0] * pad_len  # Pad with tone 0
        tones.append(t)
        tone_lengths.append(len(b["tones"]))

    return {
        "mel": torch.tensor(np.stack(mels), dtype=torch.float32),
        "tones": torch.tensor(tones, dtype=torch.long),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "tone_lengths": torch.tensor(tone_lengths, dtype=torch.long),
        "ids": [b["id"] for b in batch],
    }


def count_parameters(model) -> tuple[int, int]:
    """Count model parameters.

    Returns:
        Tuple of (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device: str,
    step_counter: int,
    config: TrainingConfig,
    best_val_acc: float,
    val_loader,
) -> tuple[float, int, float]:
    """Train for one epoch.

    Returns:
        Tuple of (avg_loss, step_counter, best_val_acc)
    """
    import torch

    model.train()
    total_loss = 0
    num_batches = 0

    for batch in dataloader:
        if batch is None:
            continue

        mel = batch["mel"].to(device)
        tones = batch["tones"].to(device)
        tone_lengths = batch["tone_lengths"]

        # Forward pass
        boundary_logits, tone_logits = model(mel)

        # Compute loss on frame-level tone predictions
        # For simplicity, use average over all frames per syllable
        # In production, would use CTC or attention-based alignment

        # Simple approach: use first K frames for K syllables
        batch_size, time, n_tones = tone_logits.shape
        max_syls = tones.shape[1]

        # Uniformly distribute frames across syllables
        loss = torch.tensor(0.0, device=device)
        count = 0

        for b in range(batch_size):
            n_syls = tone_lengths[b].item()
            n_frames = batch["lengths"][b].item()

            if n_syls == 0 or n_frames == 0:
                continue

            frames_per_syl = n_frames // n_syls

            for s in range(n_syls):
                start_frame = s * frames_per_syl
                end_frame = min((s + 1) * frames_per_syl, n_frames)

                if end_frame <= start_frame:
                    continue

                # Average logits over syllable frames
                syl_logits = tone_logits[b, start_frame:end_frame].mean(dim=0)
                target = tones[b, s]

                loss += criterion(syl_logits.unsqueeze(0), target.unsqueeze(0))
                count += 1

        if count > 0:
            loss = loss / count

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        step_counter += 1

        # Checkpoint
        if step_counter % config.checkpoint_every == 0:
            checkpoint_path = config.checkpoint_dir / f"checkpoint_{step_counter}.pt"
            torch.save({
                "step": step_counter,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
            }, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

            # Evaluate and save best
            val_acc = evaluate(model, val_loader, device)
            logger.info(f"Step {step_counter} - Val Accuracy: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_path = config.checkpoint_dir / "best_model.pt"
                torch.save({
                    "step": step_counter,
                    "model_state_dict": model.state_dict(),
                    "val_accuracy": val_acc,
                }, best_path)
                logger.info(f"New best model! Accuracy: {val_acc:.4f}")

            model.train()

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, step_counter, best_val_acc


def evaluate(model, dataloader, device: str) -> float:
    """Evaluate model on validation set.

    Returns:
        Overall tone accuracy
    """
    import torch

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue

            mel = batch["mel"].to(device)
            tones = batch["tones"]
            tone_lengths = batch["tone_lengths"]

            _, tone_logits = model(mel)

            batch_size = mel.shape[0]

            for b in range(batch_size):
                n_syls = tone_lengths[b].item()
                n_frames = batch["lengths"][b].item()

                if n_syls == 0 or n_frames == 0:
                    continue

                frames_per_syl = n_frames // n_syls

                for s in range(n_syls):
                    start_frame = s * frames_per_syl
                    end_frame = min((s + 1) * frames_per_syl, n_frames)

                    if end_frame <= start_frame:
                        continue

                    syl_logits = tone_logits[b, start_frame:end_frame].mean(dim=0)
                    pred = syl_logits.argmax().item()
                    target = tones[b, s].item()

                    if pred == target:
                        correct += 1
                    total += 1

    return correct / max(total, 1)


def evaluate_detailed(model, dataloader, device: str) -> dict:
    """Detailed evaluation with per-tone breakdown.

    Returns:
        Dict with overall accuracy and per-tone metrics
    """
    import torch

    model.eval()
    per_tone_correct = defaultdict(int)
    per_tone_total = defaultdict(int)
    confusion = defaultdict(lambda: defaultdict(int))

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue

            mel = batch["mel"].to(device)
            tones = batch["tones"]
            tone_lengths = batch["tone_lengths"]

            _, tone_logits = model(mel)

            batch_size = mel.shape[0]

            for b in range(batch_size):
                n_syls = tone_lengths[b].item()
                n_frames = batch["lengths"][b].item()

                if n_syls == 0 or n_frames == 0:
                    continue

                frames_per_syl = n_frames // n_syls

                for s in range(n_syls):
                    start_frame = s * frames_per_syl
                    end_frame = min((s + 1) * frames_per_syl, n_frames)

                    if end_frame <= start_frame:
                        continue

                    syl_logits = tone_logits[b, start_frame:end_frame].mean(dim=0)
                    pred = syl_logits.argmax().item()
                    target = tones[b, s].item()

                    per_tone_total[target] += 1
                    confusion[target][pred] += 1

                    if pred == target:
                        per_tone_correct[target] += 1

    # Calculate metrics
    results = {
        "per_tone_accuracy": {},
        "per_tone_counts": {},
        "confusion_matrix": {},
    }

    total_correct = 0
    total_samples = 0

    for tone in range(5):
        total = per_tone_total[tone]
        correct = per_tone_correct[tone]
        total_samples += total
        total_correct += correct

        if total > 0:
            results["per_tone_accuracy"][tone] = correct / total
            results["per_tone_counts"][tone] = total
            results["confusion_matrix"][tone] = dict(confusion[tone])
        else:
            results["per_tone_accuracy"][tone] = 0.0
            results["per_tone_counts"][tone] = 0
            results["confusion_matrix"][tone] = {}

    results["overall_accuracy"] = total_correct / max(total_samples, 1)
    results["total_samples"] = total_samples

    return results


def main():
    import torch
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="Train SyllableToneModel")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Checkpoint every N steps")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sentences", type=Path, default=SENTENCES_JSON)
    parser.add_argument("--audio-dir", type=Path, default=AUDIO_DIR)
    parser.add_argument("--no-class-weights", action="store_true", help="Disable class weighting")

    args = parser.parse_args()

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )

    # Create checkpoint directory
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading training data...")
    train_samples, val_samples = load_training_data(args.sentences, args.audio_dir)

    if not train_samples:
        logger.error("No training samples found!")
        return

    # Create model
    from mandarin_grader.model.syllable_tone_model import SyllableToneModel, ModelConfig
    model_config = ModelConfig()
    model = SyllableToneModel(model_config)
    model = model.to(config.device)

    # Report model architecture and size
    total_params, trainable_params = count_parameters(model)
    logger.info(f"Model created, device: {config.device}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

    # Create datasets and loaders
    train_dataset = ToneDataset(train_samples, model_config)
    val_dataset = ToneDataset(val_samples, model_config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Compute class weights
    if not args.no_class_weights:
        class_weights = compute_class_weights(train_samples, n_classes=5)
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=config.device)
        criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
        logger.info("Using class-weighted loss")
    else:
        criterion = torch.nn.CrossEntropyLoss()
        logger.info("Using unweighted loss")

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    step_counter = 0
    best_val_acc = 0.0
    training_history = []

    logger.info(f"Starting training for {config.epochs} epochs...")

    for epoch in range(config.epochs):
        avg_loss, step_counter, best_val_acc = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=config.device,
            step_counter=step_counter,
            config=config,
            best_val_acc=best_val_acc,
            val_loader=val_loader,
        )

        # Epoch evaluation - both train and val
        train_acc = evaluate(model, train_loader, config.device)
        val_acc = evaluate(model, val_loader, config.device)
        logger.info(f"Epoch {epoch+1}/{config.epochs} - Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        training_history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
        })

        # Update best model at epoch end
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = config.checkpoint_dir / "best_model.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_acc,
            }, best_path)
            logger.info(f"New best model at epoch {epoch+1}! Accuracy: {val_acc:.4f}")

    # Final evaluation on both train and val
    logger.info("\n" + "="*50)
    logger.info("FINAL EVALUATION")
    logger.info("="*50)

    train_detailed = evaluate_detailed(model, train_loader, config.device)
    val_detailed = evaluate_detailed(model, val_loader, config.device)

    logger.info(f"\n{'='*50}")
    logger.info("TRAINING SET RESULTS")
    logger.info(f"{'='*50}")
    logger.info(f"Overall Accuracy: {train_detailed['overall_accuracy']:.4f}")
    logger.info(f"Total Syllables: {train_detailed['total_samples']}")
    logger.info("\nPer-Tone Accuracy (Train):")
    for tone in range(5):
        acc = train_detailed["per_tone_accuracy"][tone]
        count = train_detailed["per_tone_counts"][tone]
        logger.info(f"  Tone {tone}: {acc:.4f} ({count} samples)")

    logger.info(f"\n{'='*50}")
    logger.info("VALIDATION SET RESULTS")
    logger.info(f"{'='*50}")
    logger.info(f"Overall Accuracy: {val_detailed['overall_accuracy']:.4f}")
    logger.info(f"Total Syllables: {val_detailed['total_samples']}")
    logger.info("\nPer-Tone Accuracy (Val):")
    for tone in range(5):
        acc = val_detailed["per_tone_accuracy"][tone]
        count = val_detailed["per_tone_counts"][tone]
        logger.info(f"  Tone {tone}: {acc:.4f} ({count} samples)")

    # Overfitting analysis
    train_acc = train_detailed['overall_accuracy']
    val_acc = val_detailed['overall_accuracy']
    gap = train_acc - val_acc

    logger.info(f"\n{'='*50}")
    logger.info("ANALYSIS")
    logger.info(f"{'='*50}")
    logger.info(f"Train Accuracy: {train_acc:.4f}")
    logger.info(f"Val Accuracy:   {val_acc:.4f}")
    logger.info(f"Gap (train-val): {gap:.4f}")

    if train_acc < 0.4:
        logger.warning("LOW TRAIN ACCURACY - Model is underfitting or pipeline issue")
    if gap > 0.15:
        logger.warning("LARGE GAP - Model is overfitting")
    elif gap < 0.05 and train_acc < 0.5:
        logger.warning("SMALL GAP + LOW ACC - Model may be predicting noise")

    # Use val_detailed as the main results for backward compatibility
    detailed_results = val_detailed

    # Save final model and report
    final_path = config.checkpoint_dir / "final_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "training_history": training_history,
        "final_results": detailed_results,
    }, final_path)
    logger.info(f"\nFinal model saved to {final_path}")

    # Save report as JSON
    report_path = config.checkpoint_dir / "training_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "config": {
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
            },
            "model": {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "size_mb": total_params * 4 / 1024 / 1024,
                "architecture": {
                    "cnn_channels": model_config.cnn_channels,
                    "lstm_hidden": model_config.lstm_hidden,
                    "lstm_layers": model_config.lstm_layers,
                    "n_mels": model_config.n_mels,
                },
            },
            "history": training_history,
            "train_results": train_detailed,
            "val_results": val_detailed,
            "analysis": {
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "gap": gap,
            },
            "best_val_accuracy": best_val_acc,
        }, f, indent=2)
    logger.info(f"Training report saved to {report_path}")


if __name__ == "__main__":
    main()
