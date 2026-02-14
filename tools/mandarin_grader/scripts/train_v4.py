#!/usr/bin/env python3
"""Training script for SyllablePredictorV4.

Trains the autoregressive syllable+tone model V4 with:
- CNN front-end (4x downsampling)
- Attention pooling (PMA)
- Rotary Position Embeddings (RoPE)

Usage:
    # Run overfit test first
    python train_v4.py --overfit-test

    # Full training with synthetic data (default)
    python train_v4.py --epochs 30 --checkpoint-dir checkpoints_v4_run1

    # Train with AISHELL-3 data
    python train_v4.py --data-source aishell3 --data-dir datasets/aishell3

    # Train with mixed data sources
    python train_v4.py --data-source synthetic,aishell3 \\
        --data-dir data/synthetic_train,datasets/aishell3

    # Resume from checkpoint
    python train_v4.py --epochs 30 --checkpoint-dir checkpoints_v4_run1 --resume best_model.pt

    # List available data sources
    python train_v4.py --list-sources
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Default paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SENTENCES_JSON = PROJECT_ROOT / "apps/mobile_flutter/assets/datasets/sentences.zh.json"
SYLLABLES_DIR = Path(__file__).parent.parent / "data" / "syllables_v2"
SYNTHETIC_DIR = Path(__file__).parent.parent / "data" / "synthetic_train"
DEFAULT_CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints_v4"


def get_warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Create a scheduler with linear warmup then cosine decay.

    Args:
        optimizer: The optimizer
        warmup_steps: Number of warmup steps (linear from 0 to base LR)
        total_steps: Total training steps

    Returns:
        LambdaLR scheduler
    """
    import torch
    import math

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def setup_logging(checkpoint_dir: Path) -> logging.Logger:
    """Setup logging to both console and file."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_file = checkpoint_dir / "train.log"

    logger = logging.getLogger("train_v4")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers

    # File handler
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(ch)

    return logger


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    log_every_epochs: int = 5
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
    device: str = "cuda"

    # Overfit test
    overfit_test: bool = False
    overfit_samples: int = 8
    overfit_steps: int = 200


def load_synthetic_data(synthetic_dir: Path, logger) -> tuple[list, list, dict]:
    """Load synthetic training data (legacy function for backwards compatibility)."""
    return load_training_data(["synthetic"], [synthetic_dir], logger)


def load_training_data(
    sources: list[str],
    data_dirs: list[Path],
    logger,
    train_split: float = 0.8,
    max_sentences_per_source: int | None = None,
) -> tuple[list, list, dict[str, np.ndarray]]:
    """Load training data from multiple sources.

    Args:
        sources: List of data source names (e.g., ["synthetic", "aishell3"])
        data_dirs: List of directories corresponding to each source
        logger: Logger instance
        train_split: Fraction for training (rest is validation)
        max_sentences_per_source: Optional limit per source

    Returns:
        (train_sentences, val_sentences) lists
    """
    from mandarin_grader.data.data_source import DataSourceRegistry, SentenceInfo
    from mandarin_grader.data.autoregressive_dataset import SyntheticSentenceInfo

    all_sentences = []

    # Mel cache for tar-based sources (pre-computed mel features in memory)
    mel_cache = {}

    for source_name, data_dir in zip(sources, data_dirs):
        logger.info(f"Loading data source: {source_name} from {data_dir}")

        try:
            source = DataSourceRegistry.get(source_name)
            if not source.is_available(data_dir):
                logger.warning(f"  Source not available at {data_dir}")
                continue

            kwargs = {}
            if max_sentences_per_source:
                kwargs["max_sentences"] = max_sentences_per_source

            sentences = source.load(data_dir, **kwargs)
            logger.info(f"  Loaded {len(sentences)} sentences from {source_name}")

            # If source has mel cache (tar-based), merge it
            if hasattr(source, "get_mel_cache"):
                source_cache = source.get_mel_cache()
                mel_cache.update(source_cache)
                logger.info(f"  Mel cache: {len(source_cache)} files pre-loaded")

            # Convert SentenceInfo to SyntheticSentenceInfo for compatibility
            for s in sentences:
                all_sentences.append(SyntheticSentenceInfo(
                    id=s.id,
                    audio_path=s.audio_path,
                    text=s.text,
                    syllables=s.syllables,
                    syllable_boundaries=s.syllable_boundaries,
                    sample_rate=s.sample_rate,
                    total_samples=s.total_samples,
                ))

        except Exception as e:
            logger.error(f"  Error loading {source_name}: {e}")
            continue

    logger.info(f"Total sentences: {len(all_sentences)}")

    if not all_sentences:
        return [], [], {}

    # Shuffle and split
    np.random.seed(42)
    indices = np.random.permutation(len(all_sentences))
    split_idx = int(len(all_sentences) * train_split)

    train = [all_sentences[i] for i in indices[:split_idx]]
    val = [all_sentences[i] for i in indices[split_idx:]]
    logger.info(f"Train: {len(train)}, Val: {len(val)}")
    return train, val, mel_cache


def list_available_sources():
    """Print available data sources and exit."""
    from mandarin_grader.data.data_source import DataSourceRegistry

    print("Available data sources:")
    print()
    for name in DataSourceRegistry.list_sources():
        source = DataSourceRegistry.get(name)
        print(f"  {name}")
        print(f"    {source.description}")
        print()


def create_dataloader(sentences: list, batch_size: int, shuffle: bool, augment: bool, context_mode: str = "pinyin", preload: bool = False, logger=None, mel_cache: dict | None = None):
    """Create PyTorch DataLoader.

    Args:
        sentences: List of sentence info
        batch_size: Batch size
        shuffle: Whether to shuffle
        augment: Whether to augment audio
        context_mode: "pinyin" = actual syllables, "position" = only syllable index
        preload: Whether to preload all audio into memory (recommended for large datasets)
        logger: Logger for progress reporting
        mel_cache: Pre-loaded mel cache from tar sources (optional)
    """
    import torch
    from torch.utils.data import DataLoader
    from mandarin_grader.data.autoregressive_dataset import AutoregressiveDataset, spec_augment
    from mandarin_grader.model.syllable_predictor_v4 import (
        SyllablePredictorConfigV4, SyllableVocab, extract_mel_spectrogram,
    )

    config = SyllablePredictorConfigV4()
    vocab = SyllableVocab()

    dataset = AutoregressiveDataset(
        sentences=sentences,
        sample_rate=config.sample_rate,
        chunk_duration_s=1.0,
        margin_s=0.1,
        augment=augment,
    )

    # If mel cache provided (from tar sources), pre-populate dataset cache
    if mel_cache:
        dataset._mel_cache.update(mel_cache)
        if logger:
            logger.info(f"  Injected {len(mel_cache)} precomputed mel entries")

    if preload:
        def progress(loaded, total):
            if logger:
                logger.info(f"  Preloading audio: {loaded}/{total} ({100*loaded/total:.0f}%)")
        if logger:
            logger.info("Preloading audio files into memory...")
        dataset.preload_audio(progress_callback=progress)

    def collate_fn(batch):
        mels, pinyin_ids_list, target_syls, target_tones = [], [], [], []
        max_pinyin_len = 0

        for sample in batch:
            if sample.mel_chunk is not None:
                mel = sample.mel_chunk
            elif sample.audio_chunk is not None:
                mel = extract_mel_spectrogram(sample.audio_chunk, config)
                if augment:  # Apply SpecAugment to freshly computed mel
                    mel = spec_augment(mel)
            else:
                raise ValueError(f"Sample {sample.sample_id} has neither mel_chunk nor audio_chunk")
            mels.append(mel)

            if context_mode == "pinyin":
                # Original: encode actual syllables
                ids = vocab.encode_sequence(sample.pinyin_context, add_bos=True)
            else:
                # Position-only: encode [BOS, position_index]
                # Position token = 2 + syllable_idx (0=PAD, 1=BOS, 2+=positions)
                position_token = 2 + sample.syllable_idx
                ids = [vocab.bos_token, position_token]

            pinyin_ids_list.append(ids)
            max_pinyin_len = max(max_pinyin_len, len(ids))

            target_syls.append(vocab.encode(sample.target_syllable))
            target_tones.append(sample.target_tone)

        # Pad mels
        max_time = max(m.shape[1] for m in mels)
        n_mels = mels[0].shape[0]
        padded_mels = np.zeros((len(mels), n_mels, max_time), dtype=np.float32)
        audio_masks = np.zeros((len(mels), max_time), dtype=bool)
        for i, mel in enumerate(mels):
            padded_mels[i, :, :mel.shape[1]] = mel
            audio_masks[i, mel.shape[1]:] = True

        # Pad pinyin
        padded_pinyin = np.zeros((len(batch), max_pinyin_len), dtype=np.int64)
        pinyin_masks = np.zeros((len(batch), max_pinyin_len), dtype=bool)
        for i, ids in enumerate(pinyin_ids_list):
            padded_pinyin[i, :len(ids)] = ids
            pinyin_masks[i, len(ids):] = True

        return {
            "mel": torch.tensor(padded_mels, dtype=torch.float32),
            "pinyin_ids": torch.tensor(padded_pinyin, dtype=torch.long),
            "audio_mask": torch.tensor(audio_masks, dtype=torch.bool),
            "pinyin_mask": torch.tensor(pinyin_masks, dtype=torch.bool),
            "target_syllable": torch.tensor(target_syls, dtype=torch.long),
            "target_tone": torch.tensor(target_tones, dtype=torch.long),
        }

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_fn, num_workers=0)


def evaluate(model, dataloader, device: str) -> tuple[float, float]:
    """Evaluate model. Returns (syllable_acc, tone_acc)."""
    import torch
    model.eval()
    syl_correct, tone_correct, total = 0, 0, 0

    with torch.no_grad():
        for batch in dataloader:
            mel = batch["mel"].to(device)
            pinyin_ids = batch["pinyin_ids"].to(device)
            audio_mask = batch["audio_mask"].to(device)
            pinyin_mask = batch["pinyin_mask"].to(device)

            syl_logits, tone_logits = model(mel, pinyin_ids, audio_mask, pinyin_mask)

            syl_correct += (syl_logits.argmax(-1).cpu() == batch["target_syllable"]).sum().item()
            tone_correct += (tone_logits.argmax(-1).cpu() == batch["target_tone"]).sum().item()
            total += mel.shape[0]

    return syl_correct / max(total, 1), tone_correct / max(total, 1)


def evaluate_detailed(model, dataloader, device: str) -> dict:
    """Detailed evaluation with per-tone breakdown."""
    import torch
    model.eval()

    per_tone_correct = defaultdict(int)
    per_tone_total = defaultdict(int)
    syl_correct, total = 0, 0

    with torch.no_grad():
        for batch in dataloader:
            mel = batch["mel"].to(device)
            pinyin_ids = batch["pinyin_ids"].to(device)
            audio_mask = batch["audio_mask"].to(device)
            pinyin_mask = batch["pinyin_mask"].to(device)

            syl_logits, tone_logits = model(mel, pinyin_ids, audio_mask, pinyin_mask)
            syl_pred = syl_logits.argmax(-1).cpu()
            tone_pred = tone_logits.argmax(-1).cpu()

            for i in range(mel.shape[0]):
                tone = batch["target_tone"][i].item()
                per_tone_total[tone] += 1
                if tone_pred[i].item() == tone:
                    per_tone_correct[tone] += 1
                if syl_pred[i].item() == batch["target_syllable"][i].item():
                    syl_correct += 1
                total += 1

    tone_acc = sum(per_tone_correct.values()) / max(total, 1)
    return {
        "syllable_accuracy": syl_correct / max(total, 1),
        "tone_accuracy": tone_acc,
        "per_tone_accuracy": {t: per_tone_correct[t] / max(per_tone_total[t], 1) for t in range(5)},
        "per_tone_counts": dict(per_tone_total),
        "total_samples": total,
    }


def run_overfit_test(model, train_loader, config: TrainingConfig, device: str, logger) -> bool:
    """Run overfit test on small batch."""
    import torch

    logger.info("=" * 60)
    logger.info("OVERFIT TEST")
    logger.info(f"Samples: {config.overfit_samples}, Steps: {config.overfit_steps}")
    logger.info("=" * 60)

    batch = next(iter(train_loader))
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key][:config.overfit_samples]

    mel = batch["mel"].to(device)
    pinyin_ids = batch["pinyin_ids"].to(device)
    audio_mask = batch["audio_mask"].to(device)
    pinyin_mask = batch["pinyin_mask"].to(device)
    target_syl = batch["target_syllable"].to(device)
    target_tone = batch["target_tone"].to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate * 10)
    syl_criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    tone_criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    model.train()
    for step in range(config.overfit_steps):
        optimizer.zero_grad()
        syl_logits, tone_logits = model(mel, pinyin_ids, audio_mask, pinyin_mask)
        syl_loss = syl_criterion(syl_logits, target_syl)
        tone_loss = tone_criterion(tone_logits, target_tone)
        loss = 0.7 * syl_loss + 0.3 * tone_loss
        loss.backward()
        optimizer.step()

        if (step + 1) % 50 == 0:
            syl_acc = (syl_logits.argmax(-1) == target_syl).float().mean().item()
            tone_acc = (tone_logits.argmax(-1) == target_tone).float().mean().item()
            logger.info(f"Step {step+1}: Loss={loss.item():.4f}, Syl={syl_acc:.4f}, Tone={tone_acc:.4f}")

    model.eval()
    with torch.no_grad():
        syl_logits, tone_logits = model(mel, pinyin_ids, audio_mask, pinyin_mask)
        syl_acc = (syl_logits.argmax(-1) == target_syl).float().mean().item()
        tone_acc = (tone_logits.argmax(-1) == target_tone).float().mean().item()

    passed = syl_acc > 0.9 and tone_acc > 0.9
    logger.info(f"Final: Syl={syl_acc:.4f}, Tone={tone_acc:.4f} - {'PASSED' if passed else 'FAILED'}")
    return passed


def train(model, train_loader, val_loader, config: TrainingConfig, logger, start_epoch: int = 0):
    """Main training loop."""
    import torch

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = config.epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)  # 5% warmup
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)
    syl_criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    tone_criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0

    logger.info(f"Starting training from epoch {start_epoch + 1} to {config.epochs}")
    logger.info(f"Warmup steps: {warmup_steps} ({warmup_steps / len(train_loader):.1f} epochs)")
    logger.info(f"Logging every {config.log_every_epochs} epochs")

    import time as _time
    for epoch in range(start_epoch, config.epochs):
        model.train()
        total_loss, num_batches = 0, 0
        epoch_start = _time.time()

        for batch in train_loader:
            # Progress logging every 500 batches
            if num_batches > 0 and num_batches % 500 == 0:
                elapsed = _time.time() - epoch_start
                ms_per_batch = elapsed / num_batches * 1000
                eta_min = (len(train_loader) - num_batches) * ms_per_batch / 60000
                logger.info(f"  Epoch {epoch+1} batch {num_batches}/{len(train_loader)} | {ms_per_batch:.0f}ms/batch | ETA: {eta_min:.1f}min")
            mel = batch["mel"].to(config.device)
            pinyin_ids = batch["pinyin_ids"].to(config.device)
            audio_mask = batch["audio_mask"].to(config.device)
            pinyin_mask = batch["pinyin_mask"].to(config.device)
            target_syl = batch["target_syllable"].to(config.device)
            target_tone = batch["target_tone"].to(config.device)

            optimizer.zero_grad()
            syl_logits, tone_logits = model(mel, pinyin_ids, audio_mask, pinyin_mask)
            syl_loss = syl_criterion(syl_logits, target_syl)
            tone_loss = tone_criterion(tone_logits, target_tone)
            loss = 0.7 * syl_loss + 0.3 * tone_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        # Log and checkpoint every N epochs
        if (epoch + 1) % config.log_every_epochs == 0 or epoch == config.epochs - 1:
            train_syl, train_tone = evaluate(model, train_loader, config.device)
            val_syl, val_tone = evaluate(model, val_loader, config.device)

            logger.info(
                f"Epoch {epoch+1:3d}/{config.epochs} | Loss: {avg_loss:.4f} | "
                f"Train: {train_syl:.4f}/{train_tone:.4f} | Val: {val_syl:.4f}/{val_tone:.4f}"
            )

            # Save checkpoint
            ckpt_path = config.checkpoint_dir / f"checkpoint_epoch{epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_syl_accuracy": val_syl,
                "val_tone_accuracy": val_tone,
            }, ckpt_path)

            # Save best model
            val_combined = (val_syl + val_tone) / 2
            if val_combined > best_val_acc:
                best_val_acc = val_combined
                best_path = config.checkpoint_dir / "best_model.pt"
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "val_syl_accuracy": val_syl,
                    "val_tone_accuracy": val_tone,
                }, best_path)
                logger.info(f"  -> New best model! Combined: {val_combined:.4f}")

    return best_val_acc


def main():
    import torch

    parser = argparse.ArgumentParser(
        description="Train SyllablePredictorV4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--log-every-epochs", type=int, default=5)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Data source arguments
    parser.add_argument(
        "--data-source", type=str, default="synthetic",
        help="Data source(s) to use, comma-separated (e.g., 'synthetic,aishell3')"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Data directory(ies), comma-separated. If not provided, uses defaults."
    )
    parser.add_argument(
        "--max-sentences", type=int, default=None,
        help="Maximum sentences per data source (useful for quick tests)"
    )

    # Legacy argument for backwards compatibility
    parser.add_argument("--synthetic-dir", type=Path, default=SYNTHETIC_DIR,
                        help="(Deprecated) Use --data-source and --data-dir instead")

    parser.add_argument("--resume", type=str, default=None, help="Checkpoint filename to resume from")
    parser.add_argument("--overfit-test", action="store_true")
    parser.add_argument("--list-sources", action="store_true", help="List available data sources and exit")

    # Model architecture arguments
    parser.add_argument("--d-model", type=int, default=192, help="Model dimension (default: 192)")
    parser.add_argument("--n-layers", type=int, default=4, help="Number of transformer layers (default: 4)")
    parser.add_argument("--n-heads", type=int, default=6, help="Number of attention heads (default: 6)")
    parser.add_argument("--dim-feedforward", type=int, default=384, help="FFN hidden dimension (default: 384)")

    # Context mode
    parser.add_argument(
        "--context-mode", type=str, default="pinyin", choices=["pinyin", "position"],
        help="Context mode: 'pinyin' = actual syllables (default), 'position' = only syllable count/index"
    )

    args = parser.parse_args()

    # Handle --list-sources
    if args.list_sources:
        list_available_sources()
        return

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        log_every_epochs=args.log_every_epochs,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        overfit_test=args.overfit_test,
    )

    logger = setup_logging(config.checkpoint_dir)
    logger.info("=" * 60)
    logger.info("SyllablePredictorV4 Training")
    logger.info("=" * 60)
    logger.info(f"Command: python {' '.join(sys.argv)}")

    # Parse data sources
    sources = [s.strip() for s in args.data_source.split(",")]

    # Parse data directories
    if args.data_dir:
        data_dirs = [Path(d.strip()) for d in args.data_dir.split(",")]
    else:
        # Use defaults based on source names
        data_dirs = []
        for source in sources:
            if source == "synthetic":
                data_dirs.append(args.synthetic_dir)
            elif source == "aishell3":
                data_dirs.append(Path(__file__).parent.parent / "datasets" / "aishell3" / "data_aishell3")
            else:
                data_dirs.append(Path(__file__).parent.parent / "datasets" / source)

    # Ensure same number of sources and directories
    if len(sources) != len(data_dirs):
        logger.error(f"Mismatch: {len(sources)} sources but {len(data_dirs)} directories")
        return

    logger.info(f"Data sources: {sources}")
    for src, dir in zip(sources, data_dirs):
        logger.info(f"  {src}: {dir}")

    # Load data
    train_sentences, val_sentences, mel_cache = load_training_data(
        sources, data_dirs, logger,
        max_sentences_per_source=args.max_sentences,
    )
    if not train_sentences:
        logger.error("No training data!")
        return

    # Create model
    from mandarin_grader.model.syllable_predictor_v4 import SyllablePredictorV4, SyllablePredictorConfigV4
    model_config = SyllablePredictorConfigV4(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dim_feedforward=args.dim_feedforward,
    )
    model = SyllablePredictorV4(model_config).to(config.device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {total_params:,} params ({total_params * 4 / 1024 / 1024:.2f} MB)")
    logger.info(f"Device: {config.device}")

    # Create dataloaders (preload audio for non-mel sources).
    # If mel_cache is provided, no audio preload is needed for that source.
    logger.info(f"Context mode: {args.context_mode}")
    preload = not mel_cache
    train_loader = create_dataloader(train_sentences, config.batch_size, shuffle=True, augment=True, context_mode=args.context_mode, preload=preload, logger=logger, mel_cache=mel_cache)
    val_loader = create_dataloader(val_sentences, config.batch_size, shuffle=False, augment=False, context_mode=args.context_mode, preload=preload, logger=logger, mel_cache=mel_cache)
    logger.info(f"Batches: Train={len(train_loader)}, Val={len(val_loader)}")

    # Overfit test
    if config.overfit_test:
        run_overfit_test(model, train_loader, config, config.device, logger)
        return

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        ckpt_path = config.checkpoint_dir / args.resume
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=config.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = checkpoint.get("epoch", 0)
            logger.info(f"Resumed from {ckpt_path} (epoch {start_epoch})")

    # Train
    best_acc = train(model, train_loader, val_loader, config, logger, start_epoch)

    # Final evaluation
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 60)

    train_results = evaluate_detailed(model, train_loader, config.device)
    val_results = evaluate_detailed(model, val_loader, config.device)

    logger.info(f"Train - Syl: {train_results['syllable_accuracy']:.4f}, Tone: {train_results['tone_accuracy']:.4f}")
    logger.info(f"Val   - Syl: {val_results['syllable_accuracy']:.4f}, Tone: {val_results['tone_accuracy']:.4f}")

    for t in range(5):
        logger.info(f"  Tone {t}: {val_results['per_tone_accuracy'][t]:.4f} ({val_results['per_tone_counts'].get(t, 0)} samples)")

    # Save final report
    report = {
        "config": {"epochs": config.epochs, "batch_size": config.batch_size, "lr": config.learning_rate},
        "model_params": total_params,
        "train_results": train_results,
        "val_results": val_results,
        "best_val_combined": best_acc,
    }
    with open(config.checkpoint_dir / "training_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Training complete. Best combined accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
