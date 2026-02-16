#!/usr/bin/env python3
"""Training script for SyllablePredictorV6 - FlexAttention Transformer.

Usage:
    python train_v6.py --data-source aishell3 --data-dir datasets/aishell3_tar --epochs 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

SYNTHETIC_DIR = Path(__file__).parent.parent / "data" / "synthetic_train"
DEFAULT_CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints_v6"


def get_warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int):
    import torch
    import math

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def setup_logging(checkpoint_dir: Path) -> logging.Logger:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_file = checkpoint_dir / "train.log"

    logger = logging.getLogger("train_v6")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(ch)

    return logger


@dataclass
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    log_every_epochs: int = 1
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
    device: str = "cuda"
    overfit_test: bool = False
    overfit_samples: int = 8
    overfit_steps: int = 200
    use_compile: bool = False  # torch.compile for FlexAttention


def load_training_data(
    sources: list[str],
    data_dirs: list[Path],
    logger,
    train_split: float = 0.8,
    max_sentences_per_source: int | None = None,
) -> tuple[list, list, dict[str, np.ndarray]]:
    from mandarin_grader.data.synthetic_source import SyntheticDataSource
    from mandarin_grader.data.aishell_tar_source import AISHELL3TarDataSource
    from mandarin_grader.data.tts_source import TTSDataSource
    from mandarin_grader.data.autoregressive_dataset import SyntheticSentenceInfo

    source_classes = {
        "synthetic": SyntheticDataSource(),
        "aishell3": AISHELL3TarDataSource(),
        "tts": TTSDataSource(),
    }

    all_sentences = []
    mel_cache = {}

    for source_name, data_dir in zip(sources, data_dirs):
        logger.info(f"Loading data source: {source_name} from {data_dir}")

        try:
            if source_name not in source_classes:
                logger.error(f"  Unknown data source: {source_name}")
                continue

            source = source_classes[source_name]
            if not source.is_available(data_dir):
                logger.warning(f"  Source not available at {data_dir}")
                continue

            kwargs = {}
            if max_sentences_per_source:
                kwargs["max_sentences"] = max_sentences_per_source

            sentences = source.load(data_dir, **kwargs)
            logger.info(f"  Loaded {len(sentences)} sentences from {source_name}")

            if hasattr(source, "get_mel_cache"):
                source_cache = source.get_mel_cache()
                mel_cache.update(source_cache)
                logger.info(f"  Mel cache: {len(source_cache)} files pre-loaded")

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

    np.random.seed(42)
    indices = np.random.permutation(len(all_sentences))
    split_idx = int(len(all_sentences) * train_split)

    train = [all_sentences[i] for i in indices[:split_idx]]
    val = [all_sentences[i] for i in indices[split_idx:]]
    logger.info(f"Train: {len(train)}, Val: {len(val)}")
    return train, val, mel_cache


def create_dataloader(
    sentences: list,
    batch_size: int,
    shuffle: bool,
    augment: bool,
    preload: bool = False,
    logger=None,
    mel_cache: dict | None = None,
    max_duration_s: float = 10.0,
    speed_variation: float = 0.1,
    pitch_shift_semitones: float = 0.0,
    formant_shift_percent: float = 0.0,
):
    import torch
    from torch.utils.data import DataLoader
    from mandarin_grader.data.full_sentence_dataset import FullSentenceDataset
    from mandarin_grader.data.autoregressive_dataset import spec_augment
    from mandarin_grader.model.syllable_predictor_v6 import SyllablePredictorConfigV6, SyllableVocab
    from mandarin_grader.model.syllable_predictor_v4 import extract_mel_spectrogram, SyllablePredictorConfigV4

    config = SyllablePredictorConfigV6()
    vocab = SyllableVocab()
    mel_config = SyllablePredictorConfigV4()

    dataset = FullSentenceDataset(
        sentences=sentences,
        sample_rate=config.sample_rate,
        max_duration_s=max_duration_s,
        augment=augment,
        speed_variation=speed_variation,
        pitch_shift_semitones=pitch_shift_semitones,
        formant_shift_percent=formant_shift_percent,
    )

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

    max_frames = int(max_duration_s * 100)

    def collate_fn(batch):
        mels, positions, target_syls, target_tones = [], [], [], []

        for sample in batch:
            if sample.mel_full is not None:
                mel = sample.mel_full
            elif sample.audio_full is not None:
                mel = extract_mel_spectrogram(sample.audio_full, mel_config)
                if augment:
                    mel = spec_augment(mel)
            else:
                raise ValueError(f"Sample {sample.sample_id} has neither mel nor audio")

            if mel.shape[1] > max_frames:
                mel = mel[:, :max_frames]

            mels.append(mel)
            positions.append(sample.position)
            target_syls.append(vocab.encode(sample.target_syllable))
            target_tones.append(sample.target_tone)

        n_mels = mels[0].shape[0]
        padded_mels = np.zeros((len(batch), n_mels, max_frames), dtype=np.float32)
        for i, mel in enumerate(mels):
            padded_mels[i, :, :mel.shape[1]] = mel

        return {
            "mel": torch.tensor(padded_mels, dtype=torch.float32),
            "position": torch.tensor(positions, dtype=torch.long),
            "target_syllable": torch.tensor(target_syls, dtype=torch.long),
            "target_tone": torch.tensor(target_tones, dtype=torch.long),
        }

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_fn, num_workers=0)


def evaluate(model, dataloader, device: str) -> tuple[float, float]:
    import torch
    model.eval()
    syl_correct, tone_correct, total = 0, 0, 0

    with torch.no_grad():
        for batch in dataloader:
            mel = batch["mel"].to(device)
            position = batch["position"].to(device)

            syl_logits, tone_logits = model(mel, position)

            syl_correct += (syl_logits.argmax(-1).cpu() == batch["target_syllable"]).sum().item()
            tone_correct += (tone_logits.argmax(-1).cpu() == batch["target_tone"]).sum().item()
            total += mel.shape[0]

    return syl_correct / max(total, 1), tone_correct / max(total, 1)


def train(model, train_loader, val_loader, config: TrainingConfig, logger, start_epoch: int = 0):
    import torch

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = config.epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)
    syl_criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    tone_criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0

    logger.info(f"Starting training from epoch {start_epoch + 1} to {config.epochs}")
    logger.info(f"Warmup steps: {warmup_steps}")

    for epoch in range(start_epoch, config.epochs):
        model.train()
        total_loss, num_batches = 0, 0
        epoch_start = time.time()

        for batch in train_loader:
            mel = batch["mel"].to(config.device)
            position = batch["position"].to(config.device)
            target_syl = batch["target_syllable"].to(config.device)
            target_tone = batch["target_tone"].to(config.device)

            optimizer.zero_grad()
            syl_logits, tone_logits = model(mel, position)
            syl_loss = syl_criterion(syl_logits, target_syl)
            tone_loss = tone_criterion(tone_logits, target_tone)
            loss = 0.7 * syl_loss + 0.3 * tone_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / max(num_batches, 1)
        ms_per_batch = (epoch_time / num_batches) * 1000

        if (epoch + 1) % config.log_every_epochs == 0 or epoch == config.epochs - 1:
            train_syl, train_tone = evaluate(model, train_loader, config.device)
            val_syl, val_tone = evaluate(model, val_loader, config.device)

            logger.info(
                f"Epoch {epoch+1:3d}/{config.epochs} | Loss: {avg_loss:.4f} | "
                f"Train: {train_syl:.4f}/{train_tone:.4f} | Val: {val_syl:.4f}/{val_tone:.4f} | "
                f"{ms_per_batch:.1f}ms/batch"
            )

            ckpt_path = config.checkpoint_dir / f"checkpoint_epoch{epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_syl_accuracy": val_syl,
                "val_tone_accuracy": val_tone,
            }, ckpt_path)

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

    parser = argparse.ArgumentParser(description="Train SyllablePredictorV6 (FlexAttention Transformer)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--log-every-epochs", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--data-source", type=str, default="synthetic")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--max-duration-s", type=float, default=10.0)

    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--overfit-test", action="store_true")

    # Transformer architecture (V6 FlexAttention)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--attention-window", type=int, default=32)

    parser.add_argument("--speed-variation", type=float, default=0.1)
    parser.add_argument("--pitch-shift", type=float, default=0.0)
    parser.add_argument("--formant-shift", type=float, default=0.0)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for FlexAttention optimization")

    args = parser.parse_args()

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        log_every_epochs=args.log_every_epochs,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        overfit_test=args.overfit_test,
        use_compile=args.compile,
    )

    logger = setup_logging(config.checkpoint_dir)
    logger.info("=" * 60)
    logger.info("SyllablePredictorV6 Training (FlexAttention Transformer)")
    logger.info("=" * 60)
    logger.info(f"Command: python {' '.join(sys.argv)}")

    sources = [s.strip() for s in args.data_source.split(",")]

    if args.data_dir:
        data_dirs = [Path(d.strip()) for d in args.data_dir.split(",")]
    else:
        data_dirs = []
        for source in sources:
            if source == "synthetic":
                data_dirs.append(SYNTHETIC_DIR)
            elif source == "aishell3":
                data_dirs.append(Path(__file__).parent.parent / "datasets" / "aishell3" / "data_aishell3")
            else:
                data_dirs.append(Path(__file__).parent.parent / "datasets" / source)

    if len(sources) != len(data_dirs):
        logger.error(f"Mismatch: {len(sources)} sources but {len(data_dirs)} directories")
        return

    logger.info(f"Data sources: {sources}")

    train_sentences, val_sentences, mel_cache = load_training_data(
        sources, data_dirs, logger,
        max_sentences_per_source=args.max_sentences,
    )
    if not train_sentences:
        logger.error("No training data!")
        return

    from mandarin_grader.model.syllable_predictor_v6 import SyllablePredictorV6, SyllablePredictorConfigV6, FLEX_ATTENTION_AVAILABLE
    model_config = SyllablePredictorConfigV6(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        attention_window=args.attention_window,
        max_audio_frames=int(args.max_duration_s * 100),
    )
    model = SyllablePredictorV6(model_config).to(config.device)

    # Optional: torch.compile for FlexAttention optimization
    if config.use_compile:
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {total_params:,} params ({total_params * 4 / 1024 / 1024:.2f} MB)")
    logger.info(f"Architecture: d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}")
    logger.info(f"Attention window: {args.attention_window}")
    logger.info(f"FlexAttention available: {FLEX_ATTENTION_AVAILABLE}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Augmentation: speed=±{args.speed_variation*100:.0f}%, pitch=±{args.pitch_shift:.1f}st, formant=±{args.formant_shift:.0f}%")

    preload = not mel_cache

    train_loader = create_dataloader(
        train_sentences, config.batch_size, shuffle=True, augment=True,
        preload=preload, logger=logger, mel_cache=mel_cache,
        max_duration_s=args.max_duration_s,
        speed_variation=args.speed_variation,
        pitch_shift_semitones=args.pitch_shift,
        formant_shift_percent=args.formant_shift,
    )
    val_loader = create_dataloader(
        val_sentences, config.batch_size, shuffle=False, augment=False,
        preload=preload, logger=logger, mel_cache=mel_cache,
        max_duration_s=args.max_duration_s,
    )
    logger.info(f"Batches: Train={len(train_loader)}, Val={len(val_loader)}")

    start_epoch = 0
    if args.resume:
        ckpt_path = config.checkpoint_dir / args.resume
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=config.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = checkpoint.get("epoch", 0)
            logger.info(f"Resumed from {ckpt_path} (epoch {start_epoch})")

    best_acc = train(model, train_loader, val_loader, config, logger, start_epoch)

    logger.info("=" * 60)
    logger.info(f"Training complete. Best combined accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
