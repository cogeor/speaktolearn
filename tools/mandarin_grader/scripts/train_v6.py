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

# Silence floor value in mel domain: log(epsilon) where epsilon=1e-9
# This must match the value used in extract_mel_spectrogram
MEL_SILENCE_FLOOR = np.log(1e-9)  # ≈ -20.72

SYNTHETIC_DIR = Path(__file__).parent.parent / "data" / "synthetic_train"
DEFAULT_CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints_v6"


class CollateFn:
    """Picklable collate function for multi-worker DataLoader on Windows."""

    def __init__(
        self,
        max_frames: int,
        random_padding: bool,
        augment: bool,
        augment_preset: str = "mobile",
        use_context_mask: bool = False,
        max_syllable_position: int | None = None,
    ):
        self.max_frames = max_frames
        self.random_padding = random_padding
        self.augment = augment
        self.augment_preset = augment_preset
        self.use_context_mask = use_context_mask
        self.max_syllable_position = max_syllable_position
        # Lazy import to avoid pickling issues
        self._vocab = None
        self._mel_config = None
        self._mask_token = None
        self._pad_token = 0

    def _get_vocab(self):
        if self._vocab is None:
            from mandarin_grader.model.syllable_predictor_v6 import SyllableVocab
            self._vocab = SyllableVocab()
            # MASK token = next ID after vocab
            self._mask_token = len(self._vocab)
        return self._vocab

    def _get_mel_config(self):
        if self._mel_config is None:
            from mandarin_grader.model.syllable_predictor_v4 import SyllablePredictorConfigV4
            self._mel_config = SyllablePredictorConfigV4()
        return self._mel_config

    def __call__(self, batch):
        import torch
        from mandarin_grader.model.syllable_predictor_v4 import extract_mel_spectrogram
        from mandarin_grader.data.mel_augmentation import get_preset_config, apply_mel_augmentation

        vocab = self._get_vocab()
        mel_config = self._get_mel_config()

        mel_aug_config = None
        if self.augment:
            mel_aug_config = get_preset_config(self.augment_preset)

        mels, positions, target_syls, target_tones = [], [], [], []
        context_ids_list = []

        for sample in batch:
            # 1. Get base mel spectrogram
            if sample.mel_full is not None:
                mel = sample.mel_full.copy()
            elif sample.audio_full is not None:
                mel = extract_mel_spectrogram(sample.audio_full, mel_config)
            else:
                raise ValueError(f"Sample {sample.sample_id} has neither mel nor audio")

            # 2. Apply augmentation locally within the worker process just before padding
            if self.augment and mel_aug_config is not None:
                mel = apply_mel_augmentation(mel, mel_aug_config)

            # 3. Truncate to max frames
            if mel.shape[1] > self.max_frames:
                mel = mel[:, :self.max_frames]

            # 4. Apply CMVN (Cepstral Mean and Variance Normalization)
            # Computed per-utterance over only the valid length to handle volume/EQ differences
            mel_mean = np.mean(mel, axis=1, keepdims=True)
            mel_std = np.std(mel, axis=1, keepdims=True)
            # Add epsilon to prevent division by zero for silent frames
            mel = (mel - mel_mean) / (mel_std + 1e-5)

            mels.append(mel)
            positions.append(sample.position)
            target_syls.append(vocab.encode(sample.target_syllable))
            target_tones.append(sample.target_tone)

            # Build context_ids for context-mask mode
            if self.use_context_mask:
                ctx = np.full(self.max_syllable_position, self._pad_token, dtype=np.int64)
                ctx_syls = sample.context_syllables or []
                for i in range(self.max_syllable_position):
                    if i == sample.position:
                        ctx[i] = self._mask_token
                    elif i < len(ctx_syls):
                        ctx[i] = vocab.encode(ctx_syls[i])
                    # else: stays PAD (short sentence)
                context_ids_list.append(ctx)

        # Pad mels to max_frames with silence floor
        # Even though CMVN centers around 0, 0.0 represents average energy, not silence.
        # Since the mel is now normalized (mean=0, std=1), a 5-sigma outlier is a solid true silence.
        CMVN_PAD_VALUE = -5.0
        n_mels = mels[0].shape[0]
        padded_mels = np.full((len(batch), n_mels, self.max_frames), CMVN_PAD_VALUE, dtype=np.float32)
        audio_masks = np.ones((len(batch), self.max_frames), dtype=bool)

        for i, mel in enumerate(mels):
            mel_len = mel.shape[1]
            if self.random_padding and self.augment and mel_len < self.max_frames:
                max_offset = self.max_frames - mel_len
                start_offset = np.random.randint(0, max_offset + 1)
            else:
                start_offset = 0
            end_offset = start_offset + mel_len
            padded_mels[i, :, start_offset:end_offset] = mel
            audio_masks[i, start_offset:end_offset] = False

        result = {
            "mel": torch.tensor(padded_mels, dtype=torch.float32),
            "position": torch.tensor(positions, dtype=torch.long),
            "audio_mask": torch.tensor(audio_masks, dtype=torch.bool),
            "target_syllable": torch.tensor(target_syls, dtype=torch.long),
            "target_tone": torch.tensor(target_tones, dtype=torch.long),
        }

        if self.use_context_mask:
            result["context_ids"] = torch.tensor(
                np.stack(context_ids_list), dtype=torch.long
            )

        return result


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
    load_test_set: bool = False,
) -> tuple[list, list, list, dict[str, np.ndarray]]:
    from mandarin_grader.data.synthetic_source import SyntheticDataSource
    from mandarin_grader.data.aishell_tar_source import AISHELL3TarDataSource
    from mandarin_grader.data.tts_source import TTSDataSource
    from mandarin_grader.data.autoregressive_dataset import SyntheticSentenceInfo

    # Note: AISHELL3TarDataSource works for any tar dataset with same format
    # (including openai_tts_tar, etc.)
    source_classes = {
        "synthetic": SyntheticDataSource(),
        "aishell3": AISHELL3TarDataSource(),
        "openai_tts_tar": AISHELL3TarDataSource(),
        "tts_tar": AISHELL3TarDataSource(),  # Generic tar format
        "tts": TTSDataSource(),  # Raw wav format
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
            
            logger.info("  Starting source.load()...")
            sentences = source.load(data_dir, **kwargs)
            logger.info(f"  Finished source.load(). Loaded {len(sentences)} sentences from {source_name}")

            logger.info("  Starting get_mel_cache()...")
            if hasattr(source, "get_mel_cache"):
                source_cache = source.get_mel_cache()
                mel_cache.update(source_cache)
                logger.info(f"  Finished get_mel_cache(). Mel cache: {len(source_cache)} files pre-loaded")
            else:
                logger.info("  No get_mel_cache method found.")

            logger.info("  Starting appending to all_sentences...")
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
            logger.info("  Finished appending to all_sentences.")

        except Exception as e:
            logger.error(f"  Error loading {source_name}: {e}")
            continue

    logger.info(f"Total sentences: {len(all_sentences)}")

    if not all_sentences:
        return [], [], [], {}

    logger.info("  Starting permutation...")
    np.random.seed(42)
    indices = np.random.permutation(len(all_sentences))
    split_idx = int(len(all_sentences) * train_split)
    logger.info("  Finished permutation.")

    logger.info("  Starting train/val split...")
    train = [all_sentences[i] for i in indices[:split_idx]]
    val = [all_sentences[i] for i in indices[split_idx:]]
    logger.info(f"Train: {len(train)}, Val: {len(val)}")

    # Load test set (official AISHELL-3 test split with different speakers)
    test_sentences = []
    if load_test_set:
        logger.info("  Starting test set loading...")
        for source_name, data_dir in zip(sources, data_dirs):
            if source_name != "aishell3":
                continue
            try:
                source = source_classes[source_name]
                logger.info(f"    Starting test source.load() for {source_name}...")
                test_sents = source.load(data_dir, split="test")
                # Limit the test set to essentially 500 samples so we don't spend ~4 minutes extracting
                # 24,000+ files from tarballs just for standard evaluation loops.
                test_sents = test_sents[:500] 
                logger.info(f"    Loaded {len(test_sents)} test sentences from {source_name} (subset)")

                logger.info(f"    Starting test get_mel_cache() for {source_name}...")
                if hasattr(source, "get_mel_cache"):
                    test_cache = source.get_mel_cache()
                    mel_cache.update(test_cache)
                    logger.info(f"    Test mel cache: {len(test_cache)} files")

                logger.info(f"    Starting test appending to test_sentences...")
                for s in test_sents:
                    test_sentences.append(SyntheticSentenceInfo(
                        id=s.id,
                        audio_path=s.audio_path,
                        text=s.text,
                        syllables=s.syllables,
                        syllable_boundaries=s.syllable_boundaries,
                        sample_rate=s.sample_rate,
                        total_samples=s.total_samples,
                    ))
                logger.info(f"    Finished test appending to test_sentences.")
            except Exception as e:
                logger.error(f"  Error loading test set from {source_name}: {e}")

        logger.info(f"Test: {len(test_sentences)}")

    return train, val, test_sentences, mel_cache


def _worker_init_fn(worker_id):
    """Ensure independent random state for each worker process to avoid RNG lock contention."""
    import numpy as np
    import random
    import torch
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloader(
    sentences: list,
    batch_size: int,
    shuffle: bool,
    augment: bool,
    preload: bool = False,
    logger=None,
    mel_cache: dict | None = None,
    max_duration_s: float = 10.0,
    max_syllable_position: int | None = None,
    random_padding: bool = True,
    augment_preset: str = "none",
    use_context_mask: bool = False,
):
    """Create dataloader for V6 training.

    Args:
        sentences: List of sentence info objects
        batch_size: Batch size
        shuffle: Whether to shuffle data
        augment: Whether to apply augmentation
        preload: Whether to preload audio into memory
        logger: Logger instance
        mel_cache: Precomputed mel spectrograms cache
        max_duration_s: Maximum audio duration in seconds
        max_syllable_position: Only train on syllables at positions < this value
        random_padding: Whether to randomly place audio within the padded frame
        augment_preset: Named preset for mel-domain augmentations (resolved in CollateFn)
        use_context_mask: Whether to build context_ids for context-mask mode
    """
    from torch.utils.data import DataLoader
    from mandarin_grader.data.full_sentence_dataset import FullSentenceDataset
    from mandarin_grader.model.syllable_predictor_v6 import SyllablePredictorConfigV6

    config = SyllablePredictorConfigV6()

    dataset = FullSentenceDataset(
        sentences=sentences,
        sample_rate=config.sample_rate,
        max_duration_s=max_duration_s,
        max_syllable_position=max_syllable_position,
        augment=augment,
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
    collate_fn = CollateFn(
        max_frames, random_padding, augment,
        augment_preset=augment_preset,
        use_context_mask=use_context_mask,
        max_syllable_position=max_syllable_position,
    )

    # Use multiple workers for parallel data loading/augmentation
    num_workers = 4 if augment else 0
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_fn, num_workers=num_workers,
                      worker_init_fn=_worker_init_fn if num_workers > 0 else None,
                      persistent_workers=num_workers > 0)


def evaluate(model, dataloader, device: str, max_batches: int | None = None, use_context_mask: bool = False) -> tuple[float, float]:
    import torch
    model.eval()
    syl_correct, tone_correct, total = 0, 0, 0
    batches_processed = 0

    with torch.no_grad():
        for batch in dataloader:
            mel = batch["mel"].to(device)
            audio_mask = batch["audio_mask"].to(device)

            if use_context_mask and "context_ids" in batch:
                context_ids = batch["context_ids"].to(device)
                syl_logits, tone_logits = model(mel, audio_mask=audio_mask, context_ids=context_ids)
            else:
                position = batch["position"].to(device)
                syl_logits, tone_logits = model(mel, position, audio_mask)

            syl_correct += (syl_logits.argmax(-1).cpu() == batch["target_syllable"]).sum().item()
            tone_correct += (tone_logits.argmax(-1).cpu() == batch["target_tone"]).sum().item()
            total += mel.shape[0]

            batches_processed += 1
            if max_batches is not None and batches_processed >= max_batches:
                break

    return syl_correct / max(total, 1), tone_correct / max(total, 1)


def train(model, train_loader, val_loader, config: TrainingConfig, logger, start_epoch: int = 0, test_loader=None, use_context_mask: bool = False):
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
    if use_context_mask:
        logger.info("Context-mask mode: ENABLED")

    for epoch in range(start_epoch, config.epochs):
        model.train()
        total_loss, num_batches = 0, 0
        epoch_start = time.time()

        for batch in train_loader:
            mel = batch["mel"].to(config.device)
            audio_mask = batch["audio_mask"].to(config.device)
            target_syl = batch["target_syllable"].to(config.device)
            target_tone = batch["target_tone"].to(config.device)

            optimizer.zero_grad()

            if use_context_mask and "context_ids" in batch:
                context_ids = batch["context_ids"].to(config.device)
                syl_logits, tone_logits = model(mel, audio_mask=audio_mask, context_ids=context_ids)
            else:
                position = batch["position"].to(config.device)
                syl_logits, tone_logits = model(mel, position, audio_mask)

            syl_loss = syl_criterion(syl_logits, target_syl)
            tone_loss = tone_criterion(tone_logits, target_tone)
            loss = 0.7 * syl_loss + 0.3 * tone_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            if num_batches % 50 == 0:
                cur_ms_per_batch = ((time.time() - epoch_start) / num_batches) * 1000
                logger.info(f"  Epoch {epoch+1} | Batch {num_batches}/{len(train_loader)} | Loss: {loss.item():.4f} | {cur_ms_per_batch:.1f}ms/batch")

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / max(num_batches, 1)
        ms_per_batch = (epoch_time / num_batches) * 1000

        if (epoch + 1) % config.log_every_epochs == 0 or epoch == config.epochs - 1:
            eval_batches = 50  # Hardcoded subsampling to speed up validation

            train_syl, train_tone = evaluate(model, train_loader, config.device, max_batches=eval_batches, use_context_mask=use_context_mask)
            val_syl, val_tone = evaluate(model, val_loader, config.device, max_batches=eval_batches, use_context_mask=use_context_mask)

            logger.info(
                f"Epoch {epoch+1:3d}/{config.epochs} | Loss: {avg_loss:.4f} | "
                f"Train (sub): {train_syl:.4f}/{train_tone:.4f} | Val (sub): {val_syl:.4f}/{val_tone:.4f} | "
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

                # Evaluate on test set if available
                test_syl, test_tone = 0.0, 0.0
                if test_loader is not None:
                    test_syl, test_tone = evaluate(model, test_loader, config.device, max_batches=eval_batches, use_context_mask=use_context_mask)
                    logger.info(f"  -> New best model! Val (sub): {val_combined:.4f} | Test (sub): {test_syl:.4f}/{test_tone:.4f}")
                else:
                    logger.info(f"  -> New best model! Combined (sub): {val_combined:.4f}")

                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "val_syl_accuracy": val_syl,
                    "val_tone_accuracy": val_tone,
                    "test_syl_accuracy": test_syl,
                    "test_tone_accuracy": test_tone,
                }, best_path)

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
    parser.add_argument("--max-syllable-position", type=int, default=None,
                        help="Only train on syllables at positions < this value (for short audio tests)")

    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--overfit-test", action="store_true")

    # Transformer architecture (V6 FlexAttention)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=384)
    parser.add_argument("--attention-window", type=int, default=32)

    # Context-mask mode
    parser.add_argument("--use-context-mask", action="store_true",
                        help="Use context-mask mode (BERT-style masked prediction with sentence context). "
                             "Requires --max-syllable-position to be set.")

    # Mel-domain augmentation (works with precomputed mel cache)
    parser.add_argument("--augment-preset", type=str, default="mobile",
                        choices=["none", "light", "studio", "mobile"],
                        help="Augmentation preset: none, light, studio, mobile (default: mobile)")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable all augmentation (same as --augment-preset none)")
    parser.add_argument("--no-random-padding", action="store_true",
                        help="Disable random start/end padding (always pad at end)")
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
                data_dirs.append(Path(__file__).parent.parent / "datasets" / "aishell3_tar")
            elif source == "openai_tts_tar":
                data_dirs.append(Path(__file__).parent.parent / "datasets" / "openai_tts_tar")
            else:
                data_dirs.append(Path(__file__).parent.parent / "datasets" / source)

    if len(sources) != len(data_dirs):
        logger.error(f"Mismatch: {len(sources)} sources but {len(data_dirs)} directories")
        return

    logger.info(f"Data sources: {sources}")

    train_sentences, val_sentences, test_sentences, mel_cache = load_training_data(
        sources, data_dirs, logger,
        max_sentences_per_source=args.max_sentences,
        load_test_set=True,
    )
    if not train_sentences:
        logger.error("No training data!")
        return

    from mandarin_grader.model.syllable_predictor_v6 import SyllablePredictorV6, SyllablePredictorConfigV6, FLEX_ATTENTION_AVAILABLE
    model_config = SyllablePredictorConfigV6(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_feedforward=args.dim_feedforward,
        attention_window=args.attention_window,
        max_audio_frames=int(args.max_duration_s * 100),
        max_context_positions=args.max_syllable_position or 4,
    )
    model = SyllablePredictorV6(model_config).to(config.device)

    # Optional: torch.compile for FlexAttention optimization
    if config.use_compile:
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {total_params:,} params ({total_params * 4 / 1024 / 1024:.2f} MB)")
    logger.info(f"Architecture: d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}, dim_ff={args.dim_feedforward}")
    logger.info(f"Audio duration: {args.max_duration_s}s ({int(args.max_duration_s * 100)} frames, {int(args.max_duration_s * 100) // 4} after CNN)")
    if args.max_syllable_position:
        logger.info(f"Max syllable position: {args.max_syllable_position} (only training on first {args.max_syllable_position} syllables)")
    logger.info(f"Attention window: {args.attention_window} (sliding window + global on pos 0)")
    logger.info(f"FlexAttention available: {FLEX_ATTENTION_AVAILABLE}")
    logger.info(f"Context-mask mode: {'ENABLED' if args.use_context_mask else 'disabled'}")
    if args.use_context_mask:
        logger.info(f"  Context positions: {args.max_syllable_position}")
    logger.info(f"Device: {config.device}")

    # Configure mel-domain augmentation pipeline
    from mandarin_grader.data.mel_augmentation import get_preset_config

    random_padding = not args.no_random_padding
    preset = "none" if args.no_augment else args.augment_preset

    if preset == "none":
        mel_aug_config = None
        logger.info("Augmentation: DISABLED")
    else:
        mel_aug_config = get_preset_config(preset)
        cfg = mel_aug_config

        # Log configuration details
        logger.info(f"Augmentation preset: {preset}")
        logger.info(
            f"  Time stretch: {cfg.time_stretch.range[0]:.2f}-{cfg.time_stretch.range[1]:.2f} "
            f"(prob={cfg.time_stretch.prob:.0%})"
        )
        logger.info(
            f"  Gain: {cfg.gain.db_range[0]:+.0f} to {cfg.gain.db_range[1]:+.0f}dB "
            f"(prob={cfg.gain.prob:.0%})"
        )
        logger.info(
            f"  SpecAugment: F={cfg.spec_augment.freq_mask_param}, T={cfg.spec_augment.time_mask_param} "
            f"(prob={cfg.spec_augment.prob:.0%})"
        )
        if cfg.low_shelf_boost.enabled:
            logger.info(
                f"  LF boost: bins {cfg.low_shelf_boost.cutoff_bin_range}, "
                f"{cfg.low_shelf_boost.boost_db_range[0]:.0f}-{cfg.low_shelf_boost.boost_db_range[1]:.0f}dB "
                f"(prob={cfg.low_shelf_boost.prob:.0%})"
            )
        if cfg.spectral_noise.enabled:
            logger.info(
                f"  Spectral noise: SNR {cfg.spectral_noise.snr_db_range[0]:.0f}-{cfg.spectral_noise.snr_db_range[1]:.0f}dB "
                f"(prob={cfg.spectral_noise.prob:.0%})"
            )
        if cfg.temporal_smear.enabled:
            logger.info(
                f"  Temporal smear: {cfg.temporal_smear.decay_frames_range[0]}-{cfg.temporal_smear.decay_frames_range[1]} frames, "
                f"wet {cfg.temporal_smear.wet_ratio_range[0]:.0%}-{cfg.temporal_smear.wet_ratio_range[1]:.0%} "
                f"(prob={cfg.temporal_smear.prob:.0%})"
            )
        logger.info(f"  Random padding: {'enabled' if random_padding else 'disabled'}")

    preload = not mel_cache

    train_loader = create_dataloader(
        train_sentences, config.batch_size, shuffle=True, augment=True,
        preload=preload, logger=logger, mel_cache=mel_cache,
        max_duration_s=args.max_duration_s,
        max_syllable_position=args.max_syllable_position,
        random_padding=random_padding,
        augment_preset=preset,
        use_context_mask=args.use_context_mask,
    )
    val_loader = create_dataloader(
        val_sentences, config.batch_size, shuffle=False, augment=False,
        preload=preload, logger=logger, mel_cache=mel_cache,
        max_duration_s=args.max_duration_s,
        max_syllable_position=args.max_syllable_position,
        random_padding=False,  # Validation always pads at end
        use_context_mask=args.use_context_mask,
    )

    # Create test loader if test set available
    test_loader = None
    if test_sentences:
        test_loader = create_dataloader(
            test_sentences, config.batch_size, shuffle=False, augment=False,
            preload=preload, logger=logger, mel_cache=mel_cache,
            max_duration_s=args.max_duration_s,
            max_syllable_position=args.max_syllable_position,
            random_padding=False,
            use_context_mask=args.use_context_mask,
        )
        logger.info(f"Batches: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")
    else:
        logger.info(f"Batches: Train={len(train_loader)}, Val={len(val_loader)}")

    start_epoch = 0
    if args.resume:
        ckpt_path = config.checkpoint_dir / args.resume
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=config.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = checkpoint.get("epoch", 0)
            logger.info(f"Resumed from {ckpt_path} (epoch {start_epoch})")

    best_acc = train(model, train_loader, val_loader, config, logger, start_epoch, test_loader, use_context_mask=args.use_context_mask)

    logger.info("=" * 60)
    logger.info(f"Training complete. Best combined accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
