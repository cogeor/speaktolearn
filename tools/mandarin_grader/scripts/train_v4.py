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

from mandarin_grader.pitch import extract_f0_pyin, normalize_f0, hz_to_semitones

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


def get_curriculum_ratio(epoch: int) -> float:
    """Get curriculum ratio for mixing AISHELL-3 and TTS data.

    Returns ratio of AISHELL-3 samples (1.0 = 100% native, 0.0 = 100% TTS).
    Gradually shifts from native to TTS data over training epochs.

    Args:
        epoch: Current epoch (1-indexed)

    Returns:
        Ratio of AISHELL-3 samples in the mix
    """
    if epoch <= 10:
        return 0.9  # 90% AISHELL-3, 10% TTS
    elif epoch <= 20:
        return 0.7  # 70% AISHELL-3, 30% TTS
    elif epoch <= 30:
        return 0.5  # 50% AISHELL-3, 50% TTS
    else:
        return 0.3  # 30% AISHELL-3, 70% TTS


def get_domain_lambda(epoch: int, total_epochs: int) -> float:
    """Get domain lambda for gradient reversal strength.

    Ramps from 0 to 1.0 over training using sigmoid schedule for smooth transition.
    This gradually increases the domain adversarial loss contribution.

    Args:
        epoch: Current epoch (1-indexed)
        total_epochs: Total number of training epochs

    Returns:
        Domain lambda value between 0 and 1
    """
    progress = epoch / max(total_epochs, 1)
    # Sigmoid ramp: 2 / (1 + exp(-10p)) - 1
    return 2.0 / (1.0 + np.exp(-10 * progress)) - 1.0


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

    # Domain adversarial training
    domain_adapt: bool = False
    tts_dir: Path | None = None


def extract_f0_features(
    audio: np.ndarray,
    sr: int = 16000,
    hop_length: int = 160,
) -> np.ndarray:
    """Extract normalized F0 features from audio.

    Args:
        audio: Audio samples [n_samples]
        sr: Sample rate
        hop_length: Hop length for frame extraction

    Returns:
        Normalized F0 [n_frames], 0 for unvoiced
    """
    f0_hz, voicing = extract_f0_pyin(
        audio, sr=sr, fmin=50.0, fmax=500.0, hop_length=hop_length
    )
    semitones = hz_to_semitones(f0_hz, ref_hz=100.0)
    normalized = normalize_f0(semitones, voicing)
    return normalized.astype(np.float32)


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


class MixedDomainDataset:
    """Dataset that mixes samples from two domains with curriculum learning.

    Wraps two AutoregressiveDatasets (native and TTS) and provides a combined
    interface with domain labels. The mixing ratio can be adjusted per epoch.

    Domain labels:
        0 = AISHELL-3 / native speech
        1 = TTS-generated speech
    """

    def __init__(
        self,
        native_dataset,  # AutoregressiveDataset for AISHELL-3
        tts_dataset,     # AutoregressiveDataset for TTS
        native_ratio: float = 0.9,
    ):
        """Initialize mixed domain dataset.

        Args:
            native_dataset: Dataset for native speech (domain 0)
            tts_dataset: Dataset for TTS speech (domain 1)
            native_ratio: Initial ratio of native samples (0.0 to 1.0)
        """
        self.native_dataset = native_dataset
        self.tts_dataset = tts_dataset
        self.native_ratio = native_ratio

        # Track which samples are from which domain
        self._native_len = len(native_dataset)
        self._tts_len = len(tts_dataset)
        self._total_len = self._native_len + self._tts_len

        # Build sampling indices
        self._rebuild_indices()

    def _rebuild_indices(self):
        """Rebuild sampling indices based on current ratio."""
        # Number of samples to take from each domain
        target_native = int(self._total_len * self.native_ratio)
        target_tts = self._total_len - target_native

        # Cap to available samples
        native_count = min(target_native, self._native_len)
        tts_count = min(target_tts, self._tts_len)

        # Build index list: (domain, idx_in_domain)
        self._indices = []

        # Sample from native
        native_indices = np.random.choice(self._native_len, size=native_count, replace=native_count > self._native_len)
        for idx in native_indices:
            self._indices.append((0, idx))

        # Sample from TTS
        tts_indices = np.random.choice(self._tts_len, size=tts_count, replace=tts_count > self._tts_len)
        for idx in tts_indices:
            self._indices.append((1, idx))

        # Shuffle
        np.random.shuffle(self._indices)

    def set_epoch(self, epoch: int):
        """Update mixing ratio for curriculum learning and rebuild indices.

        Args:
            epoch: Current epoch (1-indexed)
        """
        self.native_ratio = get_curriculum_ratio(epoch)
        self._rebuild_indices()

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        """Get sample with domain label.

        Returns:
            Tuple of (sample, domain_label) where domain_label is 0 or 1
        """
        domain, domain_idx = self._indices[idx]
        if domain == 0:
            sample = self.native_dataset[domain_idx]
        else:
            sample = self.tts_dataset[domain_idx]
        return sample, domain


def create_dataloader(sentences: list, batch_size: int, shuffle: bool, augment: bool, context_mode: str = "pinyin", preload: bool = False, logger=None, mel_cache: dict | None = None, use_pitch: bool = False, include_domain_labels: bool = False):
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
        use_pitch: Whether to extract and return F0 pitch features
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
        f0_list = [] if use_pitch else None
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

            # F0 extraction (if enabled)
            if use_pitch:
                if sample.f0_chunk is not None:
                    # Use pre-extracted F0 from dataset
                    f0_list.append(sample.f0_chunk)
                elif sample.audio_chunk is not None:
                    # Extract F0 from audio
                    f0 = extract_f0_features(sample.audio_chunk, config.sample_rate, config.hop_length)
                    f0_list.append(f0)
                else:
                    # For precomputed mel without F0, use zeros
                    f0_list.append(np.zeros(mel.shape[1], dtype=np.float32))

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

        result = {
            "mel": torch.tensor(padded_mels, dtype=torch.float32),
            "pinyin_ids": torch.tensor(padded_pinyin, dtype=torch.long),
            "audio_mask": torch.tensor(audio_masks, dtype=torch.bool),
            "pinyin_mask": torch.tensor(pinyin_masks, dtype=torch.bool),
            "target_syllable": torch.tensor(target_syls, dtype=torch.long),
            "target_tone": torch.tensor(target_tones, dtype=torch.long),
        }

        # Pad and add F0 if used
        if use_pitch:
            max_f0_len = max(len(f) for f in f0_list)
            padded_f0 = np.zeros((len(batch), max_f0_len), dtype=np.float32)
            for i, f0 in enumerate(f0_list):
                padded_f0[i, :len(f0)] = f0
            result["f0"] = torch.tensor(padded_f0, dtype=torch.float32)

        return result

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_fn, num_workers=0)


def create_mixed_domain_dataloader(
    native_sentences: list,
    tts_sentences: list,
    batch_size: int,
    augment: bool,
    context_mode: str = "pinyin",
    preload: bool = False,
    logger=None,
    native_mel_cache: dict | None = None,
    tts_mel_cache: dict | None = None,
    use_pitch: bool = False,
    initial_native_ratio: float = 0.9,
):
    """Create DataLoader for mixed domain training with domain labels.

    Args:
        native_sentences: List of sentence info for native speech (AISHELL-3)
        tts_sentences: List of sentence info for TTS speech
        batch_size: Batch size
        augment: Whether to augment audio
        context_mode: "pinyin" = actual syllables, "position" = only syllable index
        preload: Whether to preload all audio into memory
        logger: Logger for progress reporting
        native_mel_cache: Pre-loaded mel cache for native source
        tts_mel_cache: Pre-loaded mel cache for TTS source
        use_pitch: Whether to extract and return F0 pitch features
        initial_native_ratio: Initial curriculum ratio (default 0.9 = 90% native)

    Returns:
        Tuple of (DataLoader, MixedDomainDataset) - dataset returned for set_epoch calls
    """
    import torch
    from torch.utils.data import DataLoader
    from mandarin_grader.data.autoregressive_dataset import AutoregressiveDataset, spec_augment
    from mandarin_grader.model.syllable_predictor_v4 import (
        SyllablePredictorConfigV4, SyllableVocab, extract_mel_spectrogram,
    )

    config = SyllablePredictorConfigV4()
    vocab = SyllableVocab()

    # Create native dataset
    native_dataset = AutoregressiveDataset(
        sentences=native_sentences,
        sample_rate=config.sample_rate,
        chunk_duration_s=1.0,
        margin_s=0.1,
        augment=augment,
    )
    if native_mel_cache:
        native_dataset._mel_cache.update(native_mel_cache)
        if logger:
            logger.info(f"  Injected {len(native_mel_cache)} precomputed mel entries (native)")

    # Create TTS dataset
    tts_dataset = AutoregressiveDataset(
        sentences=tts_sentences,
        sample_rate=config.sample_rate,
        chunk_duration_s=1.0,
        margin_s=0.1,
        augment=augment,
    )
    if tts_mel_cache:
        tts_dataset._mel_cache.update(tts_mel_cache)
        if logger:
            logger.info(f"  Injected {len(tts_mel_cache)} precomputed mel entries (TTS)")

    # Preload audio if needed
    if preload:
        def progress(loaded, total):
            if logger:
                logger.info(f"  Preloading audio: {loaded}/{total} ({100*loaded/total:.0f}%)")
        if logger:
            logger.info("Preloading native audio files into memory...")
        native_dataset.preload_audio(progress_callback=progress)
        if logger:
            logger.info("Preloading TTS audio files into memory...")
        tts_dataset.preload_audio(progress_callback=progress)

    # Create mixed dataset
    mixed_dataset = MixedDomainDataset(
        native_dataset=native_dataset,
        tts_dataset=tts_dataset,
        native_ratio=initial_native_ratio,
    )

    def collate_fn(batch):
        """Collate function for mixed domain batches."""
        mels, pinyin_ids_list, target_syls, target_tones, domain_labels = [], [], [], [], []
        f0_list = [] if use_pitch else None
        max_pinyin_len = 0

        for sample, domain in batch:
            if sample.mel_chunk is not None:
                mel = sample.mel_chunk
            elif sample.audio_chunk is not None:
                mel = extract_mel_spectrogram(sample.audio_chunk, config)
                if augment:
                    mel = spec_augment(mel)
            else:
                raise ValueError(f"Sample {sample.sample_id} has neither mel_chunk nor audio_chunk")
            mels.append(mel)

            # F0 extraction (if enabled)
            if use_pitch:
                if sample.f0_chunk is not None:
                    f0_list.append(sample.f0_chunk)
                elif sample.audio_chunk is not None:
                    f0 = extract_f0_features(sample.audio_chunk, config.sample_rate, config.hop_length)
                    f0_list.append(f0)
                else:
                    f0_list.append(np.zeros(mel.shape[1], dtype=np.float32))

            if context_mode == "pinyin":
                ids = vocab.encode_sequence(sample.pinyin_context, add_bos=True)
            else:
                position_token = 2 + sample.syllable_idx
                ids = [vocab.bos_token, position_token]

            pinyin_ids_list.append(ids)
            max_pinyin_len = max(max_pinyin_len, len(ids))

            target_syls.append(vocab.encode(sample.target_syllable))
            target_tones.append(sample.target_tone)
            domain_labels.append(domain)

        # Pad mels
        max_time = max(m.shape[1] for m in mels)
        n_mels = mels[0].shape[0]
        padded_mels = np.zeros((len(batch), n_mels, max_time), dtype=np.float32)
        audio_masks = np.zeros((len(batch), max_time), dtype=bool)
        for i, mel in enumerate(mels):
            padded_mels[i, :, :mel.shape[1]] = mel
            audio_masks[i, mel.shape[1]:] = True

        # Pad pinyin
        padded_pinyin = np.zeros((len(batch), max_pinyin_len), dtype=np.int64)
        pinyin_masks = np.zeros((len(batch), max_pinyin_len), dtype=bool)
        for i, ids in enumerate(pinyin_ids_list):
            padded_pinyin[i, :len(ids)] = ids
            pinyin_masks[i, len(ids):] = True

        result = {
            "mel": torch.tensor(padded_mels, dtype=torch.float32),
            "pinyin_ids": torch.tensor(padded_pinyin, dtype=torch.long),
            "audio_mask": torch.tensor(audio_masks, dtype=torch.bool),
            "pinyin_mask": torch.tensor(pinyin_masks, dtype=torch.bool),
            "target_syllable": torch.tensor(target_syls, dtype=torch.long),
            "target_tone": torch.tensor(target_tones, dtype=torch.long),
            "domain_label": torch.tensor(domain_labels, dtype=torch.long),
        }

        # Pad and add F0 if used
        if use_pitch:
            max_f0_len = max(len(f) for f in f0_list)
            padded_f0 = np.zeros((len(batch), max_f0_len), dtype=np.float32)
            for i, f0 in enumerate(f0_list):
                padded_f0[i, :len(f0)] = f0
            result["f0"] = torch.tensor(padded_f0, dtype=torch.float32)

        return result

    dataloader = DataLoader(
        mixed_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )

    return dataloader, mixed_dataset


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

            # Handle F0 if present
            f0 = batch.get("f0")
            if f0 is not None:
                f0 = f0.to(device)

            syl_logits, tone_logits = model(mel, pinyin_ids, audio_mask, pinyin_mask, f0=f0)

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

            # Handle F0 if present
            f0 = batch.get("f0")
            if f0 is not None:
                f0 = f0.to(device)

            syl_logits, tone_logits = model(mel, pinyin_ids, audio_mask, pinyin_mask, f0=f0)
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


def evaluate_by_domain(model, dataloader, device: str) -> dict:
    """Evaluate model with per-domain breakdown.

    Args:
        model: The model to evaluate
        dataloader: DataLoader with domain_label in batches
        device: Device to use

    Returns:
        Dict with per-domain and combined accuracies:
        {
            "aishell3": {"syllable_acc": float, "tone_acc": float, "total": int},
            "tts": {"syllable_acc": float, "tone_acc": float, "total": int},
            "combined": {"syllable_acc": float, "tone_acc": float, "total": int},
        }
    """
    import torch
    model.eval()

    # Per-domain counters
    domain_stats = {
        0: {"syl_correct": 0, "tone_correct": 0, "total": 0},  # AISHELL-3
        1: {"syl_correct": 0, "tone_correct": 0, "total": 0},  # TTS
    }

    with torch.no_grad():
        for batch in dataloader:
            mel = batch["mel"].to(device)
            pinyin_ids = batch["pinyin_ids"].to(device)
            audio_mask = batch["audio_mask"].to(device)
            pinyin_mask = batch["pinyin_mask"].to(device)
            domain_labels = batch["domain_label"]

            # Handle F0 if present
            f0 = batch.get("f0")
            if f0 is not None:
                f0 = f0.to(device)

            # Get model output (ignore domain_logits if returned)
            output = model(mel, pinyin_ids, audio_mask, pinyin_mask, f0=f0)
            syl_logits, tone_logits = output[0], output[1]

            syl_pred = syl_logits.argmax(-1).cpu()
            tone_pred = tone_logits.argmax(-1).cpu()

            # Track per-domain metrics
            for i in range(mel.shape[0]):
                domain = domain_labels[i].item()
                target_syl = batch["target_syllable"][i].item()
                target_tone = batch["target_tone"][i].item()

                domain_stats[domain]["total"] += 1
                if syl_pred[i].item() == target_syl:
                    domain_stats[domain]["syl_correct"] += 1
                if tone_pred[i].item() == target_tone:
                    domain_stats[domain]["tone_correct"] += 1

    # Compute accuracies
    def compute_acc(stats):
        total = max(stats["total"], 1)
        return {
            "syllable_acc": stats["syl_correct"] / total,
            "tone_acc": stats["tone_correct"] / total,
            "total": stats["total"],
        }

    aishell_stats = compute_acc(domain_stats[0])
    tts_stats = compute_acc(domain_stats[1])

    # Combined stats
    combined_total = domain_stats[0]["total"] + domain_stats[1]["total"]
    combined_syl = domain_stats[0]["syl_correct"] + domain_stats[1]["syl_correct"]
    combined_tone = domain_stats[0]["tone_correct"] + domain_stats[1]["tone_correct"]

    return {
        "aishell3": aishell_stats,
        "tts": tts_stats,
        "combined": {
            "syllable_acc": combined_syl / max(combined_total, 1),
            "tone_acc": combined_tone / max(combined_total, 1),
            "total": combined_total,
        },
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

    # Handle F0 if present
    f0 = batch.get("f0")
    if f0 is not None:
        f0 = f0.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate * 10)
    syl_criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    tone_criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    model.train()
    for step in range(config.overfit_steps):
        optimizer.zero_grad()
        syl_logits, tone_logits = model(mel, pinyin_ids, audio_mask, pinyin_mask, f0=f0)
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
        syl_logits, tone_logits = model(mel, pinyin_ids, audio_mask, pinyin_mask, f0=f0)
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

            # Handle F0 if present
            f0 = batch.get("f0")
            if f0 is not None:
                f0 = f0.to(config.device)

            optimizer.zero_grad()
            syl_logits, tone_logits = model(mel, pinyin_ids, audio_mask, pinyin_mask, f0=f0)
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


def train_domain_adapt(
    model,
    train_loader,
    mixed_dataset,  # MixedDomainDataset for curriculum updates
    val_loader,
    config: TrainingConfig,
    logger,
    start_epoch: int = 0,
):
    """Training loop with domain adversarial adaptation.

    Uses gradient reversal to learn domain-invariant features and
    curriculum learning to gradually shift from native to TTS data.

    Args:
        model: SyllablePredictorV4 with use_domain_adversarial=True
        train_loader: DataLoader from create_mixed_domain_dataloader
        mixed_dataset: MixedDomainDataset for set_epoch calls
        val_loader: Validation DataLoader (also with domain labels)
        config: TrainingConfig
        logger: Logger
        start_epoch: Epoch to resume from

    Returns:
        Best validation accuracy
    """
    import torch

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = config.epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    syl_criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    tone_criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    domain_criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0

    logger.info(f"Starting domain adversarial training from epoch {start_epoch + 1} to {config.epochs}")
    logger.info(f"Warmup steps: {warmup_steps} ({warmup_steps / len(train_loader):.1f} epochs)")
    logger.info(f"Logging every {config.log_every_epochs} epochs")

    import time as _time
    for epoch in range(start_epoch, config.epochs):
        # Update curriculum ratio for this epoch
        mixed_dataset.set_epoch(epoch + 1)
        native_ratio = get_curriculum_ratio(epoch + 1)
        domain_lambda = get_domain_lambda(epoch + 1, config.epochs)

        logger.info(f"Epoch {epoch+1}: native_ratio={native_ratio:.2f}, domain_lambda={domain_lambda:.3f}")

        model.train()
        total_loss, total_task_loss, total_domain_loss = 0, 0, 0
        domain_correct, domain_total = 0, 0
        num_batches = 0
        epoch_start = _time.time()

        for batch in train_loader:
            # Progress logging every 500 batches
            if num_batches > 0 and num_batches % 500 == 0:
                elapsed = _time.time() - epoch_start
                ms_per_batch = elapsed / num_batches * 1000
                eta_min = (len(train_loader) - num_batches) * ms_per_batch / 60000
                logger.info(
                    f"  Epoch {epoch+1} batch {num_batches}/{len(train_loader)} | "
                    f"{ms_per_batch:.0f}ms/batch | ETA: {eta_min:.1f}min"
                )

            mel = batch["mel"].to(config.device)
            pinyin_ids = batch["pinyin_ids"].to(config.device)
            audio_mask = batch["audio_mask"].to(config.device)
            pinyin_mask = batch["pinyin_mask"].to(config.device)
            target_syl = batch["target_syllable"].to(config.device)
            target_tone = batch["target_tone"].to(config.device)
            domain_labels = batch["domain_label"].to(config.device)

            # Handle F0 if present
            f0 = batch.get("f0")
            if f0 is not None:
                f0 = f0.to(config.device)

            optimizer.zero_grad()

            # Forward with domain_lambda > 0 to get domain logits
            output = model(mel, pinyin_ids, audio_mask, pinyin_mask, f0=f0, domain_lambda=domain_lambda)
            syl_logits, tone_logits = output[0], output[1]

            # Task losses
            syl_loss = syl_criterion(syl_logits, target_syl)
            tone_loss = tone_criterion(tone_logits, target_tone)
            task_loss = 0.7 * syl_loss + 0.3 * tone_loss

            # Domain loss (if domain logits returned)
            if len(output) == 3:
                domain_logits = output[2]
                domain_loss = domain_criterion(domain_logits, domain_labels)
                # Total loss: task loss + domain loss (gradient already reversed in forward)
                loss = task_loss + domain_lambda * domain_loss

                # Track domain accuracy
                domain_pred = domain_logits.argmax(-1)
                domain_correct += (domain_pred == domain_labels).sum().item()
                domain_total += mel.shape[0]
                total_domain_loss += domain_loss.item()
            else:
                loss = task_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_task_loss += task_loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_task_loss = total_task_loss / max(num_batches, 1)
        avg_domain_loss = total_domain_loss / max(num_batches, 1)
        domain_acc = domain_correct / max(domain_total, 1)

        # Log and checkpoint every N epochs
        if (epoch + 1) % config.log_every_epochs == 0 or epoch == config.epochs - 1:
            # Per-domain evaluation
            domain_metrics = evaluate_by_domain(model, val_loader, config.device)

            logger.info(
                f"Epoch {epoch+1:3d}/{config.epochs} | "
                f"Loss: {avg_loss:.4f} (task={avg_task_loss:.4f}, dom={avg_domain_loss:.4f}) | "
                f"Domain acc: {domain_acc:.4f}"
            )
            logger.info(
                f"  AISHELL-3: Syl={domain_metrics['aishell3']['syllable_acc']:.4f}, "
                f"Tone={domain_metrics['aishell3']['tone_acc']:.4f} "
                f"({domain_metrics['aishell3']['total']} samples)"
            )
            logger.info(
                f"  TTS:       Syl={domain_metrics['tts']['syllable_acc']:.4f}, "
                f"Tone={domain_metrics['tts']['tone_acc']:.4f} "
                f"({domain_metrics['tts']['total']} samples)"
            )
            logger.info(
                f"  Combined:  Syl={domain_metrics['combined']['syllable_acc']:.4f}, "
                f"Tone={domain_metrics['combined']['tone_acc']:.4f}"
            )

            # Save checkpoint
            ckpt_path = config.checkpoint_dir / f"checkpoint_epoch{epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "domain_metrics": domain_metrics,
                "domain_lambda": domain_lambda,
                "native_ratio": native_ratio,
            }, ckpt_path)

            # Save best model (based on combined accuracy)
            val_combined = (
                domain_metrics["combined"]["syllable_acc"] +
                domain_metrics["combined"]["tone_acc"]
            ) / 2
            if val_combined > best_val_acc:
                best_val_acc = val_combined
                best_path = config.checkpoint_dir / "best_model.pt"
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "domain_metrics": domain_metrics,
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

    # Pitch fusion
    parser.add_argument(
        "--use-pitch", action="store_true",
        help="Enable F0 pitch fusion for tone classification"
    )

    # Domain adversarial training
    parser.add_argument(
        "--domain-adapt", action="store_true",
        help="Enable domain adversarial training (DANN) with TTS data"
    )
    parser.add_argument(
        "--tts-dir", type=Path, default=None,
        help="Directory containing TTS data for domain adaptation (required if --domain-adapt)"
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
        domain_adapt=args.domain_adapt,
        tts_dir=args.tts_dir,
    )

    # Validate domain adaptation args
    if config.domain_adapt and config.tts_dir is None:
        print("Error: --tts-dir is required when using --domain-adapt")
        return

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
        use_pitch=args.use_pitch,
        use_domain_adversarial=config.domain_adapt,
    )
    model = SyllablePredictorV4(model_config).to(config.device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {total_params:,} params ({total_params * 4 / 1024 / 1024:.2f} MB)")
    logger.info(f"Device: {config.device}")

    # Create dataloaders (preload audio for non-mel sources).
    # If mel_cache is provided, no audio preload is needed for that source.
    logger.info(f"Context mode: {args.context_mode}")
    logger.info(f"Pitch fusion: {'enabled' if args.use_pitch else 'disabled'}")
    logger.info(f"Domain adaptation: {'enabled' if config.domain_adapt else 'disabled'}")

    preload = not mel_cache

    # Domain adaptation mode: need separate loaders for native and TTS
    if config.domain_adapt:
        from mandarin_grader.data.data_source import DataSourceRegistry

        # Load TTS data
        logger.info(f"Loading TTS data from: {config.tts_dir}")
        tts_source = DataSourceRegistry.get("tts")
        if not tts_source.is_available(config.tts_dir):
            logger.error(f"TTS data not found at {config.tts_dir}")
            return

        tts_sentences_all = tts_source.load(config.tts_dir, max_sentences=args.max_sentences)
        logger.info(f"Loaded {len(tts_sentences_all)} TTS sentences")

        # Split TTS data
        np.random.seed(42)
        tts_indices = np.random.permutation(len(tts_sentences_all))
        tts_split_idx = int(len(tts_sentences_all) * 0.8)
        tts_train = [tts_sentences_all[i] for i in tts_indices[:tts_split_idx]]
        tts_val = [tts_sentences_all[i] for i in tts_indices[tts_split_idx:]]
        logger.info(f"TTS split: Train={len(tts_train)}, Val={len(tts_val)}")

        # Create mixed domain dataloaders
        train_loader, mixed_dataset = create_mixed_domain_dataloader(
            native_sentences=train_sentences,
            tts_sentences=tts_train,
            batch_size=config.batch_size,
            augment=True,
            context_mode=args.context_mode,
            preload=preload,
            logger=logger,
            native_mel_cache=mel_cache,
            use_pitch=args.use_pitch,
        )

        # Validation loader also needs domain labels
        val_loader, val_mixed_dataset = create_mixed_domain_dataloader(
            native_sentences=val_sentences,
            tts_sentences=tts_val,
            batch_size=config.batch_size,
            augment=False,
            context_mode=args.context_mode,
            preload=preload,
            logger=logger,
            native_mel_cache=mel_cache,
            use_pitch=args.use_pitch,
            initial_native_ratio=0.5,  # Balanced for validation
        )
        logger.info(f"Mixed domain batches: Train={len(train_loader)}, Val={len(val_loader)}")
    else:
        train_loader = create_dataloader(train_sentences, config.batch_size, shuffle=True, augment=True, context_mode=args.context_mode, preload=preload, logger=logger, mel_cache=mel_cache, use_pitch=args.use_pitch)
        val_loader = create_dataloader(val_sentences, config.batch_size, shuffle=False, augment=False, context_mode=args.context_mode, preload=preload, logger=logger, mel_cache=mel_cache, use_pitch=args.use_pitch)
        mixed_dataset = None
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
    if config.domain_adapt:
        best_acc = train_domain_adapt(
            model, train_loader, mixed_dataset, val_loader, config, logger, start_epoch
        )

        # Final evaluation with per-domain breakdown
        logger.info("=" * 60)
        logger.info("FINAL EVALUATION (Domain Adaptation)")
        logger.info("=" * 60)

        domain_metrics = evaluate_by_domain(model, val_loader, config.device)

        logger.info(
            f"AISHELL-3: Syl={domain_metrics['aishell3']['syllable_acc']:.4f}, "
            f"Tone={domain_metrics['aishell3']['tone_acc']:.4f} "
            f"({domain_metrics['aishell3']['total']} samples)"
        )
        logger.info(
            f"TTS:       Syl={domain_metrics['tts']['syllable_acc']:.4f}, "
            f"Tone={domain_metrics['tts']['tone_acc']:.4f} "
            f"({domain_metrics['tts']['total']} samples)"
        )
        logger.info(
            f"Combined:  Syl={domain_metrics['combined']['syllable_acc']:.4f}, "
            f"Tone={domain_metrics['combined']['tone_acc']:.4f}"
        )

        # Save final report
        report = {
            "config": {
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "lr": config.learning_rate,
                "domain_adapt": True,
            },
            "model_params": total_params,
            "domain_metrics": domain_metrics,
            "best_val_combined": best_acc,
        }
    else:
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
