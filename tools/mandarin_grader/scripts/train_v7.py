#!/usr/bin/env python3
"""Training script for SyllablePredictorV7 - CTC-based Architecture.

Usage:
    python train_v7.py --data-source aishell3 --data-dir datasets/aishell3_tar --epochs 50
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Silence floor value in mel domain: log(epsilon) where epsilon=1e-9
# This must match the value used in extract_mel_spectrogram
MEL_SILENCE_FLOOR = np.log(1e-9)  # ≈ -20.72

SYNTHETIC_DIR = Path(__file__).parent.parent / "data" / "synthetic_train"
DEFAULT_CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints_v7"


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

    logger = logging.getLogger("train_v7")
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
    batch_size: int = 32  # Smaller than V6 due to full sequence processing
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    log_every_epochs: int = 1
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
    device: str = "cuda"


def edit_distance(ref: list, hyp: list) -> int:
    """Compute edit distance (Levenshtein distance) between two sequences."""
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


def compute_error_rate(refs: list[list[int]], hyps: list[list[int]]) -> float:
    """Compute error rate (edit distance / reference length)."""
    total_errors = 0
    total_length = 0
    for ref, hyp in zip(refs, hyps):
        total_errors += edit_distance(ref, hyp)
        total_length += len(ref)
    if total_length == 0:
        return 0.0
    return total_errors / total_length


def load_training_data(
    sources: list[str],
    data_dirs: list[Path],
    logger,
    train_split: float = 0.8,
    max_sentences_per_source: int | None = None,
) -> tuple[list, list, dict[str, np.ndarray]]:
    """Load training data (same as V6)."""
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


def create_ctc_dataloader(
    sentences: list,
    batch_size: int,
    shuffle: bool,
    augment: bool,
    logger=None,
    mel_cache: dict | None = None,
    max_duration_s: float = 10.0,
    speed_variation: float = 0.1,
    random_padding: bool = True,
):
    """Create dataloader for CTC training.

    Unlike V6 dataloader which creates (mel, position, target) samples,
    V7 CTC dataloader creates (mel, syllable_seq, tone_seq) samples.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torch.nn.utils.rnn import pad_sequence
    from mandarin_grader.model.syllable_predictor_v4 import extract_mel_spectrogram, SyllablePredictorConfigV4
    from mandarin_grader.model.syllable_predictor_v7 import SyllableVocab
    from mandarin_grader.data.autoregressive_dataset import load_audio_wav, spec_augment
    from mandarin_grader.data.lexicon import _remove_tone_marks

    vocab = SyllableVocab()
    mel_config = SyllablePredictorConfigV4()

    class CTCDataset(Dataset):
        """Dataset for CTC training - each sample is a full sentence."""

        def __init__(self, sentences, augment=False):
            self.sentences = sentences
            self.augment = augment
            self.sample_rate = 16000
            self.max_samples = int(max_duration_s * self.sample_rate)
            self._mel_cache = mel_cache or {}

        def __len__(self):
            return len(self.sentences)

        def __getitem__(self, idx):
            sent = self.sentences[idx]
            key = str(sent.audio_path)

            mel = self._mel_cache.get(key)
            if mel is None:
                audio = load_audio_wav(sent.audio_path, self.sample_rate)

                # Speed variation augmentation (applied before mel extraction)
                if self.augment and speed_variation > 0:
                    factor = 1.0 + np.random.uniform(-speed_variation, speed_variation)
                    if abs(factor - 1.0) > 0.01:
                        new_length = int(len(audio) / factor)
                        if new_length > 1:
                            indices = np.linspace(0, len(audio) - 1, new_length)
                            audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

                # Truncate
                if len(audio) > self.max_samples:
                    audio = audio[:self.max_samples]

                mel = extract_mel_spectrogram(audio, mel_config)

            # spec_augment (applied after mel extraction)
            if self.augment:
                mel = spec_augment(mel.copy() if mel is self._mel_cache.get(key) else mel)
            elif mel is self._mel_cache.get(key):
                mel = mel.copy()

            # Target sequences (CTC blank at 0, so syllable IDs start at 1)
            # Vocab.encode returns ID starting at 2 (0=PAD, 1=BOS in old vocab)
            # For CTC: shift to 0=blank, 1+=syllables
            syllable_ids = []
            tone_ids = []
            for syl in sent.syllables:
                pinyin = _remove_tone_marks(syl.pinyin)
                syl_id = vocab.encode(pinyin)
                # Shift: old vocab has 0=PAD, 1=BOS, 2+=syls
                # CTC needs: 0=blank, 1+=syls
                # So subtract 1 (PAD and BOS become -1/0, which we remap)
                ctc_syl_id = max(1, syl_id - 1)  # Ensure >= 1 (blank=0)
                syllable_ids.append(ctc_syl_id)

                # Tones: 0-4 -> CTC blank=0, tones=1-5
                tone = syl.tone_surface
                tone_ids.append(tone + 1)  # Shift by 1 for CTC blank

            return {
                'mel': mel,
                'syllable_targets': syllable_ids,
                'tone_targets': tone_ids,
            }

    dataset = CTCDataset(sentences, augment=augment)

    if mel_cache and logger:
        logger.info(f"  Using {len(mel_cache)} precomputed mel entries")

    def collate_fn(batch):
        """Collate batch with variable-length padding for CTC."""
        mels = [torch.tensor(b['mel'], dtype=torch.float32) for b in batch]
        syl_targets = [torch.tensor(b['syllable_targets'], dtype=torch.long) for b in batch]
        tone_targets = [torch.tensor(b['tone_targets'], dtype=torch.long) for b in batch]

        # Get lengths before padding
        mel_lengths = torch.tensor([m.shape[1] for m in mels], dtype=torch.long)
        target_lengths = torch.tensor([len(s) for s in syl_targets], dtype=torch.long)

        # Pad mel to max length in batch: [n_mels, time] -> [batch, n_mels, max_time]
        # Use silence floor for padding, not zeros!
        # Zero in mel domain corresponds to high energy (≈ e^0 = 1)
        # Real silence produces mel values around log(1e-9) ≈ -20.72
        max_time = max(m.shape[1] for m in mels)
        n_mels = mels[0].shape[0]
        padded_mels = torch.full((len(batch), n_mels, max_time), MEL_SILENCE_FLOOR, dtype=torch.float32)
        audio_masks = torch.ones(len(batch), max_time, dtype=torch.bool)  # True = padded

        for i, m in enumerate(mels):
            t = m.shape[1]
            pad_total = max_time - t

            if random_padding and augment and pad_total > 0:
                # Random offset: distribute padding between start and end
                pad_start = np.random.randint(0, pad_total + 1)
            else:
                pad_start = 0

            padded_mels[i, :, pad_start:pad_start + t] = m
            audio_masks[i, pad_start:pad_start + t] = False  # False = real audio

        # Concatenate targets for CTCLoss (expects 1D tensor of all targets)
        syl_targets_flat = torch.cat(syl_targets)
        tone_targets_flat = torch.cat(tone_targets)

        return {
            'mel': padded_mels,
            'audio_mask': audio_masks,
            'mel_lengths': mel_lengths,
            'syllable_targets': syl_targets_flat,
            'tone_targets': tone_targets_flat,
            'target_lengths': target_lengths,
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0,
    )


def evaluate(model, dataloader, device: str, logger=None) -> tuple[float, float, float, float]:
    """Evaluate model with CTC decoding and compute error rates.

    Returns:
        (syl_accuracy, tone_accuracy, syllable_error_rate, tone_error_rate)
    """
    import torch

    model.eval()
    all_syl_refs, all_syl_hyps = [], []
    all_tone_refs, all_tone_hyps = [], []

    with torch.no_grad():
        for batch in dataloader:
            mel = batch['mel'].to(device)
            audio_mask = batch['audio_mask'].to(device)

            syl_logits, tone_logits = model(mel, audio_mask)

            # CTC decode
            syl_decoded = model.ctc_decoder.greedy_decode(syl_logits)
            tone_decoded = model.ctc_decoder.greedy_decode(tone_logits)

            # Reconstruct target sequences per sample from flat targets
            target_lengths = batch['target_lengths'].tolist()
            syl_targets = batch['syllable_targets'].tolist()
            tone_targets = batch['tone_targets'].tolist()

            offset = 0
            for i, tlen in enumerate(target_lengths):
                ref_syl = syl_targets[offset:offset + tlen]
                ref_tone = tone_targets[offset:offset + tlen]

                all_syl_refs.append(ref_syl)
                all_syl_hyps.append(syl_decoded[i])
                all_tone_refs.append(ref_tone)
                all_tone_hyps.append(tone_decoded[i])

                offset += tlen

    # Compute error rates
    syl_er = compute_error_rate(all_syl_refs, all_syl_hyps)
    tone_er = compute_error_rate(all_tone_refs, all_tone_hyps)

    # Compute exact match accuracy
    syl_exact = sum(r == h for r, h in zip(all_syl_refs, all_syl_hyps)) / len(all_syl_refs)
    tone_exact = sum(r == h for r, h in zip(all_tone_refs, all_tone_hyps)) / len(all_tone_refs)

    return syl_exact, tone_exact, syl_er, tone_er


def train(model, train_loader, val_loader, config: TrainingConfig, logger, start_epoch: int = 0):
    import torch
    import torch.nn.functional as F

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = config.epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    # CTC loss (blank=0)
    ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    best_combined = float('inf')  # Lower error rate is better

    logger.info(f"Starting training from epoch {start_epoch + 1} to {config.epochs}")
    logger.info(f"Warmup steps: {warmup_steps}")

    for epoch in range(start_epoch, config.epochs):
        model.train()
        total_loss, total_syl_loss, total_tone_loss = 0, 0, 0
        num_batches = 0
        epoch_start = time.time()

        for batch in train_loader:
            mel = batch['mel'].to(config.device)
            audio_mask = batch['audio_mask'].to(config.device)
            mel_lengths = batch['mel_lengths'].to(config.device)
            syl_targets = batch['syllable_targets'].to(config.device)
            tone_targets = batch['tone_targets'].to(config.device)
            target_lengths = batch['target_lengths'].to(config.device)

            optimizer.zero_grad()

            # Forward
            syl_logits, tone_logits = model(mel, audio_mask)

            # CTC requires log_softmax and [T, B, C] format
            # syl_logits: [B, T, C] -> [T, B, C]
            syl_log_probs = F.log_softmax(syl_logits, dim=-1).transpose(0, 1)
            tone_log_probs = F.log_softmax(tone_logits, dim=-1).transpose(0, 1)

            # Input lengths after CNN downsampling
            input_lengths = model.get_input_lengths(mel_lengths)

            # CTC loss
            syl_loss = ctc_loss(syl_log_probs, syl_targets, input_lengths, target_lengths)
            tone_loss = ctc_loss(tone_log_probs, tone_targets, input_lengths, target_lengths)

            loss = 0.7 * syl_loss + 0.3 * tone_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_syl_loss += syl_loss.item()
            total_tone_loss += tone_loss.item()
            num_batches += 1

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / max(num_batches, 1)
        avg_syl_loss = total_syl_loss / max(num_batches, 1)
        avg_tone_loss = total_tone_loss / max(num_batches, 1)
        ms_per_batch = (epoch_time / num_batches) * 1000

        if (epoch + 1) % config.log_every_epochs == 0 or epoch == config.epochs - 1:
            train_syl_acc, train_tone_acc, train_syl_er, train_tone_er = evaluate(model, train_loader, config.device)
            val_syl_acc, val_tone_acc, val_syl_er, val_tone_er = evaluate(model, val_loader, config.device)

            logger.info(
                f"Epoch {epoch+1:3d}/{config.epochs} | "
                f"Loss: {avg_loss:.4f} (syl={avg_syl_loss:.4f}, tone={avg_tone_loss:.4f}) | "
                f"Train: {train_syl_acc:.4f}/{train_tone_acc:.4f} (SER/TER: {train_syl_er:.3f}/{train_tone_er:.3f}) | "
                f"Val: {val_syl_acc:.4f}/{val_tone_acc:.4f} (SER/TER: {val_syl_er:.3f}/{val_tone_er:.3f}) | "
                f"{ms_per_batch:.1f}ms/batch"
            )

            ckpt_path = config.checkpoint_dir / f"checkpoint_epoch{epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_syl_accuracy": val_syl_acc,
                "val_tone_accuracy": val_tone_acc,
                "val_syl_error_rate": val_syl_er,
                "val_tone_error_rate": val_tone_er,
            }, ckpt_path)

            # CTC model selection uses error rate (lower is better)
            # Unlike V6 which uses accuracy (higher is better)
            # Both metrics saved for compatibility with downstream tools
            val_combined = 0.7 * val_syl_er + 0.3 * val_tone_er
            if val_combined < best_combined:
                best_combined = val_combined
                best_path = config.checkpoint_dir / "best_model.pt"
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "val_syl_accuracy": val_syl_acc,
                    "val_tone_accuracy": val_tone_acc,
                    "val_syl_error_rate": val_syl_er,
                    "val_tone_error_rate": val_tone_er,
                }, best_path)
                logger.info(f"  -> New best model! Combined ER: {val_combined:.4f}")

    return best_combined


def main():
    import torch

    parser = argparse.ArgumentParser(description="Train SyllablePredictorV7 (CTC-based)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--log-every-epochs", type=int, default=1)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--data-source", type=str, default="synthetic")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--max-duration-s", type=float, default=10.0)

    parser.add_argument("--resume", type=str, default=None)

    # Model architecture (BiLSTM)
    parser.add_argument("--lstm-hidden", type=int, default=96,
                        help="LSTM hidden size (bidirectional doubles this)")
    parser.add_argument("--lstm-layers", type=int, default=2,
                        help="Number of BiLSTM layers")
    parser.add_argument("--cnn-downsample", type=int, default=4,
                        help="CNN downsampling factor: 4=25fps, 8=12fps")

    # Augmentation
    parser.add_argument("--speed-variation", type=float, default=0.1)
    parser.add_argument("--no-random-padding", action="store_true",
                        help="Disable random padding augmentation")

    args = parser.parse_args()

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        log_every_epochs=args.log_every_epochs,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )

    logger = setup_logging(config.checkpoint_dir)
    logger.info("=" * 60)
    logger.info("SyllablePredictorV7 Training (CTC-based)")
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

    from mandarin_grader.model.syllable_predictor_v7 import SyllablePredictorV7, SyllablePredictorConfigV7
    model_config = SyllablePredictorConfigV7(
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        max_audio_frames=int(args.max_duration_s * 100),
        cnn_downsample=args.cnn_downsample,
    )
    model = SyllablePredictorV7(model_config).to(config.device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {total_params:,} params ({total_params * 4 / 1024 / 1024:.2f} MB)")
    fps = 100 / args.cnn_downsample  # mel is 100fps (10ms hop)
    logger.info(f"Architecture: BiLSTM d_model={model_config.d_model}, lstm_layers={model_config.lstm_layers}, lstm_hidden={model_config.lstm_hidden}, downsample={args.cnn_downsample}x ({fps:.0f}fps)")
    logger.info(f"Device: {config.device}")

    random_padding = not args.no_random_padding
    padding_str = "random start/end" if random_padding else "end only"
    logger.info(f"Augmentation: speed=+/-{args.speed_variation*100:.0f}%, spec_augment=on, padding={padding_str}")

    train_loader = create_ctc_dataloader(
        train_sentences, config.batch_size, shuffle=True, augment=True,
        logger=logger, mel_cache=mel_cache,
        max_duration_s=args.max_duration_s,
        speed_variation=args.speed_variation,
        random_padding=random_padding,
    )
    val_loader = create_ctc_dataloader(
        val_sentences, config.batch_size, shuffle=False, augment=False,
        logger=logger, mel_cache=mel_cache,
        max_duration_s=args.max_duration_s,
        random_padding=False,  # Validation always pads at end
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

    best_er = train(model, train_loader, val_loader, config, logger, start_epoch)

    logger.info("=" * 60)
    logger.info(f"Training complete. Best combined error rate: {best_er:.4f}")


if __name__ == "__main__":
    main()
