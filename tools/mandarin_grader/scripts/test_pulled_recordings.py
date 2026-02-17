#!/usr/bin/env python3
"""Test pulled recordings with the V4 model."""
import json
import sys
import io
from pathlib import Path

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch
import librosa

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mandarin_grader.model.syllable_predictor_v4 import (
    SyllablePredictorV4,
    SyllablePredictorConfigV4,
    SyllableVocab,
)

# Paths
RECORDINGS_DIR = Path(__file__).parent.parent / "pulled_recordings"
CHECKPOINT = Path(__file__).parent.parent / "checkpoints_v4_20M_aug" / "best_model.pt"
DATASET = Path(__file__).parent.parent.parent.parent / "apps/mobile_flutter/assets/datasets/sentences.zh.json"


def load_model():
    """Load the V4 model."""
    print(f"Loading model from {CHECKPOINT}")

    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]

    # Infer config
    d_model = state_dict["audio_type_embed"].shape[-1]
    n_layers = len(set(k.split(".")[1] for k in state_dict if k.startswith("transformer_layers.")))

    config = SyllablePredictorConfigV4(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=6,  # From training log
        dim_feedforward=state_dict["transformer_layers.0.linear1.weight"].shape[0],
        dropout=0.0,
    )

    model = SyllablePredictorV4(config)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"  d_model={d_model}, n_layers={n_layers}, n_heads=6")
    return model, config


def load_dataset():
    """Load the sentences dataset."""
    with open(DATASET, encoding="utf-8") as f:
        data = json.load(f)

    # Build id -> item map
    items = {item["id"]: item for item in data["items"]}
    return items


def extract_mel(wav_path: Path, config: SyllablePredictorConfigV4):
    """Extract mel spectrogram from wav file."""
    # Load audio
    y, sr = librosa.load(wav_path, sr=config.sample_rate)
    print(f"  Audio: {len(y)} samples, {len(y)/sr:.2f}s")

    # Extract mel
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=config.n_mels,
        hop_length=config.hop_length,
        win_length=config.win_length,
        n_fft=512,
        fmin=0,
        fmax=8000,
    )

    # Log mel
    mel = np.log(mel + 1e-9)
    print(f"  Mel: {mel.shape}")

    return mel


def score_recording(model, config, mel, pinyin_syllables, vocab):
    """Score a recording using the model."""
    device = next(model.parameters()).device

    # Prepare mel tensor - pad/truncate to max_audio_frames
    mel_frames = mel.shape[1]
    target_frames = config.max_audio_frames

    if mel_frames < target_frames:
        mel_padded = np.pad(mel, ((0, 0), (0, target_frames - mel_frames)))
    else:
        mel_padded = mel[:, :target_frames]

    mel_tensor = torch.from_numpy(mel_padded).float().unsqueeze(0).to(device)

    print(f"\n  Scoring {len(pinyin_syllables)} syllables: {' '.join(pinyin_syllables)}")

    scores = []
    for i, target_syl in enumerate(pinyin_syllables):
        # Position mode: [BOS, 2 + syllable_idx]
        pinyin_ids = torch.tensor([[1, 2 + i]], dtype=torch.long, device=device)

        # Create masks
        audio_mask = torch.zeros(1, target_frames, dtype=torch.bool, device=device)
        if mel_frames < target_frames:
            audio_mask[:, mel_frames:] = True

        pinyin_mask = torch.zeros(1, 2, dtype=torch.bool, device=device)

        # Run inference
        with torch.no_grad():
            syllable_logits, tone_logits = model(mel_tensor, pinyin_ids, audio_mask, pinyin_mask)

        # Get probability for target
        probs = torch.softmax(syllable_logits, dim=-1)
        target_id = vocab.encode(target_syl)

        if target_id >= 0 and target_id < probs.shape[1]:
            prob = probs[0, target_id].item()
        else:
            prob = 0.0
            print(f"    Warning: Unknown syllable '{target_syl}'")

        # Get top prediction
        top_id = syllable_logits.argmax(dim=-1).item()
        top_prob = probs[0, top_id].item()
        top_syl = vocab.decode(top_id) if top_id < len(vocab) else f"ID_{top_id}"

        print(f"    Syllable {i}: target='{target_syl}' prob={prob:.4f}, top='{top_syl}' ({top_prob:.4f})")
        scores.append(prob)

    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"  Average score: {avg_score:.4f}")
    return scores, avg_score


def main():
    print("=" * 60)
    print("Testing Pulled Recordings with V4 Model")
    print("=" * 60)

    # Load model and dataset
    model, config = load_model()
    dataset = load_dataset()
    vocab = SyllableVocab()

    print(f"\nVocab size: {len(vocab)}")

    # Find recordings
    recordings = sorted(RECORDINGS_DIR.glob("*.wav"))
    print(f"\nFound {len(recordings)} recordings")

    for wav_path in recordings:
        print(f"\n{'='*60}")
        print(f"Recording: {wav_path.name}")
        print("=" * 60)

        # Get text sequence ID
        ts_id = wav_path.stem  # e.g., "ts_000001"

        if ts_id not in dataset:
            print(f"  Warning: {ts_id} not found in dataset")
            continue

        item = dataset[ts_id]
        text = item["text"]
        romanization = item["romanization"]
        gloss = item.get("gloss", {})

        print(f"  Text: {text}")
        print(f"  Pinyin: {romanization}")
        gloss_en = gloss.get('en', '') if isinstance(gloss, dict) else str(gloss)
        print(f"  Gloss: {gloss_en}")

        # Parse pinyin to syllables
        syllables = romanization.strip().split()

        # Extract mel
        mel = extract_mel(wav_path, config)

        # Score
        scores, avg = score_recording(model, config, mel, syllables, vocab)

        print(f"\n  RESULT: {avg*100:.1f}%")


if __name__ == "__main__":
    main()
