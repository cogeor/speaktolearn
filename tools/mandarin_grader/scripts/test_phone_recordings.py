#!/usr/bin/env python3
"""Test phone recordings with the V4 model."""
import json
import sys
import io
from pathlib import Path

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch
import librosa

sys.path.insert(0, str(Path(__file__).parent.parent))

from mandarin_grader.model.syllable_predictor_v4 import (
    SyllablePredictorV4,
    SyllablePredictorConfigV4,
    SyllableVocab,
)

CHECKPOINT = Path(__file__).parent.parent / "checkpoints_v4_20M_aug" / "best_model.pt"
DATASET = Path(__file__).parent.parent.parent.parent / "apps/mobile_flutter/assets/datasets/sentences.zh.json"
RECORDINGS_DIR = Path(__file__).parent.parent / "pulled_recordings_phone"


def load_model():
    print(f"Loading model from {CHECKPOINT}")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]

    d_model = state_dict["audio_type_embed"].shape[-1]
    n_layers = len(set(k.split(".")[1] for k in state_dict if k.startswith("transformer_layers.")))

    config = SyllablePredictorConfigV4(
        d_model=d_model, n_layers=n_layers, n_heads=6,
        dim_feedforward=state_dict["transformer_layers.0.linear1.weight"].shape[0],
        dropout=0.0,
    )

    model = SyllablePredictorV4(config)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


def load_dataset():
    with open(DATASET, encoding="utf-8") as f:
        data = json.load(f)
    return {item["id"]: item for item in data["items"]}


def extract_mel(audio_path: Path, config):
    y, sr = librosa.load(audio_path, sr=config.sample_rate)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=config.n_mels, hop_length=config.hop_length,
        win_length=config.win_length, n_fft=512, fmin=0, fmax=8000,
    )
    return np.log(mel + 1e-9), len(y), sr


def score_recording(model, config, mel, pinyin_syllables, vocab):
    device = next(model.parameters()).device
    mel_frames = mel.shape[1]
    target_frames = config.max_audio_frames

    if mel_frames < target_frames:
        mel_padded = np.pad(mel, ((0, 0), (0, target_frames - mel_frames)))
    else:
        mel_padded = mel[:, :target_frames]

    mel_tensor = torch.from_numpy(mel_padded).float().unsqueeze(0).to(device)

    scores = []
    for i, target_syl in enumerate(pinyin_syllables):
        pinyin_ids = torch.tensor([[1, 2 + i]], dtype=torch.long, device=device)
        audio_mask = torch.zeros(1, target_frames, dtype=torch.bool, device=device)
        if mel_frames < target_frames:
            audio_mask[:, mel_frames:] = True
        pinyin_mask = torch.zeros(1, 2, dtype=torch.bool, device=device)

        with torch.no_grad():
            syllable_logits, _ = model(mel_tensor, pinyin_ids, audio_mask, pinyin_mask)

        probs = torch.softmax(syllable_logits, dim=-1)
        target_id = vocab.encode(target_syl)

        prob = probs[0, target_id].item() if 0 <= target_id < probs.shape[1] else 0.0
        top_id = syllable_logits.argmax(dim=-1).item()
        top_prob = probs[0, top_id].item()
        top_syl = vocab.decode(top_id) if top_id < len(vocab) else f"ID_{top_id}"

        print(f"    [{i}] '{target_syl}': {prob:.4f}  |  top: '{top_syl}' ({top_prob:.4f})")
        scores.append(prob)

    return scores, sum(scores) / len(scores) if scores else 0


def main():
    print("=" * 60)
    print("Testing Phone Recordings with V4 Model")
    print("=" * 60)

    model, config = load_model()
    dataset = load_dataset()
    vocab = SyllableVocab()

    # Check vocab format
    print(f"\nVocab samples: {[vocab.decode(i) for i in range(10)]}")

    recordings = sorted(RECORDINGS_DIR.glob("*.m4a"))
    print(f"\nFound {len(recordings)} recordings")

    for audio_path in recordings:
        ts_id = audio_path.stem
        item = dataset.get(ts_id)
        if not item:
            print(f"\nSkipping {ts_id} - not in dataset")
            continue

        print(f"\n{'='*60}")
        print(f"{ts_id}: {item['text']}")
        print(f"Pinyin: {item['romanization']}")
        print(f"Gloss: {item.get('gloss', {}).get('en', '')}")

        mel, samples, sr = extract_mel(audio_path, config)
        print(f"Audio: {samples/sr:.2f}s, Mel: {mel.shape}")

        syllables = item['romanization'].strip().split()
        scores, avg = score_recording(model, config, mel, syllables, vocab)
        print(f"AVERAGE: {avg*100:.1f}%")


if __name__ == "__main__":
    main()
