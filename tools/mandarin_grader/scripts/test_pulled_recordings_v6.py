#!/usr/bin/env python3
"""Test pulled recordings with the V6 model."""
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

from mandarin_grader.model.syllable_predictor_v6 import (
    SyllablePredictorV6,
    SyllablePredictorConfigV6,
    SyllableVocab,
)
from mandarin_grader.model.syllable_predictor_v4 import (
    extract_mel_spectrogram,
    SyllablePredictorConfigV4,
)
from mandarin_grader.data.lexicon import _remove_tone_marks

# Paths
RECORDINGS_DIR = Path(__file__).parent.parent / "pulled_recordings"
CHECKPOINT = Path(__file__).parent.parent / "checkpoints_v6_10s_28syl_lr3e4" / "best_model.pt"
DATASET = Path(__file__).parent.parent.parent.parent / "apps/mobile_flutter/assets/datasets/sentences.zh.json"


def load_model():
    """Load the V6 model."""
    print(f"Loading model from {CHECKPOINT}")

    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]

    # Infer config from state_dict
    d_model = state_dict["audio_type_embed"].shape[-1]
    n_layers = len(set(k.split(".")[1] for k in state_dict if k.startswith("transformer_layers.")))
    dim_ff = state_dict["transformer_layers.0.linear1.weight"].shape[0]

    print(f"  d_model={d_model}, n_layers={n_layers}, dim_ff={dim_ff}")

    config = SyllablePredictorConfigV6(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=6,
        dim_feedforward=dim_ff,
        max_audio_frames=1000,  # 10s
        max_positions=28,
    )

    model = SyllablePredictorV6(config)
    model.load_state_dict(state_dict)
    model.eval()

    return model, config


def load_dataset():
    """Load the sentences dataset."""
    with open(DATASET, encoding="utf-8") as f:
        data = json.load(f)

    items = {item["id"]: item for item in data["items"]}
    return items


def extract_mel(wav_path: Path, config: SyllablePredictorConfigV6):
    """Extract mel spectrogram from wav file.

    IMPORTANT: Must use exact same preprocessing as training.
    Training uses extract_mel_spectrogram from syllable_predictor_v4.py which:
    - Normalizes audio to [-1, 1]
    - Uses n_fft = win_length = 400
    - Uses custom numpy FFT and mel filterbanks
    - Uses np.log(mel + 1e-9)
    """
    y, sr = librosa.load(wav_path, sr=config.sample_rate)
    max_amp = np.abs(y).max()
    rms = np.sqrt(np.mean(y**2))

    print(f"  Audio: {len(y)/sr:.2f}s, max_amp={max_amp:.4f}, rms={rms:.4f}")

    if max_amp < 0.1:
        print(f"  [WARNING] Audio is nearly SILENT - expect poor results!")
    elif max_amp < 0.3:
        print(f"  [WARNING] Audio volume is LOW")

    # Use exact same preprocessing as training (from syllable_predictor_v4.py)
    mel_config = SyllablePredictorConfigV4()  # Has same mel params as V6
    mel = extract_mel_spectrogram(y.astype(np.float32), mel_config)

    return mel


def score_recording(model, config, mel, pinyin_syllables, vocab):
    """Score a recording using the V6 model."""
    device = next(model.parameters()).device

    # Pad to 1000 frames (V6 uses 10s max)
    mel_frames = mel.shape[1]
    target_frames = config.max_audio_frames

    if mel_frames < target_frames:
        mel_padded = np.pad(mel, ((0, 0), (0, target_frames - mel_frames)))
    else:
        mel_padded = mel[:, :target_frames]

    mel_tensor = torch.from_numpy(mel_padded).float().unsqueeze(0).to(device)

    # Audio mask: True for padded frames
    audio_mask = torch.zeros(1, target_frames, dtype=torch.bool, device=device)
    if mel_frames < target_frames:
        audio_mask[:, mel_frames:] = True

    print(f"\n  Scoring {len(pinyin_syllables)} syllables: {' '.join(pinyin_syllables)}")

    scores = []
    for i, target_syl_raw in enumerate(pinyin_syllables):
        # Strip tone marks for vocab lookup (vocab uses base pinyin like "ni", not "nǐ")
        target_syl = _remove_tone_marks(target_syl_raw)
        # V6: position is 0-based index
        position = torch.tensor([[i]], dtype=torch.long, device=device)

        with torch.no_grad():
            syllable_logits, tone_logits = model(mel_tensor, position, audio_mask)

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

        correct = "[OK]" if target_syl == top_syl else ""
        print(f"    [{i}] target={target_syl}({prob:.3f}) top={top_syl}({top_prob:.3f}) {correct}")
        scores.append(prob)

    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"  Average score: {avg_score:.4f}")
    return scores, avg_score


def main():
    print("=" * 60)
    print("Testing Pulled Recordings with V6 Model")
    print("=" * 60)

    model, config = load_model()
    dataset = load_dataset()
    vocab = SyllableVocab()

    print(f"\nVocab size: {len(vocab)}")

    recordings = sorted(RECORDINGS_DIR.glob("*.wav"))
    print(f"\nFound {len(recordings)} recordings")

    results = []
    for wav_path in recordings:
        print(f"\n{'='*60}")
        print(f"Recording: {wav_path.name}")
        print("=" * 60)

        ts_id = wav_path.stem

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

        syllables = romanization.strip().split()

        mel = extract_mel(wav_path, config)
        scores, avg = score_recording(model, config, mel, syllables, vocab)

        print(f"\n  RESULT: {avg*100:.1f}%")
        results.append((wav_path.name, avg))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    for name, avg in results:
        print(f"  {name}: {avg*100:.1f}%")


if __name__ == "__main__":
    main()
