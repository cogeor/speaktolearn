#!/usr/bin/env python3
"""Test pulled recordings with the deployed V6 ONNX model.

Reports accuracy both without CMVN and with CMVN to quantify the CMVN fix.
"""
import json
import sys
import io
from pathlib import Path

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import librosa
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent.parent))

from mandarin_grader.model.syllable_predictor_v6 import SyllableVocab
from mandarin_grader.model.syllable_predictor_v4 import (
    extract_mel_spectrogram,
    SyllablePredictorConfigV4,
)
from mandarin_grader.data.lexicon import _remove_tone_marks

# Paths
RECORDINGS_DIR = Path(__file__).parent.parent / "pulled_recordings"
ONNX_MODEL = Path(__file__).parent.parent.parent.parent / "apps/mobile_flutter/assets/models/model_v6.onnx"
DATASET = Path(__file__).parent.parent.parent.parent / "apps/mobile_flutter/assets/datasets/sentences.zh.json"

MAX_AUDIO_FRAMES = 1000
SAMPLE_RATE = 16000
PAD_VALUE_CMVN = -5.0
PAD_VALUE_RAW = 0.0


def load_session() -> ort.InferenceSession:
    """Load the ONNX model from the deployed asset path."""
    print(f"Loading ONNX model from {ONNX_MODEL}")
    if not ONNX_MODEL.exists():
        raise FileNotFoundError(f"ONNX model not found: {ONNX_MODEL}")
    session = ort.InferenceSession(str(ONNX_MODEL), providers=["CPUExecutionProvider"])
    print(f"  Inputs:  {[i.name for i in session.get_inputs()]}")
    print(f"  Outputs: {[o.name for o in session.get_outputs()]}")
    return session


def load_dataset():
    """Load the sentences dataset."""
    with open(DATASET, encoding="utf-8") as f:
        data = json.load(f)
    items = {item["id"]: item for item in data["items"]}
    return items


def extract_mel(wav_path: Path) -> np.ndarray:
    """Extract log-mel spectrogram using the same extractor as V6 training.

    Uses extract_mel_spectrogram from syllable_predictor_v4.py which:
    - Normalizes audio to [-1, 1]
    - Uses n_fft = win_length = 400
    - Uses custom numpy FFT and mel filterbanks
    - Applies log(mel + 1e-9)

    Returns array of shape [80, T].
    """
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
    max_amp = np.abs(y).max()
    rms = np.sqrt(np.mean(y**2))
    print(f"  Audio: {len(y)/sr:.2f}s, max_amp={max_amp:.4f}, rms={rms:.4f}")

    if max_amp < 0.1:
        print("  [WARNING] Audio is nearly SILENT - expect poor results!")
    elif max_amp < 0.3:
        print("  [WARNING] Audio volume is LOW")

    mel_config = SyllablePredictorConfigV4()
    mel = extract_mel_spectrogram(y.astype(np.float32), mel_config)
    return mel


def prepare_inputs_raw(mel: np.ndarray, position: int):
    """Prepare ONNX inputs using raw mel (no CMVN), padded with zeros."""
    mel_frames = mel.shape[1]

    if mel_frames < MAX_AUDIO_FRAMES:
        mel_padded = np.pad(mel, ((0, 0), (0, MAX_AUDIO_FRAMES - mel_frames)),
                            constant_values=PAD_VALUE_RAW)
    else:
        mel_padded = mel[:, :MAX_AUDIO_FRAMES]
        mel_frames = MAX_AUDIO_FRAMES

    mel_input = mel_padded.astype(np.float32)[np.newaxis, :, :]  # [1, 80, 1000]
    position_input = np.array([[position]], dtype=np.int64)        # [1, 1]
    audio_mask = np.zeros((1, MAX_AUDIO_FRAMES), dtype=bool)
    if mel_frames < MAX_AUDIO_FRAMES:
        audio_mask[0, mel_frames:] = True

    return mel_input, position_input, audio_mask


def prepare_inputs_cmvn(mel: np.ndarray, position: int):
    """Prepare ONNX inputs with per-utterance CMVN applied, padded with -5.0."""
    mel_frames = mel.shape[1]

    # Per-utterance CMVN: normalize per mel bin across time frames
    mel_mean = np.mean(mel, axis=1, keepdims=True)   # [80, 1]
    mel_std = np.std(mel, axis=1, keepdims=True)     # [80, 1]
    mel_normalized = (mel - mel_mean) / (mel_std + 1e-5)

    if mel_frames < MAX_AUDIO_FRAMES:
        mel_padded = np.pad(mel_normalized, ((0, 0), (0, MAX_AUDIO_FRAMES - mel_frames)),
                            constant_values=PAD_VALUE_CMVN)
    else:
        mel_padded = mel_normalized[:, :MAX_AUDIO_FRAMES]
        mel_frames = MAX_AUDIO_FRAMES

    mel_input = mel_padded.astype(np.float32)[np.newaxis, :, :]  # [1, 80, 1000]
    position_input = np.array([[position]], dtype=np.int64)        # [1, 1]
    audio_mask = np.zeros((1, MAX_AUDIO_FRAMES), dtype=bool)
    if mel_frames < MAX_AUDIO_FRAMES:
        audio_mask[0, mel_frames:] = True

    return mel_input, position_input, audio_mask


def run_inference(session: ort.InferenceSession, mel_input, position_input, audio_mask):
    """Run ONNX inference and return (syllable_logits, tone_logits)."""
    outputs = session.run(
        ["syllable_logits", "tone_logits"],
        {
            "mel": mel_input,
            "position": position_input,
            "audio_mask": audio_mask,
        },
    )
    return outputs[0], outputs[1]  # [1, 532], [1, 5]


def score_recording(session: ort.InferenceSession, mel: np.ndarray,
                    pinyin_syllables: list, vocab: SyllableVocab):
    """Score a recording using both raw and CMVN paths.

    Returns:
        scores_raw: list of target-syllable probabilities (no CMVN)
        scores_cmvn: list of target-syllable probabilities (with CMVN)
    """
    n_to_score = len(pinyin_syllables)
    print(f"\n  Scoring {n_to_score} syllables: {' '.join(pinyin_syllables)}")
    print(f"  {'Pos':<4}  {'Target':<10}  {'Raw prob':>9}  {'Raw top':<12}  {'CMVN prob':>10}  {'CMVN top':<12}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*9}  {'-'*12}  {'-'*10}  {'-'*12}")

    scores_raw = []
    scores_cmvn = []

    for i in range(n_to_score):
        target_syl_raw = pinyin_syllables[i]
        target_syl = _remove_tone_marks(target_syl_raw)
        target_id = vocab.encode(target_syl)

        # --- Raw path ---
        mel_in_r, pos_in, mask_r = prepare_inputs_raw(mel, i)
        syl_logits_r, _ = run_inference(session, mel_in_r, pos_in, mask_r)
        probs_r = np.exp(syl_logits_r[0]) / np.sum(np.exp(syl_logits_r[0]))  # softmax
        prob_r = float(probs_r[target_id]) if 0 <= target_id < len(probs_r) else 0.0
        top_id_r = int(np.argmax(syl_logits_r[0]))
        top_syl_r = vocab.decode(top_id_r) if top_id_r < len(vocab) else f"ID_{top_id_r}"

        # --- CMVN path ---
        mel_in_c, _, mask_c = prepare_inputs_cmvn(mel, i)
        syl_logits_c, _ = run_inference(session, mel_in_c, pos_in, mask_c)
        probs_c = np.exp(syl_logits_c[0]) / np.sum(np.exp(syl_logits_c[0]))
        prob_c = float(probs_c[target_id]) if 0 <= target_id < len(probs_c) else 0.0
        top_id_c = int(np.argmax(syl_logits_c[0]))
        top_syl_c = vocab.decode(top_id_c) if top_id_c < len(vocab) else f"ID_{top_id_c}"

        ok_r = "[OK]" if target_syl == top_syl_r else "    "
        ok_c = "[OK]" if target_syl == top_syl_c else "    "
        print(f"  {i:<4}  {target_syl:<10}  {prob_r:>9.3f}  {top_syl_r:<10}{ok_r}  {prob_c:>10.3f}  {top_syl_c:<10}{ok_c}")

        scores_raw.append(prob_r)
        scores_cmvn.append(prob_c)

    avg_raw = sum(scores_raw) / len(scores_raw) if scores_raw else 0.0
    avg_cmvn = sum(scores_cmvn) / len(scores_cmvn) if scores_cmvn else 0.0
    print(f"\n  Average score (raw):  {avg_raw:.4f}")
    print(f"  Average score (CMVN): {avg_cmvn:.4f}")
    return scores_raw, scores_cmvn


def main():
    print("=" * 70)
    print("Testing Pulled Recordings with Deployed V6 ONNX Model")
    print("Dual-path: raw mel vs. CMVN-normalized mel")
    print("=" * 70)

    session = load_session()
    dataset = load_dataset()
    vocab = SyllableVocab()

    print(f"\nVocab size: {len(vocab)}")

    recordings = sorted(RECORDINGS_DIR.glob("*.wav"))
    print(f"\nFound {len(recordings)} recordings")

    results = []
    for wav_path in recordings:
        print(f"\n{'='*70}")
        print(f"Recording: {wav_path.name}")
        print("=" * 70)

        ts_id = wav_path.stem

        if ts_id not in dataset:
            print(f"  Warning: {ts_id} not found in dataset")
            continue

        item = dataset[ts_id]
        text = item["text"]
        romanization = item["romanization"]
        gloss = item.get("gloss", {})
        gloss_en = gloss.get("en", "") if isinstance(gloss, dict) else str(gloss)

        print(f"  Text:    {text}")
        print(f"  Pinyin:  {romanization}")
        print(f"  Gloss:   {gloss_en}")

        syllables = romanization.strip().split()
        mel = extract_mel(wav_path)

        scores_raw, scores_cmvn = score_recording(session, mel, syllables, vocab)
        avg_raw = sum(scores_raw) / len(scores_raw) if scores_raw else 0.0
        avg_cmvn = sum(scores_cmvn) / len(scores_cmvn) if scores_cmvn else 0.0

        print(f"\n  RESULT (raw):  {avg_raw*100:.1f}%")
        print(f"  RESULT (CMVN): {avg_cmvn*100:.1f}%")
        results.append((wav_path.name, avg_raw, avg_cmvn))

    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Recording':<30}  {'Raw':>6}  {'CMVN':>6}  {'Delta':>6}")
    print(f"  {'-'*30}  {'-'*6}  {'-'*6}  {'-'*6}")
    for name, avg_raw, avg_cmvn in results:
        delta = avg_cmvn - avg_raw
        print(f"  {name:<30}  {avg_raw*100:>5.1f}%  {avg_cmvn*100:>5.1f}%  {delta*100:>+5.1f}%")

    if results:
        overall_raw = sum(r[1] for r in results) / len(results)
        overall_cmvn = sum(r[2] for r in results) / len(results)
        overall_delta = overall_cmvn - overall_raw
        print(f"  {'OVERALL':<30}  {overall_raw*100:>5.1f}%  {overall_cmvn*100:>5.1f}%  {overall_delta*100:>+5.1f}%")
        print()
        if overall_delta > 0:
            print(f"  CMVN improves accuracy by {overall_delta*100:.1f}pp")
        elif overall_delta < 0:
            print(f"  Raw (no CMVN) is better by {-overall_delta*100:.1f}pp")
        else:
            print("  CMVN has no effect on overall accuracy")


if __name__ == "__main__":
    main()
