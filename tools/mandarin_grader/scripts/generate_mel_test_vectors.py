#!/usr/bin/env python3
"""Generate mel spectrogram test vectors for Dart validation.

This script generates test audio signals and their corresponding mel spectrograms
using the same implementation as syllable_predictor_v4.py. The outputs are saved
as JSON for cross-validation with the Dart implementation.
"""

import json
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path to import from mandarin_grader
sys.path.insert(0, str(Path(__file__).parent.parent))

from mandarin_grader.model.syllable_predictor_v4 import (
    _create_mel_filterbank,
)


class MelConfig:
    """Configuration for mel extraction."""

    n_mels = 80
    sample_rate = 16000
    hop_length = 160
    win_length = 400
    n_fft = 400


def extract_mel_spectrogram(audio: np.ndarray, config: MelConfig) -> np.ndarray:
    """Extract mel spectrogram matching syllable_predictor_v4.py implementation.

    Args:
        audio: Audio samples (float32, normalized to [-1, 1])
        config: Config with mel parameters

    Returns:
        Mel spectrogram [n_mels, time]
    """
    # Normalize audio amplitude to [-1, 1] for consistent mel features
    max_abs = np.max(np.abs(audio))
    if max_abs > 1e-6:
        audio = audio / max_abs

    n_fft = config.win_length
    hop_length = config.hop_length
    n_mels = config.n_mels
    sr = config.sample_rate

    # Compute STFT
    n_frames = 1 + (len(audio) - n_fft) // hop_length
    if n_frames < 1:
        audio = np.pad(audio, (0, n_fft - len(audio) + hop_length))
        n_frames = 1

    window = np.hanning(n_fft)

    spec = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start : start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        windowed = frame * window
        fft = np.fft.rfft(windowed)
        spec[:, i] = np.abs(fft) ** 2

    mel_basis = _create_mel_filterbank(sr, n_fft, n_mels)
    mel_spec = np.dot(mel_basis, spec)

    mel_spec = np.log(mel_spec + 1e-9)

    return mel_spec.astype(np.float32)


def generate_test_signals(sr: int = 16000) -> dict[str, np.ndarray]:
    """Generate test audio signals.

    Args:
        sr: Sample rate

    Returns:
        Dictionary of test signals
    """
    duration = 1.0  # 1 second
    n_samples = int(duration * sr)
    t = np.arange(n_samples) / sr

    signals = {}

    # Silence
    signals["silence"] = np.zeros(n_samples, dtype=np.float32)

    # Pure tones at different frequencies
    signals["sine_100hz"] = np.sin(2 * np.pi * 100 * t).astype(np.float32)
    signals["sine_1000hz"] = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    signals["sine_4000hz"] = np.sin(2 * np.pi * 4000 * t).astype(np.float32)

    # White noise
    np.random.seed(42)
    signals["white_noise"] = np.random.randn(n_samples).astype(np.float32)

    # Short audio (less than one frame)
    signals["short_audio"] = np.sin(2 * np.pi * 440 * np.arange(200) / sr).astype(
        np.float32
    )

    return signals


def main():
    """Generate test vectors and save to JSON."""
    config = MelConfig()

    # Generate test signals
    print("Generating test signals...")
    signals = generate_test_signals(config.sample_rate)

    # Extract mel spectrograms
    print("Extracting mel spectrograms...")
    results = {}
    for name, audio in signals.items():
        mel_spec = extract_mel_spectrogram(audio, config)
        results[name] = {
            "audio": audio.tolist(),
            "mel_spectrogram": mel_spec.tolist(),
            "shape": list(mel_spec.shape),
        }
        print(
            f"  {name}: audio={len(audio)} samples, mel={mel_spec.shape[0]}x{mel_spec.shape[1]}"
        )

    # Save metadata
    results["_metadata"] = {
        "n_mels": config.n_mels,
        "sample_rate": config.sample_rate,
        "hop_length": config.hop_length,
        "win_length": config.win_length,
        "n_fft": config.n_fft,
    }

    # Save to JSON
    output_path = Path(__file__).parent.parent / "test_vectors" / "mel_test_vectors.json"
    output_path.parent.mkdir(exist_ok=True)

    print(f"\nSaving to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Done!")

    # Print some statistics
    print("\nTest vector statistics:")
    for name, data in results.items():
        if name == "_metadata":
            continue
        mel_spec = np.array(data["mel_spectrogram"])
        print(f"  {name}:")
        print(f"    Shape: {data['shape']}")
        print(f"    Min: {mel_spec.min():.4f}")
        print(f"    Max: {mel_spec.max():.4f}")
        print(f"    Mean: {mel_spec.mean():.4f}")


if __name__ == "__main__":
    main()
