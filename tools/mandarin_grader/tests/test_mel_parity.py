"""Cross-language mel parity test: Python -> JSON fixture -> Dart.

This test generates a reference mel spectrogram for a known 440 Hz sine wave
and exports it to a JSON fixture that the Dart parity test can load and verify.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from mandarin_grader.model.syllable_predictor_v4 import (
    SyllablePredictorConfigV4,
    extract_mel_spectrogram,
)

FIXTURE_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "apps"
    / "mobile_flutter"
    / "test"
    / "fixtures"
    / "mel_parity_reference.json"
)


def generate_sine_wave(
    frequency_hz: float = 440.0,
    duration_s: float = 1.0,
    sample_rate: int = 16000,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate a sine wave at the given frequency."""
    n_samples = int(duration_s * sample_rate)
    t = np.arange(n_samples) / sample_rate
    return (amplitude * np.sin(2 * math.pi * frequency_hz * t)).astype(np.float32)


def build_fixture() -> dict:
    """Generate the mel parity fixture dict."""
    config = SyllablePredictorConfigV4()
    audio = generate_sine_wave(
        frequency_hz=440.0,
        duration_s=1.0,
        sample_rate=config.sample_rate,
    )
    mel = extract_mel_spectrogram(audio, config)

    return {
        "audio_samples": audio.tolist(),
        "mel": mel.tolist(),
        "n_mels": config.n_mels,
        "sample_rate": config.sample_rate,
        "n_fft": config.win_length,
        "hop_length": config.hop_length,
    }


def write_fixture(data: dict) -> None:
    """Write the fixture JSON to disk."""
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FIXTURE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f)


class TestMelParity:
    """Generate and validate the cross-language mel parity fixture."""

    def test_generates_valid_fixture(self):
        """Generate the fixture file and assert basic validity."""
        data = build_fixture()
        write_fixture(data)

        # Structural assertions
        assert "audio_samples" in data
        assert "mel" in data
        assert data["n_mels"] == 80
        assert data["sample_rate"] == 16000
        assert data["n_fft"] == 400
        assert data["hop_length"] == 160

        audio = np.array(data["audio_samples"], dtype=np.float32)
        mel = np.array(data["mel"], dtype=np.float32)

        assert audio.shape == (16000,), f"Expected 16000 samples, got {audio.shape}"
        assert mel.shape[0] == 80, f"Expected 80 mel bins, got {mel.shape[0]}"
        assert mel.shape[1] > 0, "Mel spectrogram has no frames"

        assert np.all(np.isfinite(mel)), "Mel spectrogram contains non-finite values"

        assert FIXTURE_PATH.exists(), f"Fixture not written to {FIXTURE_PATH}"

    def test_mel_values_are_reasonable(self):
        """Assert the mel values look like a real log-mel spectrogram."""
        data = build_fixture()
        mel = np.array(data["mel"], dtype=np.float32)

        # Log-mel values for real audio should be well above log(1e-9) ≈ -20.7
        assert mel.max() > -15.0, "Max mel value suspiciously low"
        # And not absurdly high
        assert mel.max() < 50.0, "Max mel value suspiciously high"


if __name__ == "__main__":
    # Allow running standalone to regenerate the fixture
    import sys

    print("Generating mel parity fixture...")
    data = build_fixture()
    write_fixture(data)
    mel = np.array(data["mel"])
    print(f"  audio_samples: {len(data['audio_samples'])} samples")
    print(f"  mel shape: [{mel.shape[0]}, {mel.shape[1]}]")
    print(f"  mel min/max: {mel.min():.4f} / {mel.max():.4f}")
    print(f"  Written to: {FIXTURE_PATH}")
    sys.exit(0)
