#!/usr/bin/env python3
"""Validate Flutter WAV recordings for ML model compatibility.

This script validates that Flutter recordings meet the requirements for the
SyllablePredictorV4 model:
1. Sample rate: 16000 Hz
2. Channels: 1 (mono)
3. Amplitude range: [-1, 1] (no clipping)
4. Mel-spectrogram shape: [80, ~frames]
5. Optional: Model inference test

Usage:
    # Basic validation
    python validate_flutter_audio.py recording.wav

    # With model inference test
    python validate_flutter_audio.py recording.wav --checkpoint checkpoints/model.pt

    # Verbose output with mel-spectrogram details
    python validate_flutter_audio.py recording.wav --verbose

    # Save validation report to JSON
    python validate_flutter_audio.py recording.wav --output report.json
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent))

from mandarin_grader.data.audio import TARGET_SR, load_audio, extract_mel
from mandarin_grader.model.syllable_predictor_v4 import (
    SyllablePredictorConfigV4,
    SyllablePredictorV4,
    SyllableVocab,
)


class ValidationResult:
    """Container for validation results."""

    def __init__(self):
        self.checks: dict[str, dict[str, Any]] = {}
        self.passed = True

    def add_check(
        self,
        name: str,
        passed: bool,
        expected: Any = None,
        actual: Any = None,
        message: str = "",
    ):
        """Add a validation check result."""
        self.checks[name] = {
            "passed": passed,
            "expected": expected,
            "actual": actual,
            "message": message,
        }
        if not passed:
            self.passed = False

    def print_summary(self, verbose: bool = False):
        """Print validation summary."""
        print("\n" + "=" * 70)
        print("FLUTTER AUDIO VALIDATION REPORT")
        print("=" * 70)

        for name, check in self.checks.items():
            status = "PASS" if check["passed"] else "FAIL"
            symbol = "✓" if check["passed"] else "✗"
            print(f"\n[{symbol}] {name}: {status}")

            if verbose or not check["passed"]:
                if check["expected"] is not None:
                    print(f"    Expected: {check['expected']}")
                if check["actual"] is not None:
                    print(f"    Actual:   {check['actual']}")
                if check["message"]:
                    print(f"    Note:     {check['message']}")

        print("\n" + "=" * 70)
        final_status = "ALL CHECKS PASSED" if self.passed else "VALIDATION FAILED"
        print(f"Result: {final_status}")
        print("=" * 70 + "\n")

        return self.passed

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        return {
            "passed": self.passed,
            "checks": self.checks,
        }


def validate_wav_format(audio_path: Path) -> ValidationResult:
    """Validate WAV file format using wave module.

    Args:
        audio_path: Path to WAV file

    Returns:
        ValidationResult with format checks
    """
    result = ValidationResult()

    try:
        with wave.open(str(audio_path), "rb") as wav:
            # Check sample rate
            sample_rate = wav.getframerate()
            result.add_check(
                "Sample Rate",
                passed=(sample_rate == TARGET_SR),
                expected=f"{TARGET_SR} Hz",
                actual=f"{sample_rate} Hz",
                message="Must record at 16kHz for ML model compatibility",
            )

            # Check channels (mono)
            channels = wav.getnchannels()
            result.add_check(
                "Channels",
                passed=(channels == 1),
                expected="1 (mono)",
                actual=f"{channels}",
                message="Model expects mono audio",
            )

            # Check sample width (16-bit PCM recommended)
            sample_width = wav.getsampwidth()
            result.add_check(
                "Bit Depth",
                passed=(sample_width == 2),
                expected="2 bytes (16-bit PCM)",
                actual=f"{sample_width} bytes",
                message="16-bit PCM is standard for audio processing",
            )

            # Check duration (should be reasonable, not empty)
            n_frames = wav.getnframes()
            duration_sec = n_frames / sample_rate if sample_rate > 0 else 0
            result.add_check(
                "Duration",
                passed=(0.1 <= duration_sec <= 60),
                expected="0.1s to 60s",
                actual=f"{duration_sec:.2f}s ({n_frames} frames)",
                message="Audio should not be empty or excessively long",
            )

    except wave.Error as e:
        result.add_check(
            "WAV File Format",
            passed=False,
            message=f"Failed to read WAV file: {e}",
        )
    except Exception as e:
        result.add_check(
            "WAV File Format",
            passed=False,
            message=f"Unexpected error: {e}",
        )

    return result


def validate_audio_content(audio_path: Path) -> tuple[ValidationResult, np.ndarray | None]:
    """Validate audio content (amplitude range).

    Args:
        audio_path: Path to audio file

    Returns:
        Tuple of (ValidationResult, audio_samples)
    """
    result = ValidationResult()

    try:
        # Load audio using librosa (handles resampling if needed)
        audio = load_audio(audio_path, target_sr=TARGET_SR)

        # Check amplitude range
        min_amp = float(audio.min())
        max_amp = float(audio.max())
        abs_max = max(abs(min_amp), abs(max_amp))

        result.add_check(
            "Amplitude Range",
            passed=(abs_max <= 1.0),
            expected="[-1.0, 1.0]",
            actual=f"[{min_amp:.4f}, {max_amp:.4f}]",
            message="Normalized amplitude prevents clipping" if abs_max <= 1.0 else "Audio may be clipping!",
        )

        # Check for silence (RMS should be reasonable)
        rms = float(np.sqrt(np.mean(audio**2)))
        result.add_check(
            "Signal Level (RMS)",
            passed=(rms >= 0.001),
            expected=">= 0.001",
            actual=f"{rms:.6f}",
            message="Audio has reasonable signal level" if rms >= 0.001 else "Audio is too quiet or silent",
        )

        # Check for extreme clipping (many samples at exactly ±1.0)
        clipped_samples = int(np.sum((np.abs(audio) >= 0.99)))
        clipping_ratio = clipped_samples / len(audio)
        result.add_check(
            "Clipping",
            passed=(clipping_ratio < 0.01),
            expected="< 1% of samples at ±1.0",
            actual=f"{clipping_ratio*100:.2f}% ({clipped_samples}/{len(audio)} samples)",
            message="Minimal clipping detected" if clipping_ratio < 0.01 else "Significant clipping detected!",
        )

        return result, audio

    except Exception as e:
        result.add_check(
            "Audio Loading",
            passed=False,
            message=f"Failed to load audio: {e}",
        )
        return result, None


def validate_mel_spectrogram(audio: np.ndarray, config: SyllablePredictorConfigV4) -> ValidationResult:
    """Validate mel-spectrogram extraction.

    Args:
        audio: Audio samples
        config: Model configuration with mel parameters

    Returns:
        ValidationResult with mel-spectrogram checks
    """
    result = ValidationResult()

    try:
        # Extract mel-spectrogram using training parameters
        mel = extract_mel(
            audio,
            sr=config.sample_rate,
            n_mels=config.n_mels,
            hop_length=config.hop_length,
            win_length=config.win_length,
        )

        # Check shape
        n_mels, n_frames = mel.shape
        duration_sec = len(audio) / config.sample_rate
        expected_frames = int(duration_sec * 100)  # ~100 frames per second

        result.add_check(
            "Mel-Spectrogram Shape",
            passed=(n_mels == config.n_mels),
            expected=f"[{config.n_mels}, ~{expected_frames}]",
            actual=f"[{n_mels}, {n_frames}]",
            message=f"Duration: {duration_sec:.2f}s, ~{n_frames/duration_sec:.1f} frames/sec" if duration_sec > 0 else "",
        )

        # Check that mel values are finite
        is_finite = bool(np.all(np.isfinite(mel)))
        result.add_check(
            "Mel Values",
            passed=is_finite,
            expected="All finite values",
            actual="All finite" if is_finite else "Contains NaN or Inf",
            message="Mel-spectrogram computed successfully",
        )

        # Check mel value range (log-mel should be in dB scale, typically negative)
        mel_min = float(mel.min())
        mel_max = float(mel.max())
        result.add_check(
            "Mel Value Range",
            passed=(mel_min < 0 and mel_max > mel_min),
            expected="Negative dB values with range",
            actual=f"[{mel_min:.2f}, {mel_max:.2f}] dB",
            message="Log-mel spectrogram in expected dB range",
        )

    except Exception as e:
        result.add_check(
            "Mel-Spectrogram Extraction",
            passed=False,
            message=f"Failed to extract mel-spectrogram: {e}",
        )

    return result


def validate_model_inference(
    audio: np.ndarray,
    checkpoint_path: Path,
    config: SyllablePredictorConfigV4,
) -> ValidationResult:
    """Validate model inference on the audio.

    Args:
        audio: Audio samples
        checkpoint_path: Path to model checkpoint
        config: Model configuration

    Returns:
        ValidationResult with inference checks
    """
    result = ValidationResult()

    try:
        import torch

        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SyllablePredictorV4(config).to(device)
        vocab = SyllableVocab()

        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            result.add_check(
                "Model Loading",
                passed=True,
                actual=f"Loaded checkpoint (epoch {checkpoint.get('epoch', '?')})",
                message=f"Model on device: {device}",
            )

            # Extract mel-spectrogram
            mel = extract_mel(
                audio,
                sr=config.sample_rate,
                n_mels=config.n_mels,
                hop_length=config.hop_length,
                win_length=config.win_length,
            )

            # Prepare inputs (use BOS token as minimal context)
            mel_tensor = torch.from_numpy(mel).float().unsqueeze(0).to(device)
            pinyin_ids = torch.tensor([[vocab.bos_token]], dtype=torch.long).to(device)

            # Run inference
            with torch.no_grad():
                syllable_logits, tone_logits = model(mel_tensor, pinyin_ids)

            # Check output shapes
            syl_shape = tuple(syllable_logits.shape)
            tone_shape = tuple(tone_logits.shape)

            result.add_check(
                "Inference Execution",
                passed=True,
                expected=f"Syllable: [1, {config.n_syllables}], Tone: [1, {config.n_tones}]",
                actual=f"Syllable: {syl_shape}, Tone: {tone_shape}",
                message="Model inference completed successfully",
            )

            # Get predictions
            syl_pred = syllable_logits.argmax(dim=-1).item()
            tone_pred = tone_logits.argmax(dim=-1).item()
            predicted_syllable = vocab.decode(syl_pred)

            result.add_check(
                "Prediction",
                passed=True,
                actual=f"Syllable: {predicted_syllable} (id={syl_pred}), Tone: {tone_pred}",
                message="Model produced valid predictions",
            )

        else:
            result.add_check(
                "Model Loading",
                passed=False,
                message=f"Checkpoint not found: {checkpoint_path}",
            )

    except ImportError:
        result.add_check(
            "Model Inference",
            passed=False,
            message="PyTorch not available. Install torch to run inference tests.",
        )
    except Exception as e:
        result.add_check(
            "Model Inference",
            passed=False,
            message=f"Inference failed: {e}",
        )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Validate Flutter WAV recordings for ML model compatibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "audio_file",
        type=Path,
        help="Path to Flutter WAV recording",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to V4 model checkpoint (optional, for inference test)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed information for all checks",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save validation report to JSON file",
    )

    args = parser.parse_args()

    print("\nValidating Flutter Audio Recording")
    print(f"File: {args.audio_file}")
    print(f"Size: {args.audio_file.stat().st_size / 1024:.2f} KB\n")

    # Initialize combined results
    all_results = ValidationResult()

    # 1. Validate WAV format
    print("[1/4] Checking WAV file format...")
    format_result = validate_wav_format(args.audio_file)
    all_results.checks.update(format_result.checks)
    if not format_result.passed:
        all_results.passed = False

    # 2. Validate audio content
    print("[2/4] Checking audio content...")
    content_result, audio = validate_audio_content(args.audio_file)
    all_results.checks.update(content_result.checks)
    if not content_result.passed:
        all_results.passed = False

    # 3. Validate mel-spectrogram (if audio loaded successfully)
    if audio is not None:
        print("[3/4] Validating mel-spectrogram extraction...")
        config = SyllablePredictorConfigV4()
        mel_result = validate_mel_spectrogram(audio, config)
        all_results.checks.update(mel_result.checks)
        if not mel_result.passed:
            all_results.passed = False

        # 4. Optional: Model inference test
        if args.checkpoint:
            print("[4/4] Testing model inference...")
            inference_result = validate_model_inference(audio, args.checkpoint, config)
            all_results.checks.update(inference_result.checks)
            if not inference_result.passed:
                all_results.passed = False
        else:
            print("[4/4] Skipping model inference (no checkpoint provided)")
    else:
        print("[3/4] Skipping mel-spectrogram validation (audio loading failed)")
        print("[4/4] Skipping model inference (audio loading failed)")

    # Print summary
    all_passed = all_results.print_summary(verbose=args.verbose)

    # Save to JSON if requested
    if args.output:
        output_data = {
            "file": str(args.audio_file),
            "validation": all_results.to_dict(),
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Validation report saved to: {args.output}\n")

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
