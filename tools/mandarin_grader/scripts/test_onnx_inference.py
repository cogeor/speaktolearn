#!/usr/bin/env python3
"""Test ONNX model inference with various input shapes.

This script tests the deployed ONNX model with inputs that match
what the Flutter app would send, to diagnose any shape mismatches.
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

# Path to deployed model
MODEL_PATH = Path(__file__).parent.parent.parent.parent / "apps/mobile_flutter/assets/models/model.onnx"


def test_inference(mel_frames: int, pinyin_len: int, description: str):
    """Test inference with specific input shapes."""
    print(f"\n{'='*60}")
    print(f"Test: {description}")
    print(f"  mel_frames={mel_frames}, pinyin_len={pinyin_len}")
    print(f"{'='*60}")

    # Load model
    session = ort.InferenceSession(str(MODEL_PATH))

    # Print model input specs
    print("\nModel inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.shape} ({inp.type})")

    print("\nModel outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.shape} ({out.type})")

    # Create test inputs
    mel = np.random.randn(1, 80, mel_frames).astype(np.float32)
    pinyin_ids = np.array([[1] + [10] * (pinyin_len - 1)], dtype=np.int64)  # BOS + tokens
    audio_mask = np.zeros((1, mel_frames), dtype=bool)
    pinyin_mask = np.zeros((1, pinyin_len), dtype=bool)

    print(f"\nInput shapes:")
    print(f"  mel: {mel.shape}")
    print(f"  pinyin_ids: {pinyin_ids.shape}")
    print(f"  audio_mask: {audio_mask.shape}")
    print(f"  pinyin_mask: {pinyin_mask.shape}")

    # Run inference
    try:
        outputs = session.run(
            None,
            {
                'mel': mel,
                'pinyin_ids': pinyin_ids,
                'audio_mask': audio_mask,
                'pinyin_mask': pinyin_mask,
            }
        )
        syllable_logits, tone_logits = outputs
        print(f"\nOutput shapes:")
        print(f"  syllable_logits: {syllable_logits.shape}")
        print(f"  tone_logits: {tone_logits.shape}")

        # Get predictions
        syllable_pred = np.argmax(syllable_logits, axis=-1)[0]
        tone_pred = np.argmax(tone_logits, axis=-1)[0]

        # Softmax for probabilities
        syllable_probs = np.exp(syllable_logits - np.max(syllable_logits))
        syllable_probs /= syllable_probs.sum()
        max_prob = syllable_probs.max()

        print(f"\nPredictions:")
        print(f"  syllable_id: {syllable_pred}, max_prob: {max_prob:.4f}")
        print(f"  tone_id: {tone_pred}")
        print(f"\n[OK] SUCCESS")
        return True

    except Exception as e:
        print(f"\n[FAIL] {e}")
        return False


def main():
    print(f"Testing ONNX model: {MODEL_PATH}")
    print(f"Model exists: {MODEL_PATH.exists()}")

    if not MODEL_PATH.exists():
        print("ERROR: Model file not found!")
        sys.exit(1)

    results = []

    # Test 1: Position mode format (mel=100, pinyin=2) - what export uses
    results.append(test_inference(100, 2, "Position mode (mel=100, pinyin=2)"))

    # Test 2: Shorter audio (mel=50, pinyin=2)
    results.append(test_inference(50, 2, "Shorter audio (mel=50, pinyin=2)"))

    # Test 3: Longer audio (mel=150, pinyin=2)
    results.append(test_inference(150, 2, "Longer audio (mel=150, pinyin=2)"))

    # Test 4: Wrong pinyin length (mel=100, pinyin=1) - should fail
    results.append(test_inference(100, 1, "Wrong pinyin len=1 (should fail)"))

    # Test 5: Wrong pinyin length (mel=100, pinyin=5) - should fail
    results.append(test_inference(100, 5, "Wrong pinyin len=5 (should fail)"))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n[OK] All tests passed - ONNX model is working correctly")
    else:
        print("\n[FAIL] Some tests failed - investigate the failures above")
        sys.exit(1)


if __name__ == "__main__":
    main()
