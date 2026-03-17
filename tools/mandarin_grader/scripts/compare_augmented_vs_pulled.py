#!/usr/bin/env python3
"""Compare augmented AISHELL mel spectrograms to pulled recordings.

This shows how the 'mobile' augmentation preset transforms AISHELL
data to look more like real mobile phone recordings.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from mandarin_grader.data.audio import load_audio
from mandarin_grader.data.mel_augmentation import (
    apply_mel_augmentation,
    get_preset_config,
)
from mandarin_grader.model.syllable_predictor_v4 import (
    extract_mel_spectrogram,
    SyllablePredictorConfigV4,
)


def main():
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / 'augmentation_comparison'
    output_dir.mkdir(exist_ok=True)

    # Load AISHELL sample
    aishell_path = base_dir / "datasets" / "aishell3" / "train" / "wav" / "SSB0005" / "SSB00050001.wav"
    if not aishell_path.exists():
        print(f"AISHELL sample not found: {aishell_path}")
        return

    # Load pulled recording
    pulled_paths = [
        base_dir / "pulled_recordings_new" / "ts_000001.wav",
        base_dir / "pulled_recordings" / "ts_000001.wav",
    ]
    pulled_path = None
    for p in pulled_paths:
        if p.exists():
            pulled_path = p
            break
    if not pulled_path:
        print("No pulled recordings found")
        return

    # Extract mels using the same extractor the model uses (n_fft=400, numpy FFT)
    _mel_config = SyllablePredictorConfigV4()

    print("Loading AISHELL sample...")
    aishell_audio = load_audio(aishell_path, target_sr=16000)
    aishell_mel = extract_mel_spectrogram(aishell_audio, _mel_config)

    print("Loading pulled recording...")
    pulled_audio = load_audio(pulled_path, target_sr=16000)
    pulled_mel = extract_mel_spectrogram(pulled_audio, _mel_config)

    # Generate augmented versions
    mobile_config = get_preset_config('mobile')
    studio_config = get_preset_config('studio')

    # Generate multiple mobile augmentations
    mobile_augs = []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        aug = apply_mel_augmentation(aishell_mel, mobile_config, rng)
        mobile_augs.append(aug)

    studio_aug = apply_mel_augmentation(aishell_mel, studio_config, rng=np.random.default_rng(42))

    # Truncate for display (first 300 frames = 3s)
    max_frames = 300
    aishell_mel = aishell_mel[:, :max_frames]
    pulled_mel = pulled_mel[:, :max_frames]
    studio_aug = studio_aug[:, :max_frames]
    mobile_augs = [aug[:, :max_frames] for aug in mobile_augs]

    # Create comparison plot
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle('Domain Adaptation: AISHELL + Mobile Augmentation vs Pulled Recording', fontsize=14)

    vmin, vmax = -18, 5

    # Row 1: Original sources
    im = axes[0, 0].imshow(aishell_mel, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('AISHELL Original (studio)', fontsize=10)
    axes[0, 0].set_ylabel('Mel bin')

    axes[0, 1].imshow(pulled_mel, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f'Pulled Recording ({pulled_path.name})', fontsize=10)

    axes[0, 2].imshow(studio_aug, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, 2].set_title('AISHELL + Studio Augment', fontsize=10)

    # Row 2-3: Mobile augmented samples
    for i, aug in enumerate(mobile_augs):
        row = 1 + i // 3
        col = i % 3
        axes[row, col].imshow(aug, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[row, col].set_title(f'AISHELL + Mobile Augment #{i+1}', fontsize=10)
        if col == 0:
            axes[row, col].set_ylabel('Mel bin')
        if row == 2:
            axes[row, col].set_xlabel('Frame')

    plt.tight_layout()
    out_path = output_dir / 'aishell_vs_pulled_comparison.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

    # Print statistics comparison
    print("\n" + "=" * 70)
    print("Statistics Comparison")
    print("=" * 70)

    def print_stats(mel, name):
        print(f"{name:30s}: mean={mel.mean():7.3f}, std={mel.std():6.3f}, "
              f"min={mel.min():7.3f}, max={mel.max():6.3f}")

    print_stats(aishell_mel, "AISHELL Original")
    print_stats(pulled_mel, "Pulled Recording")
    print_stats(studio_aug, "AISHELL + Studio Aug")
    for i, aug in enumerate(mobile_augs[:3]):
        print_stats(aug, f"AISHELL + Mobile Aug #{i+1}")

    # Compare low-frequency content
    print("\nLow-frequency energy (bins 0-10):")
    print(f"  AISHELL:        {aishell_mel[:10, :].mean():.3f}")
    print(f"  Pulled:         {pulled_mel[:10, :].mean():.3f}")
    print(f"  Mobile Aug #1:  {mobile_augs[0][:10, :].mean():.3f}")


if __name__ == '__main__':
    main()
