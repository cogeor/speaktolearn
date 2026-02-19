#!/usr/bin/env python
"""Test augmentation to ensure output matches real recording profiles.

This script compares:
1. AISHELL3 training data profile (studio recordings)
2. Pulled recordings profile (real user recordings from mobile)
3. Augmented AISHELL3 data profile (what model sees during training)

Goal: Augmented training data should have similar characteristics to real
user recordings to improve generalization.

Key differences identified:
- AISHELL3: RMS ~0.03-0.06, peak ~0.2-0.4, clean studio
- Pulled recordings: RMS ~0.01-0.04, peak ~0.1-0.4, more variable volume
- Some pulled recordings have very low volume (ts_000009: rms=0.01)
- Some have high volume (ts_000008 old: rms=0.18, peak=0.94)

Recommended augmentations:
1. Volume variation: -15dB to +6dB to match pulled_recordings range
2. Speed variation: ±5% (already in V6)
3. Pitch shift: ±2 semitones (already in V6)
4. Formant shift: ±10% (already in V6)
5. Random padding: distribute silence before/after (new feature)
6. Background noise: SNR 15-40dB range
"""

import argparse
import wave
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load WAV file and return audio + sample rate."""
    with wave.open(str(path), 'rb') as w:
        sr = w.getframerate()
        nf = w.getnframes()
        raw = w.readframes(nf)
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sr


def compute_audio_profile(audio: np.ndarray, sr: int = 16000) -> dict:
    """Compute audio profile statistics."""
    rms = np.sqrt(np.mean(audio**2))
    peak = np.abs(audio).max()
    duration = len(audio) / sr

    # Compute energy envelope for silence detection
    frame_size = int(0.025 * sr)  # 25ms frames
    hop_size = int(0.010 * sr)    # 10ms hop

    n_frames = max(1, (len(audio) - frame_size) // hop_size + 1)
    energy = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_size
        end = min(start + frame_size, len(audio))
        frame = audio[start:end]
        energy[i] = np.sqrt(np.mean(frame**2))

    # Find voice activity
    thresh = 0.01  # -40dB threshold
    voice_frames = energy > thresh
    if voice_frames.any():
        first_voice = np.argmax(voice_frames) * hop_size / sr
        last_voice = (len(voice_frames) - np.argmax(voice_frames[::-1]) - 1) * hop_size / sr
        silence_before = first_voice
        silence_after = duration - last_voice
    else:
        silence_before = 0
        silence_after = 0

    # Compute mel spectrogram stats
    try:
        import librosa
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80, hop_length=160)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_mean = mel_db.mean()
        mel_std = mel_db.std()
    except ImportError:
        mel_mean = 0
        mel_std = 0

    return {
        'duration': duration,
        'rms': rms,
        'rms_db': 20 * np.log10(rms + 1e-10),
        'peak': peak,
        'peak_db': 20 * np.log10(peak + 1e-10),
        'silence_before': silence_before,
        'silence_after': silence_after,
        'mel_mean': mel_mean,
        'mel_std': mel_std,
    }


def apply_augmentation(
    audio: np.ndarray,
    sr: int = 16000,
    speed_var: float = 0.05,
    pitch_shift_st: float = 2.0,
    formant_shift_pct: float = 10.0,
    volume_var_db: float = 12.0,
    noise_snr_db: Optional[float] = 30.0,
    random_padding_s: float = 0.3,
) -> np.ndarray:
    """Apply training augmentation to audio."""
    from mandarin_grader.data.augmentation import pitch_shift, formant_shift

    result = audio.copy()

    # 1. Pitch shift
    if pitch_shift_st > 0:
        semitones = np.random.uniform(-pitch_shift_st, pitch_shift_st)
        if abs(semitones) > 0.1:
            result = pitch_shift(result, semitones, sr)

    # 2. Formant shift
    if formant_shift_pct > 0:
        shift_ratio = 1.0 + np.random.uniform(-formant_shift_pct, formant_shift_pct) / 100.0
        if abs(shift_ratio - 1.0) > 0.01:
            result = formant_shift(result, shift_ratio, sr)

    # 3. Speed variation
    if speed_var > 0:
        factor = 1.0 + np.random.uniform(-speed_var, speed_var)
        if abs(factor - 1.0) > 0.01:
            new_length = int(len(result) / factor)
            if new_length > 1:
                indices = np.linspace(0, len(result) - 1, new_length)
                result = np.interp(indices, np.arange(len(result)), result).astype(np.float32)

    # 4. Volume variation
    if volume_var_db > 0:
        gain_db = np.random.uniform(-volume_var_db, volume_var_db / 2)
        gain_linear = 10 ** (gain_db / 20)
        result = result * gain_linear

    # 5. Additive noise
    if noise_snr_db is not None:
        signal_power = np.mean(result ** 2)
        if signal_power > 1e-10:
            snr_linear = 10 ** (noise_snr_db / 10)
            noise_power = signal_power / snr_linear
            noise = np.random.randn(len(result)).astype(np.float32) * np.sqrt(noise_power)
            result = result + noise

    # 6. Random padding (silence before/after)
    if random_padding_s > 0:
        pad_samples = int(random_padding_s * sr)
        pad_before = np.random.randint(0, pad_samples + 1)
        pad_after = np.random.randint(0, pad_samples + 1)
        result = np.pad(result, (pad_before, pad_after), mode='constant', constant_values=0)

    return result


def main():
    parser = argparse.ArgumentParser(description="Test augmentation profile matching")
    parser.add_argument("--n-augmentations", type=int, default=10,
                       help="Number of augmented versions to generate per file")
    parser.add_argument("--plot", action="store_true", help="Generate comparison plots")
    parser.add_argument("--output", type=str, default="augmentation_report.json",
                       help="Output JSON report path")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent

    # Collect profiles
    profiles = {
        'aishell3': [],
        'pulled_old': [],
        'pulled_new': [],
        'augmented_v6': [],  # Current V6 augmentation
        'augmented_new': [],  # Proposed new augmentation
    }

    # 1. Profile AISHELL3 samples
    print("=== Profiling AISHELL3 Training Data ===")
    aishell_dir = base_dir / "datasets" / "aishell3" / "train" / "wav"
    if aishell_dir.exists():
        speakers = list(aishell_dir.iterdir())[:5]
        for speaker in speakers:
            wavs = list(speaker.glob('*.wav'))[:3]
            for wav_path in wavs:
                audio, sr = load_wav(wav_path)
                profile = compute_audio_profile(audio, sr)
                profile['source'] = f"aishell3/{speaker.name}/{wav_path.name}"
                profiles['aishell3'].append(profile)
                print(f"  {profile['source']}: rms={profile['rms']:.4f}, peak={profile['peak']:.4f}")

    # 2. Profile pulled recordings
    print("\n=== Profiling Pulled Recordings (Old) ===")
    pulled_dir = base_dir / "pulled_recordings"
    if pulled_dir.exists():
        for wav_path in sorted(pulled_dir.glob('*.wav')):
            audio, sr = load_wav(wav_path)
            profile = compute_audio_profile(audio, sr)
            profile['source'] = f"pulled_old/{wav_path.name}"
            profiles['pulled_old'].append(profile)
            print(f"  {profile['source']}: rms={profile['rms']:.4f}, peak={profile['peak']:.4f}, "
                  f"silence_before={profile['silence_before']:.2f}s, silence_after={profile['silence_after']:.2f}s")

    print("\n=== Profiling Pulled Recordings (New) ===")
    pulled_new_dir = base_dir / "pulled_recordings_new"
    if pulled_new_dir.exists():
        for wav_path in sorted(pulled_new_dir.glob('*.wav')):
            audio, sr = load_wav(wav_path)
            profile = compute_audio_profile(audio, sr)
            profile['source'] = f"pulled_new/{wav_path.name}"
            profiles['pulled_new'].append(profile)
            print(f"  {profile['source']}: rms={profile['rms']:.4f}, peak={profile['peak']:.4f}, "
                  f"silence_before={profile['silence_before']:.2f}s, silence_after={profile['silence_after']:.2f}s")

    # 3. Generate augmented versions
    if profiles['aishell3']:
        print(f"\n=== Generating {args.n_augmentations} Augmented Versions (V6 settings) ===")
        source_wav = aishell_dir / list(aishell_dir.iterdir())[0] / list(list(aishell_dir.iterdir())[0].glob('*.wav'))[0]
        audio, sr = load_wav(source_wav)

        for i in range(args.n_augmentations):
            aug_audio = apply_augmentation(
                audio, sr,
                speed_var=0.05,
                pitch_shift_st=2.0,
                formant_shift_pct=10.0,
                volume_var_db=12.0,  # V6 default
                noise_snr_db=None,   # V6 had no noise
                random_padding_s=0.0,  # V6 had no random padding
            )
            profile = compute_audio_profile(aug_audio, sr)
            profile['source'] = f"augmented_v6_{i}"
            profiles['augmented_v6'].append(profile)

        print(f"\n=== Generating {args.n_augmentations} Augmented Versions (New settings) ===")
        for i in range(args.n_augmentations):
            aug_audio = apply_augmentation(
                audio, sr,
                speed_var=0.05,
                pitch_shift_st=2.0,
                formant_shift_pct=10.0,
                volume_var_db=15.0,   # Wider range
                noise_snr_db=25.0,    # Add noise
                random_padding_s=0.3, # Add random silence padding
            )
            profile = compute_audio_profile(aug_audio, sr)
            profile['source'] = f"augmented_new_{i}"
            profiles['augmented_new'].append(profile)

    # 4. Compute summary statistics
    print("\n=== Summary Statistics ===")
    summary = {}
    for group, items in profiles.items():
        if items:
            rms_vals = [p['rms'] for p in items]
            peak_vals = [p['peak'] for p in items]
            silence_before = [p['silence_before'] for p in items]
            silence_after = [p['silence_after'] for p in items]

            summary[group] = {
                'count': len(items),
                'rms_mean': float(np.mean(rms_vals)),
                'rms_std': float(np.std(rms_vals)),
                'rms_min': float(np.min(rms_vals)),
                'rms_max': float(np.max(rms_vals)),
                'peak_mean': float(np.mean(peak_vals)),
                'peak_std': float(np.std(peak_vals)),
                'silence_before_mean': float(np.mean(silence_before)),
                'silence_after_mean': float(np.mean(silence_after)),
            }

            print(f"\n{group}:")
            print(f"  RMS:  {summary[group]['rms_min']:.4f} - {summary[group]['rms_max']:.4f} "
                  f"(mean={summary[group]['rms_mean']:.4f}, std={summary[group]['rms_std']:.4f})")
            print(f"  Peak: mean={summary[group]['peak_mean']:.4f}, std={summary[group]['peak_std']:.4f}")
            print(f"  Silence: before={summary[group]['silence_before_mean']:.2f}s, "
                  f"after={summary[group]['silence_after_mean']:.2f}s")

    # 5. Recommendations
    print("\n=== Recommendations ===")
    if 'pulled_new' in summary and 'augmented_v6' in summary:
        pulled_rms_range = (summary['pulled_new']['rms_min'], summary['pulled_new']['rms_max'])
        v6_rms_range = (summary['augmented_v6']['rms_min'], summary['augmented_v6']['rms_max'])

        print(f"Pulled recordings RMS range: {pulled_rms_range[0]:.4f} - {pulled_rms_range[1]:.4f}")
        print(f"V6 augmented RMS range:      {v6_rms_range[0]:.4f} - {v6_rms_range[1]:.4f}")

        if pulled_rms_range[0] < v6_rms_range[0] or pulled_rms_range[1] > v6_rms_range[1]:
            print("  -> MISMATCH: V6 volume range doesn't cover pulled recordings")
            print("  -> Recommendation: Increase volume_var_db to 15-18dB")

        if 'pulled_new' in summary:
            if summary['pulled_new']['silence_before_mean'] > 0.1:
                print(f"  -> Pulled recordings have ~{summary['pulled_new']['silence_before_mean']:.2f}s silence before speech")
                print("  -> Recommendation: Enable random_padding augmentation")

    # 6. Save report
    # Convert numpy types to Python native for JSON
    def convert_to_native(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        return obj

    report = {
        'profiles': convert_to_native({k: v for k, v in profiles.items()}),
        'summary': summary,
    }
    output_path = base_dir / args.output
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {output_path}")

    # 7. Generate plots
    if args.plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # RMS distribution
        ax = axes[0, 0]
        for group in ['aishell3', 'pulled_old', 'pulled_new', 'augmented_v6', 'augmented_new']:
            if profiles[group]:
                rms_vals = [p['rms'] for p in profiles[group]]
                ax.hist(rms_vals, bins=20, alpha=0.5, label=group)
        ax.set_xlabel('RMS')
        ax.set_ylabel('Count')
        ax.set_title('RMS Distribution')
        ax.legend()

        # Peak distribution
        ax = axes[0, 1]
        for group in ['aishell3', 'pulled_old', 'pulled_new', 'augmented_v6', 'augmented_new']:
            if profiles[group]:
                peak_vals = [p['peak'] for p in profiles[group]]
                ax.hist(peak_vals, bins=20, alpha=0.5, label=group)
        ax.set_xlabel('Peak')
        ax.set_ylabel('Count')
        ax.set_title('Peak Distribution')
        ax.legend()

        # Silence before
        ax = axes[1, 0]
        for group in ['pulled_old', 'pulled_new', 'augmented_new']:
            if profiles[group]:
                vals = [p['silence_before'] for p in profiles[group]]
                ax.hist(vals, bins=20, alpha=0.5, label=group)
        ax.set_xlabel('Silence Before (s)')
        ax.set_ylabel('Count')
        ax.set_title('Silence Before Speech')
        ax.legend()

        # RMS scatter
        ax = axes[1, 1]
        colors = {'aishell3': 'blue', 'pulled_old': 'red', 'pulled_new': 'orange',
                  'augmented_v6': 'green', 'augmented_new': 'purple'}
        for group, color in colors.items():
            if profiles[group]:
                rms_vals = [p['rms'] for p in profiles[group]]
                peak_vals = [p['peak'] for p in profiles[group]]
                ax.scatter(rms_vals, peak_vals, c=color, alpha=0.5, label=group)
        ax.set_xlabel('RMS')
        ax.set_ylabel('Peak')
        ax.set_title('RMS vs Peak')
        ax.legend()

        plt.tight_layout()
        plot_path = base_dir / "augmentation_comparison.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
