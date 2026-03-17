#!/usr/bin/env python3
"""Compare audio characteristics between AISHELL and pulled recordings.

This script quantifies the domain gap between studio-quality AISHELL recordings
and real-world mobile phone recordings by measuring:

1. SRMR (Speech-to-Reverberation Modulation Energy Ratio) - reverb estimation
2. High-frequency energy ratio - codec artifact detection
3. Spectral centroid statistics - frequency distribution
4. Mel spectrogram statistics - training feature distribution
5. SNR estimation - noise level

Usage:
    python compare_audio_domains.py
    python compare_audio_domains.py --plot
    python compare_audio_domains.py --output comparison_report.json

References:
- SRMR: Falk et al., "A Non-Intrusive Quality and Intelligibility Measure of
  Reverberant and Dereverberated Speech", IEEE TASLP 2010
- High-freq analysis: AAC codecs typically cut off at 15-16kHz (128kbps)
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class AudioMetrics:
    """Comprehensive audio metrics for domain comparison."""

    source: str
    file_path: str

    # Basic properties
    duration_s: float = 0.0
    sample_rate: int = 16000

    # Volume metrics
    rms: float = 0.0
    rms_db: float = 0.0
    peak: float = 0.0
    peak_db: float = 0.0
    dynamic_range_db: float = 0.0

    # Reverb metrics (SRMR)
    srmr: float = 0.0  # Higher = cleaner/less reverberant
    srmr_norm: float = 0.0  # Normalized SRMR

    # Temporal envelope metrics (reverb indicators)
    envelope_variance: float = 0.0  # Lower = more reverb (smoothed envelope)
    envelope_kurtosis: float = 0.0  # Lower = more reverb (less peaky)
    attack_sharpness: float = 0.0  # Lower = more reverb (slower attacks)

    # Spectral metrics
    spectral_centroid_mean: float = 0.0
    spectral_centroid_std: float = 0.0
    spectral_bandwidth_mean: float = 0.0
    spectral_flatness_mean: float = 0.0

    # High-frequency analysis (codec detection)
    hf_energy_ratio: float = 0.0  # Energy above 5kHz / total energy
    hf_rolloff_hz: float = 0.0  # Frequency below which 85% of energy exists

    # Low-frequency analysis (room resonance)
    lf_energy_ratio: float = 0.0  # Energy below 300Hz / total

    # Mel spectrogram stats
    mel_mean: float = 0.0
    mel_std: float = 0.0
    mel_min: float = 0.0
    mel_max: float = 0.0

    # Mel temporal dynamics
    mel_delta_mean: float = 0.0  # Mean of delta features
    mel_delta_std: float = 0.0   # Std of delta features

    # Silence/padding
    silence_before_s: float = 0.0
    silence_after_s: float = 0.0

    # Estimated SNR
    estimated_snr_db: float = 0.0


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load WAV file and return normalized audio + sample rate."""
    with wave.open(str(path), 'rb') as w:
        sr = w.getframerate()
        nf = w.getnframes()
        nc = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(nf)

        if sw == 2:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 4:
            audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            audio = np.frombuffer(raw, dtype=np.int8).astype(np.float32) / 128.0

        # Convert to mono if stereo
        if nc == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)

    return audio, sr


def load_audio_any_format(path: Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load audio from any format using librosa."""
    import librosa
    audio, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return audio.astype(np.float32), sr


def compute_srmr(audio: np.ndarray, sr: int = 16000) -> tuple[float, float]:
    """Compute SRMR (Speech-to-Reverberation Modulation Energy Ratio).

    Returns (srmr, srmr_norm) tuple.
    Higher values indicate cleaner/less reverberant speech.
    Typical values:
    - Clean studio: > 6
    - Moderate reverb: 3-6
    - Heavy reverb: < 3
    """
    try:
        from srmrpy import srmr
        # Fast version for efficiency
        ratio, energy = srmr(audio, sr, fast=True, norm=False)
        ratio_norm, _ = srmr(audio, sr, fast=True, norm=True)
        return float(ratio), float(ratio_norm)
    except ImportError:
        # Fallback: estimate reverb via modulation spectrum analysis
        return _estimate_srmr_fallback(audio, sr)


def _estimate_srmr_fallback(audio: np.ndarray, sr: int) -> tuple[float, float]:
    """Fallback SRMR estimation using modulation spectrum analysis.

    This is a simplified approximation when SRMRpy is not installed.
    Based on the principle that reverberant speech has more energy in
    low modulation frequencies (<4Hz) compared to clean speech.
    """
    import librosa

    # Compute envelope using multiple frequency bands (gammatone approximation)
    n_bands = 8
    frame_length = int(0.032 * sr)  # 32ms frames
    hop_length = int(0.004 * sr)    # 4ms hop for better modulation resolution

    # Get amplitude envelope per band
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_bands,
        hop_length=hop_length, n_fft=frame_length * 2
    )

    if mel_spec.shape[1] < 32:
        return 0.0, 0.0

    # Compute modulation spectrum for each band
    mod_sr = sr / hop_length  # ~250 Hz modulation sample rate

    total_speech_energy = 0.0
    total_reverb_energy = 0.0

    for band_idx in range(n_bands):
        band_env = mel_spec[band_idx, :]

        # Remove DC and normalize
        band_env = band_env - np.mean(band_env)
        if np.std(band_env) < 1e-10:
            continue
        band_env = band_env / np.std(band_env)

        # FFT of envelope (modulation spectrum)
        n_fft = min(256, len(band_env))
        mod_spectrum = np.abs(np.fft.rfft(band_env, n=n_fft)) ** 2
        mod_freqs = np.fft.rfftfreq(n_fft, 1.0 / mod_sr)

        # Speech modulation range: 3-10Hz (captures syllable rate ~4-5Hz)
        speech_band = (mod_freqs >= 3) & (mod_freqs <= 10)
        # Reverb/slow modulation: 0.5-3Hz
        reverb_band = (mod_freqs >= 0.5) & (mod_freqs < 3)

        total_speech_energy += np.sum(mod_spectrum[speech_band])
        total_reverb_energy += np.sum(mod_spectrum[reverb_band])

    # Compute ratio
    if total_reverb_energy < 1e-10:
        return 8.0, 6.0  # Very clean signal

    ratio = total_speech_energy / total_reverb_energy

    # Scale to approximate SRMR range (typically 2-10 for speech)
    # Clean studio: 6-10, Moderate reverb: 3-6, Heavy reverb: <3
    srmr_approx = np.log(ratio + 1) * 3.0
    srmr_approx = float(np.clip(srmr_approx, 0, 12))

    return srmr_approx, srmr_approx * 0.7


def compute_envelope_metrics(audio: np.ndarray, sr: int) -> dict:
    """Compute temporal envelope metrics for reverb detection.

    Reverberant speech has:
    - Lower envelope variance (smoothed by reverb tail)
    - Lower envelope kurtosis (less peaky)
    - Slower attack times (smeared transients)
    """
    import librosa
    from scipy import stats

    # Compute amplitude envelope
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)

    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

    if len(rms) < 10:
        return {'envelope_variance': 0, 'envelope_kurtosis': 0, 'attack_sharpness': 0}

    # Normalize envelope
    rms_norm = rms / (np.max(rms) + 1e-10)

    # Variance of envelope (lower = more reverb)
    env_variance = float(np.var(rms_norm))

    # Kurtosis of envelope (lower = more reverb, less peaky)
    env_kurtosis = float(stats.kurtosis(rms_norm))

    # Attack sharpness: mean of positive derivatives
    # (lower = smoother attacks = more reverb)
    diff = np.diff(rms_norm)
    positive_diffs = diff[diff > 0]
    attack_sharpness = float(np.mean(positive_diffs)) if len(positive_diffs) > 0 else 0

    return {
        'envelope_variance': env_variance,
        'envelope_kurtosis': env_kurtosis,
        'attack_sharpness': attack_sharpness,
    }


def compute_spectral_metrics(audio: np.ndarray, sr: int) -> dict:
    """Compute spectral analysis metrics."""
    import librosa

    # Compute STFT
    n_fft = 2048
    hop_length = 512

    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]

    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]

    # Spectral flatness (measures how noise-like vs tonal)
    flatness = librosa.feature.spectral_flatness(S=S)[0]

    # Spectral rolloff (frequency below which 85% of energy exists)
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0]

    # High-frequency energy ratio
    # For 16kHz audio, Nyquist is 8kHz, so use 5-7kHz as "high" range
    # This detects if phone recording has less high freq content
    hf_threshold = 5000  # Adjusted for 16kHz sample rate
    hf_idx = np.argmin(np.abs(freqs - hf_threshold))

    total_energy = np.sum(S ** 2)
    hf_energy = np.sum(S[hf_idx:, :] ** 2)
    hf_ratio = hf_energy / (total_energy + 1e-10)

    # Low-frequency energy ratio (room resonance indicator)
    lf_threshold = 300
    lf_idx = np.argmin(np.abs(freqs - lf_threshold))
    lf_energy = np.sum(S[:lf_idx, :] ** 2)
    lf_ratio = lf_energy / (total_energy + 1e-10)

    return {
        'spectral_centroid_mean': float(np.mean(centroid)),
        'spectral_centroid_std': float(np.std(centroid)),
        'spectral_bandwidth_mean': float(np.mean(bandwidth)),
        'spectral_flatness_mean': float(np.mean(flatness)),
        'hf_energy_ratio': float(hf_ratio),
        'hf_rolloff_hz': float(np.mean(rolloff)),
        'lf_energy_ratio': float(lf_ratio),
    }


def compute_mel_stats(audio: np.ndarray, sr: int) -> dict:
    """Compute mel spectrogram statistics matching training pipeline."""
    from mandarin_grader.model.syllable_predictor_v4 import (
        extract_mel_spectrogram,
        SyllablePredictorConfigV4,
    )

    # Use the same extractor the model uses (n_fft=400, custom numpy FFT)
    config = SyllablePredictorConfigV4()
    mel_log = extract_mel_spectrogram(audio, config)

    # Compute delta features (temporal dynamics)
    # Reverb tends to smooth out temporal changes
    if mel_log.shape[1] > 3:
        delta = np.diff(mel_log, axis=1)
        delta_mean = float(np.mean(np.abs(delta)))
        delta_std = float(np.std(delta))
    else:
        delta_mean = 0.0
        delta_std = 0.0

    return {
        'mel_mean': float(np.mean(mel_log)),
        'mel_std': float(np.std(mel_log)),
        'mel_min': float(np.min(mel_log)),
        'mel_max': float(np.max(mel_log)),
        'mel_delta_mean': delta_mean,
        'mel_delta_std': delta_std,
    }


def estimate_snr(audio: np.ndarray, sr: int) -> float:
    """Estimate SNR by comparing speech vs silence regions."""
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)

    # Compute frame energies
    n_frames = (len(audio) - frame_length) // hop_length + 1
    if n_frames < 5:
        return 0.0

    energies = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        frame = audio[start:end]
        energies[i] = np.mean(frame ** 2)

    # Estimate noise floor (bottom 10% of frames)
    sorted_energies = np.sort(energies)
    noise_floor = np.mean(sorted_energies[:max(1, len(sorted_energies) // 10)])

    # Estimate signal (top 50% of frames)
    signal_level = np.mean(sorted_energies[len(sorted_energies) // 2:])

    if noise_floor < 1e-10:
        return 60.0  # Very clean

    snr = 10 * np.log10(signal_level / noise_floor)
    return float(np.clip(snr, 0, 60))


def detect_silence_regions(audio: np.ndarray, sr: int, threshold: float = 0.01) -> tuple[float, float]:
    """Detect silence at beginning and end of audio."""
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)

    n_frames = (len(audio) - frame_length) // hop_length + 1
    if n_frames < 3:
        return 0.0, 0.0

    # Compute frame RMS
    rms = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        frame = audio[start:end]
        rms[i] = np.sqrt(np.mean(frame ** 2))

    # Find first/last frame above threshold
    voice_frames = rms > threshold

    if not voice_frames.any():
        return len(audio) / sr, 0.0

    first_voice = np.argmax(voice_frames)
    last_voice = len(voice_frames) - np.argmax(voice_frames[::-1]) - 1

    silence_before = first_voice * hop_length / sr
    silence_after = (len(voice_frames) - last_voice - 1) * hop_length / sr

    return float(silence_before), float(silence_after)


def analyze_audio_file(path: Path, source_label: str) -> AudioMetrics:
    """Compute all metrics for a single audio file."""
    # Load audio
    if path.suffix.lower() in ['.wav']:
        try:
            audio, sr = load_wav(path)
        except Exception:
            audio, sr = load_audio_any_format(path)
    else:
        audio, sr = load_audio_any_format(path)

    # Resample to 16kHz if needed
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    metrics = AudioMetrics(
        source=source_label,
        file_path=str(path),
        duration_s=len(audio) / sr,
        sample_rate=sr,
    )

    # Volume metrics
    metrics.rms = float(np.sqrt(np.mean(audio ** 2)))
    metrics.rms_db = float(20 * np.log10(metrics.rms + 1e-10))
    metrics.peak = float(np.abs(audio).max())
    metrics.peak_db = float(20 * np.log10(metrics.peak + 1e-10))
    metrics.dynamic_range_db = metrics.peak_db - metrics.rms_db

    # SRMR (reverb estimation)
    metrics.srmr, metrics.srmr_norm = compute_srmr(audio, sr)

    # Envelope metrics (reverb indicators)
    envelope = compute_envelope_metrics(audio, sr)
    metrics.envelope_variance = envelope['envelope_variance']
    metrics.envelope_kurtosis = envelope['envelope_kurtosis']
    metrics.attack_sharpness = envelope['attack_sharpness']

    # Spectral metrics
    spectral = compute_spectral_metrics(audio, sr)
    metrics.spectral_centroid_mean = spectral['spectral_centroid_mean']
    metrics.spectral_centroid_std = spectral['spectral_centroid_std']
    metrics.spectral_bandwidth_mean = spectral['spectral_bandwidth_mean']
    metrics.spectral_flatness_mean = spectral['spectral_flatness_mean']
    metrics.hf_energy_ratio = spectral['hf_energy_ratio']
    metrics.hf_rolloff_hz = spectral['hf_rolloff_hz']
    metrics.lf_energy_ratio = spectral['lf_energy_ratio']

    # Mel stats
    mel_stats = compute_mel_stats(audio, sr)
    metrics.mel_mean = mel_stats['mel_mean']
    metrics.mel_std = mel_stats['mel_std']
    metrics.mel_min = mel_stats['mel_min']
    metrics.mel_max = mel_stats['mel_max']
    metrics.mel_delta_mean = mel_stats['mel_delta_mean']
    metrics.mel_delta_std = mel_stats['mel_delta_std']

    # Silence detection
    metrics.silence_before_s, metrics.silence_after_s = detect_silence_regions(audio, sr)

    # SNR estimation
    metrics.estimated_snr_db = estimate_snr(audio, sr)

    return metrics


def compute_group_statistics(metrics_list: list[AudioMetrics]) -> dict:
    """Compute summary statistics for a group of recordings."""
    if not metrics_list:
        return {}

    def stats(values):
        arr = np.array(values)
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
        }

    return {
        'count': len(metrics_list),
        'duration_s': stats([m.duration_s for m in metrics_list]),
        'rms': stats([m.rms for m in metrics_list]),
        'rms_db': stats([m.rms_db for m in metrics_list]),
        'srmr': stats([m.srmr for m in metrics_list]),
        'envelope_variance': stats([m.envelope_variance for m in metrics_list]),
        'envelope_kurtosis': stats([m.envelope_kurtosis for m in metrics_list]),
        'attack_sharpness': stats([m.attack_sharpness for m in metrics_list]),
        'spectral_centroid': stats([m.spectral_centroid_mean for m in metrics_list]),
        'spectral_flatness': stats([m.spectral_flatness_mean for m in metrics_list]),
        'hf_energy_ratio': stats([m.hf_energy_ratio for m in metrics_list]),
        'hf_rolloff_hz': stats([m.hf_rolloff_hz for m in metrics_list]),
        'lf_energy_ratio': stats([m.lf_energy_ratio for m in metrics_list]),
        'mel_mean': stats([m.mel_mean for m in metrics_list]),
        'mel_std': stats([m.mel_std for m in metrics_list]),
        'mel_delta_mean': stats([m.mel_delta_mean for m in metrics_list]),
        'mel_delta_std': stats([m.mel_delta_std for m in metrics_list]),
        'silence_before_s': stats([m.silence_before_s for m in metrics_list]),
        'estimated_snr_db': stats([m.estimated_snr_db for m in metrics_list]),
    }


def print_comparison_table(groups: dict[str, list[AudioMetrics]]):
    """Print formatted comparison table."""
    print("\n" + "=" * 100)
    print("DOMAIN COMPARISON: AISHELL vs Pulled Recordings")
    print("=" * 100)

    # Header
    headers = list(groups.keys())
    print(f"\n{'Metric':<30}", end="")
    for h in headers:
        print(f"{h:>20}", end="")
    print()
    print("-" * (30 + 20 * len(headers)))

    # Compute stats
    stats = {name: compute_group_statistics(metrics) for name, metrics in groups.items()}

    # Print key metrics
    metric_rows = [
        ('Sample Count', 'count', None),
        # Reverb indicators
        ('SRMR (reverb)', 'srmr', 'mean'),
        ('Envelope Variance', 'envelope_variance', 'mean'),
        ('Attack Sharpness', 'attack_sharpness', 'mean'),
        # Spectral content
        ('HF Energy Ratio (>5kHz)', 'hf_energy_ratio', 'mean'),
        ('LF Energy Ratio (<300Hz)', 'lf_energy_ratio', 'mean'),
        ('HF Rolloff (Hz)', 'hf_rolloff_hz', 'mean'),
        # Volume
        ('RMS dB', 'rms_db', 'mean'),
        ('RMS dB (std)', 'rms_db', 'std'),
        # Spectral shape
        ('Spectral Centroid', 'spectral_centroid', 'mean'),
        ('Spectral Flatness', 'spectral_flatness', 'mean'),
        # Mel features
        ('Mel Mean', 'mel_mean', 'mean'),
        ('Mel Std', 'mel_std', 'mean'),
        ('Mel Delta Mean', 'mel_delta_mean', 'mean'),
        # Other
        ('Silence Before (s)', 'silence_before_s', 'mean'),
        ('Est. SNR (dB)', 'estimated_snr_db', 'mean'),
    ]

    for label, key, subkey in metric_rows:
        print(f"{label:<30}", end="")
        for name in headers:
            if name in stats and key in stats[name]:
                if subkey is None:
                    val = stats[name][key]
                else:
                    val = stats[name][key][subkey]
                if isinstance(val, float):
                    print(f"{val:>20.4f}", end="")
                else:
                    print(f"{val:>20}", end="")
            else:
                print(f"{'N/A':>20}", end="")
        print()

    # Analysis and recommendations
    print("\n" + "=" * 100)
    print("ANALYSIS & RECOMMENDATIONS")
    print("=" * 100)

    if 'aishell3' in stats and len(stats) > 1:
        aishell_stats = stats['aishell3']

        for name, other_stats in stats.items():
            if name == 'aishell3':
                continue

            print(f"\n{name} vs AISHELL3:")
            print("-" * 50)

            # SRMR comparison (reverb)
            if 'srmr' in aishell_stats and 'srmr' in other_stats:
                srmr_diff = other_stats['srmr']['mean'] - aishell_stats['srmr']['mean']
                if srmr_diff < -1.0:
                    print(f"  [REVERB] SRMR is {abs(srmr_diff):.2f} lower -> more reverberant")
                    print(f"           Recommendation: Add reverb augmentation (RT60 0.3-0.8s)")
                elif srmr_diff > 1.0:
                    print(f"  [REVERB] SRMR is {srmr_diff:.2f} higher -> cleaner than training")

            # Envelope metrics (reverb detection)
            if 'envelope_variance' in aishell_stats and 'envelope_variance' in other_stats:
                env_var_ratio = other_stats['envelope_variance']['mean'] / (aishell_stats['envelope_variance']['mean'] + 1e-10)
                if env_var_ratio < 0.7:
                    print(f"  [REVERB] Envelope variance {(1-env_var_ratio)*100:.0f}% lower -> smoother dynamics")
                    print(f"           Indicates reverb smearing syllable boundaries")

            if 'attack_sharpness' in aishell_stats and 'attack_sharpness' in other_stats:
                attack_ratio = other_stats['attack_sharpness']['mean'] / (aishell_stats['attack_sharpness']['mean'] + 1e-10)
                if attack_ratio < 0.7:
                    print(f"  [REVERB] Attack sharpness {(1-attack_ratio)*100:.0f}% lower -> slower transients")

            # Mel delta (temporal dynamics affected by reverb)
            if 'mel_delta_mean' in aishell_stats and 'mel_delta_mean' in other_stats:
                delta_ratio = other_stats['mel_delta_mean']['mean'] / (aishell_stats['mel_delta_mean']['mean'] + 1e-10)
                if delta_ratio < 0.8:
                    print(f"  [REVERB] Mel delta {(1-delta_ratio)*100:.0f}% lower -> less temporal variation")
                    print(f"           Reverb smooths spectrogram over time")

            # Low-frequency energy (room resonance)
            if 'lf_energy_ratio' in aishell_stats and 'lf_energy_ratio' in other_stats:
                lf_diff = other_stats['lf_energy_ratio']['mean'] - aishell_stats['lf_energy_ratio']['mean']
                if lf_diff > 0.03:
                    print(f"  [ROOM] Low-freq energy {lf_diff*100:.1f}% higher -> possible room resonance")
                    print(f"         Recommendation: Add high-pass filter augmentation (100-200Hz)")

            # High-frequency comparison (codec)
            if 'hf_energy_ratio' in aishell_stats and 'hf_energy_ratio' in other_stats:
                hf_diff = other_stats['hf_energy_ratio']['mean'] - aishell_stats['hf_energy_ratio']['mean']
                if hf_diff < -0.01:
                    print(f"  [CODEC] HF energy {abs(hf_diff)*100:.1f}% lower -> codec compression detected")
                    print(f"          Recommendation: Add low-pass filter augmentation (8-14kHz)")

            # Volume comparison
            if 'rms_db' in aishell_stats and 'rms_db' in other_stats:
                rms_diff = other_stats['rms_db']['mean'] - aishell_stats['rms_db']['mean']
                rms_range = other_stats['rms_db']['max'] - other_stats['rms_db']['min']
                aishell_range = aishell_stats['rms_db']['max'] - aishell_stats['rms_db']['min']

                if abs(rms_diff) > 3 or rms_range > aishell_range + 3:
                    print(f"  [VOLUME] RMS differs by {rms_diff:.1f}dB, range {rms_range:.1f}dB vs {aishell_range:.1f}dB")
                    print(f"           Recommendation: Expand gain augmentation to cover range")

            # Spectral centroid (frequency distribution)
            if 'spectral_centroid' in aishell_stats and 'spectral_centroid' in other_stats:
                cent_diff = other_stats['spectral_centroid']['mean'] - aishell_stats['spectral_centroid']['mean']
                if abs(cent_diff) > 200:
                    direction = "higher" if cent_diff > 0 else "lower"
                    print(f"  [SPECTRAL] Centroid {abs(cent_diff):.0f}Hz {direction}")
                    print(f"             May indicate different mic frequency response")


def generate_plots(groups: dict[str, list[AudioMetrics]], output_dir: Path):
    """Generate comparison visualizations."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    colors = {'aishell3': 'blue', 'pulled_old': 'red', 'pulled_new': 'orange', 'pulled_phone': 'green'}

    # 1. SRMR distribution
    ax = axes[0, 0]
    for name, metrics in groups.items():
        if metrics:
            values = [m.srmr for m in metrics]
            ax.hist(values, bins=15, alpha=0.5, label=name, color=colors.get(name, 'gray'))
    ax.set_xlabel('SRMR (higher = less reverb)')
    ax.set_ylabel('Count')
    ax.set_title('Reverb Estimation (SRMR)')
    ax.legend()
    ax.axvline(x=5, color='black', linestyle='--', alpha=0.5, label='Clean threshold')

    # 2. High-frequency energy ratio
    ax = axes[0, 1]
    for name, metrics in groups.items():
        if metrics:
            values = [m.hf_energy_ratio * 100 for m in metrics]
            ax.hist(values, bins=15, alpha=0.5, label=name, color=colors.get(name, 'gray'))
    ax.set_xlabel('HF Energy Ratio (%)')
    ax.set_ylabel('Count')
    ax.set_title('High-Frequency Content (>8kHz)')
    ax.legend()

    # 3. RMS distribution
    ax = axes[0, 2]
    for name, metrics in groups.items():
        if metrics:
            values = [m.rms_db for m in metrics]
            ax.hist(values, bins=15, alpha=0.5, label=name, color=colors.get(name, 'gray'))
    ax.set_xlabel('RMS (dB)')
    ax.set_ylabel('Count')
    ax.set_title('Volume Level')
    ax.legend()

    # 4. Spectral centroid
    ax = axes[1, 0]
    for name, metrics in groups.items():
        if metrics:
            values = [m.spectral_centroid_mean for m in metrics]
            ax.hist(values, bins=15, alpha=0.5, label=name, color=colors.get(name, 'gray'))
    ax.set_xlabel('Spectral Centroid (Hz)')
    ax.set_ylabel('Count')
    ax.set_title('Spectral Centroid Distribution')
    ax.legend()

    # 5. Mel mean
    ax = axes[1, 1]
    for name, metrics in groups.items():
        if metrics:
            values = [m.mel_mean for m in metrics]
            ax.hist(values, bins=15, alpha=0.5, label=name, color=colors.get(name, 'gray'))
    ax.set_xlabel('Mel Spectrogram Mean')
    ax.set_ylabel('Count')
    ax.set_title('Mel Feature Distribution')
    ax.legend()

    # 6. SRMR vs HF scatter
    ax = axes[1, 2]
    for name, metrics in groups.items():
        if metrics:
            srmr_vals = [m.srmr for m in metrics]
            hf_vals = [m.hf_energy_ratio * 100 for m in metrics]
            ax.scatter(srmr_vals, hf_vals, alpha=0.6, label=name, color=colors.get(name, 'gray'))
    ax.set_xlabel('SRMR (reverb)')
    ax.set_ylabel('HF Energy Ratio (%)')
    ax.set_title('Reverb vs Codec Artifacts')
    ax.legend()

    plt.tight_layout()
    plot_path = output_dir / 'domain_comparison.png'
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare audio domains between AISHELL and pulled recordings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--aishell-dir', type=Path, default=None,
                       help='AISHELL3 dataset directory')
    parser.add_argument('--aishell-samples', type=int, default=50,
                       help='Number of AISHELL samples to analyze (default: 50)')
    parser.add_argument('--pulled-dir', type=Path, action='append', default=[],
                       help='Pulled recordings directory (can specify multiple)')
    parser.add_argument('--output', type=Path, default=None,
                       help='Output JSON report path')
    parser.add_argument('--plot', action='store_true',
                       help='Generate comparison plots')
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent

    # Default directories
    if args.aishell_dir is None:
        args.aishell_dir = base_dir / 'datasets' / 'aishell3' / 'train' / 'wav'

    if not args.pulled_dir:
        args.pulled_dir = [
            base_dir / 'pulled_recordings',
            base_dir / 'pulled_recordings_new',
            base_dir / 'pulled_recordings_phone',
        ]

    groups: dict[str, list[AudioMetrics]] = {
        'aishell3': [],
    }

    # Analyze AISHELL3 samples
    print("=" * 60)
    print("Analyzing AISHELL3 Training Data")
    print("=" * 60)

    if args.aishell_dir.exists():
        count = 0
        for speaker_dir in sorted(args.aishell_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            for wav_file in sorted(speaker_dir.glob('*.wav')):
                if count >= args.aishell_samples:
                    break
                try:
                    metrics = analyze_audio_file(wav_file, 'aishell3')
                    groups['aishell3'].append(metrics)
                    print(f"  {wav_file.name}: SRMR={metrics.srmr:.2f}, HF={metrics.hf_energy_ratio*100:.1f}%, "
                          f"RMS={metrics.rms_db:.1f}dB")
                    count += 1
                except Exception as e:
                    print(f"  ERROR {wav_file.name}: {e}")
            if count >= args.aishell_samples:
                break
        print(f"\nAnalyzed {len(groups['aishell3'])} AISHELL3 samples")
    else:
        print(f"AISHELL3 directory not found: {args.aishell_dir}")

    # Analyze pulled recordings
    for pulled_dir in args.pulled_dir:
        if not pulled_dir.exists():
            continue

        group_name = pulled_dir.name.replace('pulled_recordings', 'pulled').replace('_', '_') or 'pulled'
        if group_name == 'pulled':
            group_name = 'pulled_old'
        groups[group_name] = []

        print(f"\n{'=' * 60}")
        print(f"Analyzing {group_name}: {pulled_dir}")
        print("=" * 60)

        for audio_file in sorted(pulled_dir.glob('*.*')):
            if audio_file.suffix.lower() not in ['.wav', '.m4a', '.mp3', '.aac', '.flac']:
                continue
            if '_resaved' in audio_file.name:
                continue  # Skip resaved versions

            try:
                metrics = analyze_audio_file(audio_file, group_name)
                groups[group_name].append(metrics)
                print(f"  {audio_file.name}: SRMR={metrics.srmr:.2f}, HF={metrics.hf_energy_ratio*100:.1f}%, "
                      f"RMS={metrics.rms_db:.1f}dB, SNR={metrics.estimated_snr_db:.1f}dB")
            except Exception as e:
                print(f"  ERROR {audio_file.name}: {e}")

        print(f"Analyzed {len(groups[group_name])} recordings")

    # Remove empty groups
    groups = {k: v for k, v in groups.items() if v}

    if not groups:
        print("\nNo audio files found to analyze!")
        return 1

    # Print comparison
    print_comparison_table(groups)

    # Generate plots
    if args.plot:
        generate_plots(groups, base_dir)

    # Save report
    if args.output:
        report = {
            'metrics': {
                name: [asdict(m) for m in metrics]
                for name, metrics in groups.items()
            },
            'summary': {
                name: compute_group_statistics(metrics)
                for name, metrics in groups.items()
            },
        }
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
