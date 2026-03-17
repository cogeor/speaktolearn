"""Mel-domain augmentation pipeline for training with precomputed spectrograms.

This module provides augmentations that work directly on mel spectrograms,
enabling efficient training with cached/precomputed features while still
benefiting from data augmentation.

Augmentations supported:
- Time stretch (speed variation via interpolation)
- Gain/volume (additive offset in log domain)
- SpecAugment (time and frequency masking)
- Low-frequency attenuation (high-pass filter simulation)
- Spectral noise injection (simulates noise/codec artifacts)
- Temporal smearing (reverb-like effect)

References:
- SpecAugment: Park et al., "SpecAugment: A Simple Data Augmentation Method
  for Automatic Speech Recognition", 2019
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

# Use scipy for efficient convolution
try:
    from scipy.ndimage import convolve1d, zoom
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# -----------------------------------------------------------------------------
# Configuration dataclasses
# -----------------------------------------------------------------------------

@dataclass
class TimeStretchConfig:
    """Time stretch configuration."""
    enabled: bool = True
    prob: float = 1.0
    range: tuple[float, float] = (0.9, 1.1)  # 0.9 = faster, 1.1 = slower


@dataclass
class GainConfig:
    """Gain/volume configuration."""
    enabled: bool = True
    prob: float = 1.0
    db_range: tuple[float, float] = (-6.0, 3.0)


@dataclass
class SpecAugmentConfig:
    """SpecAugment configuration."""
    enabled: bool = True
    prob: float = 1.0
    freq_mask_param: int = 10
    time_mask_param: int = 40
    num_freq_masks: int = 2
    num_time_masks: int = 2
    mask_value: Optional[float] = None  # None = use mean


@dataclass
class LowShelfBoostConfig:
    """Low-shelf boost simulation by boosting lower mel bins.

    Addresses domain gap: pulled recordings have ~2x more LF energy
    due to room resonance and proximity effect.
    """
    enabled: bool = True
    prob: float = 0.5
    cutoff_bin_range: tuple[int, int] = (10, 25)  # Bins 0-25
    boost_db_range: tuple[float, float] = (6.0, 15.0)


@dataclass
class SpectralNoiseConfig:
    """Additive noise in mel domain.

    Addresses domain gap: pulled recordings have ~7x higher spectral
    flatness due to room noise and codec artifacts.
    """
    enabled: bool = True
    prob: float = 0.5
    snr_db_range: tuple[float, float] = (20.0, 40.0)


@dataclass
class TemporalSmearConfig:
    """Reverb-like temporal smearing via exponential decay convolution.

    Addresses domain gap: pulled recordings show reduced temporal
    dynamics (lower mel delta) due to room reverb.
    """
    enabled: bool = True
    prob: float = 0.3
    decay_frames_range: tuple[int, int] = (5, 20)  # ~50-200ms at 100fps
    wet_ratio_range: tuple[float, float] = (0.1, 0.4)


@dataclass
class MelAugmentConfig:
    """Unified configuration for mel-domain augmentation pipeline.

    Augmentations are applied in this order:
    1. Time stretch (speed variation)
    2. Low-frequency attenuation (high-pass simulation)
    3. Temporal smear (reverb simulation)
    4. Gain (volume variation)
    5. Spectral noise (noise/codec simulation)
    6. SpecAugment (time/frequency masking)
    """
    time_stretch: TimeStretchConfig = field(default_factory=TimeStretchConfig)
    gain: GainConfig = field(default_factory=GainConfig)
    spec_augment: SpecAugmentConfig = field(default_factory=SpecAugmentConfig)
    low_shelf_boost: LowShelfBoostConfig = field(default_factory=LowShelfBoostConfig)
    spectral_noise: SpectralNoiseConfig = field(default_factory=SpectralNoiseConfig)
    temporal_smear: TemporalSmearConfig = field(default_factory=TemporalSmearConfig)


# -----------------------------------------------------------------------------
# Augmentation functions
# -----------------------------------------------------------------------------

def apply_time_stretch(
    mel: NDArray[np.float32],
    stretch_factor: float,
) -> NDArray[np.float32]:
    """Apply time stretching via interpolation along time axis.

    Args:
        mel: Mel spectrogram [n_mels, time]
        stretch_factor: >1.0 = longer (slower), <1.0 = shorter (faster)

    Returns:
        Time-stretched mel spectrogram
    """
    if abs(stretch_factor - 1.0) < 0.01:
        return mel

    n_mels, n_time = mel.shape
    new_time = int(n_time * stretch_factor)
    if new_time < 1:
        return mel

    # Use scipy.ndimage.zoom for vectorized interpolation
    if SCIPY_AVAILABLE:
        return zoom(mel, (1.0, stretch_factor), order=1).astype(np.float32)

    # Fallback: numpy interpolation
    old_indices = np.arange(n_time)
    new_indices = np.linspace(0, n_time - 1, new_time)
    stretched = np.zeros((n_mels, new_time), dtype=np.float32)
    for i in range(n_mels):
        stretched[i] = np.interp(new_indices, old_indices, mel[i])
    return stretched


def apply_gain(
    mel: NDArray[np.float32],
    gain_db: float,
) -> NDArray[np.float32]:
    """Apply gain adjustment in mel (log-power) domain.

    In log domain, gain is additive: mel_new = mel + offset
    where offset = gain_db * ln(10) / 10

    Args:
        mel: Mel spectrogram [n_mels, time]
        gain_db: Gain in decibels (negative = quieter, positive = louder)

    Returns:
        Gain-adjusted mel spectrogram
    """
    if abs(gain_db) < 0.1:
        return mel
    mel_offset = np.float32(gain_db * np.log(10) / 10)
    return mel + mel_offset


def apply_spec_augment(
    mel: NDArray[np.float32],
    freq_mask_param: int = 10,
    time_mask_param: int = 40,
    num_freq_masks: int = 2,
    num_time_masks: int = 2,
    mask_value: Optional[float] = None,
) -> NDArray[np.float32]:
    """Apply SpecAugment: frequency and time masking.

    Args:
        mel: Mel spectrogram [n_mels, time]
        freq_mask_param: Maximum width of frequency mask (F parameter)
        time_mask_param: Maximum width of time mask (T parameter)
        num_freq_masks: Number of frequency masks
        num_time_masks: Number of time masks
        mask_value: Fill value for masked regions (None = use mean)

    Returns:
        Augmented mel spectrogram
    """
    mel = mel.copy()
    n_mels, n_time = mel.shape
    fill_value = mask_value if mask_value is not None else mel.mean()

    # Frequency masking
    for _ in range(num_freq_masks):
        f = np.random.randint(0, min(freq_mask_param, n_mels) + 1)
        if f > 0:
            f0 = np.random.randint(0, max(1, n_mels - f))
            mel[f0:f0 + f, :] = fill_value

    # Time masking
    for _ in range(num_time_masks):
        t = np.random.randint(0, min(time_mask_param, n_time) + 1)
        if t > 0:
            t0 = np.random.randint(0, max(1, n_time - t))
            mel[:, t0:t0 + t] = fill_value

    return mel


def apply_low_shelf_boost(
    mel: NDArray[np.float32],
    cutoff_bin: int,
    boost_db: float,
) -> NDArray[np.float32]:
    """Boost low frequencies to simulate proximity effect and room resonance.

    Pulled recordings have ~2x more low-frequency energy due to room
    resonance and proximity to the microphone.

    Args:
        mel: Mel spectrogram [n_mels, time]
        cutoff_bin: Mel bin index for cutoff (bins below are boosted)
        boost_db: Maximum boost in dB at bin 0

    Returns:
        Low-shelf boosted mel spectrogram
    """
    if cutoff_bin <= 0 or boost_db < 0.1:
        return mel

    mel = mel.copy()
    n_mels = mel.shape[0]
    cutoff_bin = min(cutoff_bin, n_mels)

    # Gradient boost: full at bin 0, zero at cutoff_bin
    boost_curve = np.linspace(boost_db, 0, cutoff_bin)
    mel_offset = boost_curve * np.log(10) / 10
    mel[:cutoff_bin, :] += mel_offset[:, np.newaxis]

    return mel


def apply_spectral_noise(
    mel: NDArray[np.float32],
    snr_db: float,
) -> NDArray[np.float32]:
    """Add Gaussian noise in mel domain.

    Pulled recordings have ~7x higher spectral flatness due to room
    noise and AAC codec artifacts. Adding noise in mel domain helps
    the model be robust to these variations.

    Args:
        mel: Mel spectrogram [n_mels, time]
        snr_db: Signal-to-noise ratio in dB (higher = less noise)

    Returns:
        Noisy mel spectrogram
    """
    if snr_db > 60:  # Effectively no noise
        return mel

    # Compute signal power and derive noise power
    signal_power = np.mean(mel ** 2)
    if signal_power < 1e-10:
        return mel

    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(*mel.shape).astype(np.float32) * np.sqrt(noise_power)

    return mel + noise


def apply_temporal_smear(
    mel: NDArray[np.float32],
    decay_frames: int,
    wet_ratio: float,
) -> NDArray[np.float32]:
    """Apply temporal smearing to simulate reverb effects.

    Reverb smooths the spectrogram over time due to late reflections.
    This is approximated by convolving with an exponential decay kernel.

    Args:
        mel: Mel spectrogram [n_mels, time]
        decay_frames: Length of exponential decay kernel in frames
        wet_ratio: Mix ratio (0 = dry, 1 = fully smeared)

    Returns:
        Temporally smeared mel spectrogram
    """
    if decay_frames < 2 or wet_ratio < 0.01:
        return mel

    # Exponential decay kernel
    kernel = np.exp(-np.arange(decay_frames) / (decay_frames / 3))
    kernel = kernel / kernel.sum()

    if SCIPY_AVAILABLE:
        # Efficient 1D convolution along time axis
        smeared = convolve1d(mel, kernel, axis=1, mode='constant', cval=mel.min())
    else:
        # Fallback: numpy convolution per row
        smeared = np.zeros_like(mel)
        for i in range(mel.shape[0]):
            smeared[i] = np.convolve(mel[i], kernel, mode='same')

    # Mix dry and wet signals
    return ((1 - wet_ratio) * mel + wet_ratio * smeared).astype(np.float32)


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def apply_mel_augmentation(
    mel: NDArray[np.float32],
    config: MelAugmentConfig,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.float32]:
    """Apply full mel-domain augmentation pipeline.

    Augmentations are applied in order:
    1. Time stretch (changes temporal structure)
    2. Low-frequency boost (frequency domain)
    3. Temporal smear (reverb-like, needs stable time axis)
    4. Gain (global level adjustment)
    5. Spectral noise (additive)
    6. SpecAugment (masking, applied last)

    Args:
        mel: Mel spectrogram [n_mels, time]
        config: Augmentation configuration
        rng: Random number generator (optional, for reproducibility)

    Returns:
        Augmented mel spectrogram
    """
    mel = mel.copy()

    if rng is None:
        rng = np.random.default_rng()

    # 1. Time stretch
    cfg = config.time_stretch
    if cfg.enabled and rng.random() < cfg.prob:
        factor = rng.uniform(cfg.range[0], cfg.range[1])
        mel = apply_time_stretch(mel, factor)

    # 2. Low-frequency boost
    cfg = config.low_shelf_boost
    if cfg.enabled and rng.random() < cfg.prob:
        cutoff = rng.integers(cfg.cutoff_bin_range[0], cfg.cutoff_bin_range[1] + 1)
        boost = rng.uniform(cfg.boost_db_range[0], cfg.boost_db_range[1])
        mel = apply_low_shelf_boost(mel, cutoff, boost)

    # 3. Temporal smear
    cfg = config.temporal_smear
    if cfg.enabled and rng.random() < cfg.prob:
        decay = rng.integers(cfg.decay_frames_range[0], cfg.decay_frames_range[1] + 1)
        wet = rng.uniform(cfg.wet_ratio_range[0], cfg.wet_ratio_range[1])
        mel = apply_temporal_smear(mel, decay, wet)

    # 4. Gain
    cfg = config.gain
    if cfg.enabled and rng.random() < cfg.prob:
        gain_db = rng.uniform(cfg.db_range[0], cfg.db_range[1])
        mel = apply_gain(mel, gain_db)

    # 5. Spectral noise
    cfg = config.spectral_noise
    if cfg.enabled and rng.random() < cfg.prob:
        snr = rng.uniform(cfg.snr_db_range[0], cfg.snr_db_range[1])
        mel = apply_spectral_noise(mel, snr)

    # 6. SpecAugment (masking last)
    cfg = config.spec_augment
    if cfg.enabled and rng.random() < cfg.prob:
        mel = apply_spec_augment(
            mel,
            freq_mask_param=cfg.freq_mask_param,
            time_mask_param=cfg.time_mask_param,
            num_freq_masks=cfg.num_freq_masks,
            num_time_masks=cfg.num_time_masks,
            mask_value=cfg.mask_value,
        )

    return mel


# -----------------------------------------------------------------------------
# Preset configurations
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Per-utterance normalization utilities
# -----------------------------------------------------------------------------

def apply_utterance_cmvn(mel: NDArray[np.float32]) -> NDArray[np.float32]:
    """Per-utterance Cepstral Mean and Variance Normalization.

    Subtracts the per-mel-bin mean and divides by the per-mel-bin standard
    deviation over the full utterance. This normalizes for channel differences
    such as volume, EQ, and microphone frequency response.

    Args:
        mel: Mel spectrogram [n_mels, time]

    Returns:
        Normalized mel spectrogram with zero mean and unit variance per bin
    """
    mean = np.mean(mel, axis=1, keepdims=True)
    std = np.std(mel, axis=1, keepdims=True)
    return ((mel - mean) / (std + 1e-5)).astype(np.float32)


def apply_pcen(
    mel: NDArray[np.float32],
    alpha: float = 0.98,
    delta: float = 2.0,
    r: float = 0.5,
    eps: float = 1e-6,
    smoothing_coeff: float = 0.025,
) -> NDArray[np.float32]:
    """Per-Channel Energy Normalization (PCEN).

    PCEN is a trainable alternative to log-mel that provides automatic
    gain control and dynamic range compression. It is particularly
    effective for far-field / noisy recordings.

    Formula: PCEN(x) = (x / (M + eps)^alpha + delta)^r - delta^r
    where M is an IIR-smoothed version of x (the AGC reference).

    The mel spectrogram is assumed to be in *linear* (power) domain.
    If your pipeline produces log-mel, exponentiate first:
        mel_linear = np.exp(log_mel)

    Args:
        mel: Linear-scale mel spectrogram [n_mels, time]
        alpha: AGC strength (0 = no AGC, 1 = full gain normalization)
        delta: Bias term that prevents near-zero input from being boosted
        r: Compression exponent (0.5 = square-root compression)
        eps: Stability epsilon added to denominator
        smoothing_coeff: IIR filter coefficient for AGC envelope estimation.
            Higher = faster adaptation (more like CMN), lower = slower (more stable).
            Typical speech value: 0.025 (100ms at 100fps hop rate).

    Returns:
        PCEN-normalized spectrogram [n_mels, time], float32
    """
    n_mels, n_time = mel.shape

    # Compute IIR-smoothed AGC reference M along time axis
    # M[t] = (1 - s) * M[t-1] + s * mel[t]
    M = np.zeros_like(mel)
    M[:, 0] = mel[:, 0]
    for t in range(1, n_time):
        M[:, t] = (1.0 - smoothing_coeff) * M[:, t - 1] + smoothing_coeff * mel[:, t]

    # PCEN: (mel / (M + eps)^alpha + delta)^r - delta^r
    agc = mel / (M + eps) ** alpha
    pcen_out = (agc + delta) ** r - (delta ** r)

    return pcen_out.astype(np.float32)


def get_preset_config(preset: str) -> MelAugmentConfig:
    """Get a preset augmentation configuration.

    Presets:
    - 'none': No augmentation
    - 'light': Conservative augmentation for clean data
    - 'studio': Moderate augmentation (original V6 settings)
    - 'mobile': Aggressive augmentation for mobile deployment

    Args:
        preset: Preset name

    Returns:
        MelAugmentConfig for the preset
    """
    if preset == 'none':
        return MelAugmentConfig(
            time_stretch=TimeStretchConfig(enabled=False),
            gain=GainConfig(enabled=False),
            spec_augment=SpecAugmentConfig(enabled=False),
            low_shelf_boost=LowShelfBoostConfig(enabled=False),
            spectral_noise=SpectralNoiseConfig(enabled=False),
            temporal_smear=TemporalSmearConfig(enabled=False),
        )

    if preset == 'light':
        return MelAugmentConfig(
            time_stretch=TimeStretchConfig(range=(0.95, 1.05)),
            gain=GainConfig(db_range=(-3.0, 3.0)),
            spec_augment=SpecAugmentConfig(freq_mask_param=5, time_mask_param=20),
            low_shelf_boost=LowShelfBoostConfig(enabled=False),
            spectral_noise=SpectralNoiseConfig(enabled=False),
            temporal_smear=TemporalSmearConfig(enabled=False),
        )

    if preset == 'studio':
        # Original V6 settings - works well for clean validation data
        return MelAugmentConfig(
            time_stretch=TimeStretchConfig(range=(0.7, 1.3)),
            gain=GainConfig(db_range=(-12.0, 6.0)),
            spec_augment=SpecAugmentConfig(freq_mask_param=10, time_mask_param=40),
            low_shelf_boost=LowShelfBoostConfig(enabled=False),
            spectral_noise=SpectralNoiseConfig(enabled=False),
            temporal_smear=TemporalSmearConfig(enabled=False),
        )

    if preset == 'mobile':
        # Tuned for mobile phone recordings with room effects
        return MelAugmentConfig(
            time_stretch=TimeStretchConfig(range=(0.8, 1.2)),  # Reduced from 0.7-1.3
            gain=GainConfig(db_range=(-15.0, 6.0)),  # Wider for quieter recordings
            spec_augment=SpecAugmentConfig(freq_mask_param=10, time_mask_param=40, num_freq_masks=3, num_time_masks=3),
            low_shelf_boost=LowShelfBoostConfig(
                enabled=True,
                prob=0.8,
                cutoff_bin_range=(10, 25),
                boost_db_range=(6.0, 15.0),
            ),
            spectral_noise=SpectralNoiseConfig(
                enabled=True,
                prob=0.8,
                snr_db_range=(5.0, 20.0),
            ),
            temporal_smear=TemporalSmearConfig(
                enabled=True,
                prob=0.8,
                decay_frames_range=(15, 50),
                wet_ratio_range=(0.3, 0.7),
            ),
        )

    raise ValueError(f"Unknown preset: {preset}. Use 'none', 'light', 'studio', or 'mobile'")
