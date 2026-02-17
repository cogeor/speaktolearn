"""Data augmentation using RVC voice enhancement.

This module provides an AudioAugmenter class that wraps RVC voice conversion
to generate augmented training data from TTS audio.

The augmenter is designed to be:
- Lazy: generates augmented audio on demand
- Cache-aware: stores enhanced files for reuse
- Graceful: handles missing RVC installation gracefully

Usage:
    augmenter = AudioAugmenter(cache_dir="data/enhanced")
    if augmenter.is_available():
        enhanced_path = augmenter.enhance(input_path, model="ModelName")
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for audio augmentation."""

    model_name: str
    f0_method: Literal["rmvpe", "crepe", "fcpe"] = "rmvpe"
    index_rate: float = 0.3
    protect_rate: float = 0.33
    clean_audio: bool = False


class AudioAugmenter:
    """Audio augmenter using RVC voice conversion.

    This class provides a clean interface for enhancing TTS audio with RVC
    while handling caching and graceful degradation when RVC is unavailable.
    """

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        """Initialize the augmenter.

        Args:
            cache_dir: Directory to cache enhanced audio files.
                       If None, uses a default location.
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / "data" / "enhanced"
        self._cache_dir = Path(cache_dir)
        self._rvc_available: bool | None = None

    @property
    def cache_dir(self) -> Path:
        """Return the cache directory path."""
        return self._cache_dir

    def is_available(self) -> bool:
        """Check if RVC is available for augmentation.

        Returns:
            True if ultimate_rvc is installed and functional.
        """
        if self._rvc_available is not None:
            return self._rvc_available

        try:
            from ultimate_rvc.core.generate.common import convert
            from ultimate_rvc.typing_extra import F0Method, RVCContentType
            self._rvc_available = True
        except ImportError:
            logger.warning("ultimate_rvc not installed - augmentation disabled")
            self._rvc_available = False
        except Exception as e:
            logger.warning(f"RVC initialization failed: {e}")
            self._rvc_available = False

        return self._rvc_available

    def list_models(self) -> list[str]:
        """List available RVC voice models.

        Returns:
            List of model names, or empty list if RVC unavailable.
        """
        if not self.is_available():
            return []

        try:
            from ultimate_rvc.common import VOICE_MODELS_DIR
            models_dir = Path(VOICE_MODELS_DIR)
        except ImportError:
            models_dir = Path.home() / ".ultimate_rvc" / "models" / "voice"

        if not models_dir.exists():
            return []

        models = []
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                pth_files = list(model_dir.glob("*.pth"))
                if pth_files:
                    models.append(model_dir.name)

        return sorted(models)

    def _cache_key(
        self,
        input_path: Path,
        config: AugmentationConfig,
    ) -> str:
        """Generate a cache key for an input/config combination.

        Args:
            input_path: Path to input audio
            config: Augmentation configuration

        Returns:
            Hash string for cache filename
        """
        key_data = f"{input_path.name}:{config.model_name}:{config.f0_method}"
        return hashlib.md5(key_data.encode()).hexdigest()[:12]

    def _cache_path(
        self,
        input_path: Path,
        config: AugmentationConfig,
    ) -> Path:
        """Get the cache path for an enhanced file.

        Args:
            input_path: Path to input audio
            config: Augmentation configuration

        Returns:
            Path where cached file would be stored
        """
        cache_key = self._cache_key(input_path, config)
        model_slug = config.model_name.lower().replace(" ", "_")[:20]
        filename = f"{input_path.stem}_{model_slug}_{cache_key}.wav"
        return self._cache_dir / filename

    def enhance(
        self,
        input_path: Path | str,
        config: AugmentationConfig,
        force: bool = False,
    ) -> Path | None:
        """Enhance an audio file using RVC.

        Args:
            input_path: Path to input audio file
            config: Augmentation configuration
            force: If True, regenerate even if cached

        Returns:
            Path to enhanced audio, or None if enhancement failed.
        """
        if not self.is_available():
            logger.debug("RVC not available, skipping enhancement")
            return None

        input_path = Path(input_path)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return None

        # Check cache
        output_path = self._cache_path(input_path, config)
        if output_path.exists() and not force:
            logger.debug(f"Using cached enhanced audio: {output_path}")
            return output_path

        # Create cache directory
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            from ultimate_rvc.core.generate.common import convert
            from ultimate_rvc.typing_extra import F0Method, RVCContentType

            f0_map = {
                "rmvpe": F0Method.RMVPE,
                "crepe": F0Method.CREPE,
                "fcpe": F0Method.FCPE,
            }

            logger.info(f"Enhancing {input_path.name} with {config.model_name}")

            result_path = convert(
                audio_track=str(input_path),
                directory=str(output_path.parent),
                model_name=config.model_name,
                f0_method=f0_map.get(config.f0_method, F0Method.RMVPE),
                index_rate=config.index_rate,
                protect_rate=config.protect_rate,
                clean_audio=config.clean_audio,
                content_type=RVCContentType.SPEECH,
                make_directory=True,
            )

            # Rename to our cache path if different
            result_path = Path(result_path)
            if result_path != output_path:
                import shutil
                shutil.move(str(result_path), str(output_path))

            logger.info(f"Enhanced audio saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            return None

    def enhance_batch(
        self,
        input_paths: list[Path],
        config: AugmentationConfig,
        force: bool = False,
    ) -> list[tuple[Path, Path | None]]:
        """Enhance multiple audio files.

        Args:
            input_paths: List of input audio paths
            config: Augmentation configuration
            force: If True, regenerate even if cached

        Returns:
            List of (input_path, output_path) tuples.
            output_path is None for failed enhancements.
        """
        results = []
        for i, input_path in enumerate(input_paths, 1):
            logger.info(f"[{i}/{len(input_paths)}] Processing {input_path.name}")
            output_path = self.enhance(input_path, config, force=force)
            results.append((input_path, output_path))

        successful = sum(1 for _, out in results if out is not None)
        logger.info(f"Enhanced {successful}/{len(input_paths)} files")
        return results


# =============================================================================
# Waveform-level augmentations (pitch shifting, formant shifting)
# =============================================================================

import numpy as np
from numpy.typing import NDArray


def pitch_shift(
    audio: NDArray[np.float32],
    semitones: float,
    sr: int = 16000,
) -> NDArray[np.float32]:
    """Shift pitch by given number of semitones without changing duration.

    Uses librosa's pitch_shift which applies a high-quality phase vocoder.

    Args:
        audio: Audio samples [n_samples], float32
        semitones: Pitch shift in semitones (positive = higher, negative = lower)
        sr: Sample rate

    Returns:
        Pitch-shifted audio [n_samples], same length as input
    """
    if abs(semitones) < 0.01:
        return audio

    import librosa

    # librosa.effects.pitch_shift uses n_steps for semitones
    shifted = librosa.effects.pitch_shift(
        y=audio,
        sr=sr,
        n_steps=semitones,
        bins_per_octave=12,
    )

    return shifted.astype(np.float32)


def formant_shift(
    audio: NDArray[np.float32],
    shift_ratio: float,
    sr: int = 16000,
) -> NDArray[np.float32]:
    """Shift formants without changing pitch or duration.

    Formant shifting simulates different vocal tract lengths:
    - shift_ratio > 1.0: Shorter vocal tract (child-like, brighter)
    - shift_ratio < 1.0: Longer vocal tract (deeper, more resonant)

    Implementation: resample -> pitch shift back
    1. Resample audio by shift_ratio (changes both pitch and formants)
    2. Pitch shift back to original pitch (preserves formant change)

    Args:
        audio: Audio samples [n_samples], float32
        shift_ratio: Formant shift ratio (1.0 = no change, 1.1 = +10% formants)
        sr: Sample rate

    Returns:
        Formant-shifted audio [n_samples], same length as input
    """
    if abs(shift_ratio - 1.0) < 0.01:
        return audio

    import librosa

    original_length = len(audio)

    # Step 1: Resample to change both pitch and formants
    # If shift_ratio > 1, we want higher formants
    # Resampling to higher rate then back = pitch down, so we do inverse
    target_sr = int(sr * shift_ratio)
    resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    # Step 2: Pitch shift back to original pitch
    # Resampling by factor R changes pitch by R, so we shift back by -log2(R)*12 semitones
    semitones_to_correct = -12 * np.log2(shift_ratio)
    corrected = librosa.effects.pitch_shift(
        y=resampled,
        sr=target_sr,
        n_steps=semitones_to_correct,
        bins_per_octave=12,
    )

    # Step 3: Resample back to original sample rate and length
    result = librosa.resample(corrected, orig_sr=target_sr, target_sr=sr)

    # Ensure same length as input
    if len(result) > original_length:
        result = result[:original_length]
    elif len(result) < original_length:
        result = np.pad(result, (0, original_length - len(result)))

    return result.astype(np.float32)


def random_pitch_shift(
    audio: NDArray[np.float32],
    max_semitones: float = 2.0,
    sr: int = 16000,
) -> NDArray[np.float32]:
    """Apply random pitch shift within range.

    Args:
        audio: Audio samples [n_samples], float32
        max_semitones: Maximum shift in either direction (default ±2 semitones)
        sr: Sample rate

    Returns:
        Pitch-shifted audio
    """
    semitones = np.random.uniform(-max_semitones, max_semitones)
    return pitch_shift(audio, semitones, sr)


def random_formant_shift(
    audio: NDArray[np.float32],
    max_shift_percent: float = 10.0,
    sr: int = 16000,
) -> NDArray[np.float32]:
    """Apply random formant shift within range.

    Args:
        audio: Audio samples [n_samples], float32
        max_shift_percent: Maximum shift in either direction (default ±10%)
        sr: Sample rate

    Returns:
        Formant-shifted audio
    """
    # Convert percent to ratio: ±10% -> [0.9, 1.1]
    shift_ratio = 1.0 + np.random.uniform(-max_shift_percent, max_shift_percent) / 100.0
    return formant_shift(audio, shift_ratio, sr)
