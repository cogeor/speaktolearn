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
