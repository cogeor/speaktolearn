#!/usr/bin/env python3
"""Voice enhancement pipeline using RVC for converting robotic TTS to natural speech.

This module provides infrastructure for enhancing TTS-generated audio using
Retrieval-based Voice Conversion (RVC) while preserving timing and prosody.

Pipeline: TTS Audio -> RVC Voice Conversion -> Natural Speech

Usage:
    # Enhance a single file
    python voice_enhancement.py enhance input.wav output.wav --model "model_name"

    # Batch enhance a directory
    python voice_enhancement.py batch input_dir/ output_dir/ --model "model_name"

    # List available models
    python voice_enhancement.py list-models

Voice Models:
    RVC models can be downloaded from:
    - https://huggingface.co/models?other=rvc
    - https://huggingface.co/collections/Nick088/rvc-v2-voice-models

    Place .pth model files in: ~/.ultimate_rvc/models/voice/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Literal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_voice_models_dir() -> Path:
    """Get the directory where RVC voice models are stored."""
    try:
        from ultimate_rvc.common import VOICE_MODELS_DIR
        return Path(VOICE_MODELS_DIR)
    except ImportError:
        # Fallback to default location
        return Path.home() / ".ultimate_rvc" / "models" / "voice"


def list_available_models() -> list[str]:
    """List available RVC voice models."""
    models_dir = get_voice_models_dir()
    if not models_dir.exists():
        return []

    models = []
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            # Check for .pth file
            pth_files = list(model_dir.glob("*.pth"))
            if pth_files:
                models.append(model_dir.name)

    return sorted(models)


def enhance_audio(
    input_path: Path,
    output_path: Path,
    model_name: str,
    f0_method: Literal["rmvpe", "crepe", "fcpe"] = "rmvpe",
    index_rate: float = 0.3,
    protect_rate: float = 0.33,
    clean_audio: bool = False,
) -> Path:
    """Enhance a single audio file using RVC.

    Args:
        input_path: Path to input audio file
        output_path: Path for output audio file
        model_name: Name of the RVC voice model to use
        f0_method: F0 extraction method (rmvpe recommended for speech)
        index_rate: Index rate for voice matching (0-1)
        protect_rate: Consonant protection rate (0-0.5)
        clean_audio: Whether to apply noise reduction

    Returns:
        Path to the enhanced audio file
    """
    from ultimate_rvc.core.generate.common import convert
    from ultimate_rvc.typing_extra import F0Method, RVCContentType

    logger.info(f"Enhancing {input_path.name} with model '{model_name}'")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Map f0 method string to enum
    f0_map = {
        "rmvpe": F0Method.RMVPE,
        "crepe": F0Method.CREPE,
        "fcpe": F0Method.FCPE,
    }

    # Run voice conversion
    result_path = convert(
        audio_track=str(input_path),
        directory=str(output_path.parent),
        model_name=model_name,
        f0_method=f0_map.get(f0_method, F0Method.RMVPE),
        index_rate=index_rate,
        protect_rate=protect_rate,
        clean_audio=clean_audio,
        content_type=RVCContentType.SPEECH,
        make_directory=True,
    )

    # Rename to desired output path if different
    if result_path != output_path:
        import shutil
        shutil.move(str(result_path), str(output_path))

    logger.info(f"Enhanced audio saved to {output_path}")
    return output_path


def enhance_batch(
    input_dir: Path,
    output_dir: Path,
    model_name: str,
    pattern: str = "*.wav",
    **kwargs,
) -> list[Path]:
    """Enhance all audio files in a directory.

    Args:
        input_dir: Directory containing input audio files
        output_dir: Directory for output files
        model_name: Name of the RVC voice model to use
        pattern: Glob pattern for input files
        **kwargs: Additional arguments passed to enhance_audio

    Returns:
        List of paths to enhanced audio files
    """
    input_files = sorted(input_dir.glob(pattern))
    if not input_files:
        logger.warning(f"No files matching '{pattern}' found in {input_dir}")
        return []

    logger.info(f"Found {len(input_files)} files to enhance")

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i, input_path in enumerate(input_files, 1):
        logger.info(f"[{i}/{len(input_files)}] Processing {input_path.name}")
        output_path = output_dir / input_path.name

        try:
            result = enhance_audio(
                input_path=input_path,
                output_path=output_path,
                model_name=model_name,
                **kwargs,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to enhance {input_path.name}: {e}")

    logger.info(f"Enhanced {len(results)}/{len(input_files)} files")
    return results


def download_model(model_url: str, model_name: str | None = None) -> Path:
    """Download an RVC model from a URL (e.g., HuggingFace).

    Args:
        model_url: URL to the model file or HuggingFace model ID
        model_name: Name to save the model as (optional)

    Returns:
        Path to the downloaded model directory
    """
    from ultimate_rvc.core.manage.models import download_voice_model

    logger.info(f"Downloading model from {model_url}")

    result = download_voice_model(
        url=model_url,
        name=model_name,
    )

    logger.info(f"Model downloaded to {result}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Voice enhancement pipeline using RVC"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Enhance single file
    enhance_parser = subparsers.add_parser(
        "enhance", help="Enhance a single audio file"
    )
    enhance_parser.add_argument("input", type=Path, help="Input audio file")
    enhance_parser.add_argument("output", type=Path, help="Output audio file")
    enhance_parser.add_argument(
        "--model", "-m", required=True, help="RVC model name"
    )
    enhance_parser.add_argument(
        "--f0-method", default="rmvpe",
        choices=["rmvpe", "crepe", "fcpe"],
        help="F0 extraction method"
    )
    enhance_parser.add_argument(
        "--index-rate", type=float, default=0.3,
        help="Index rate (0-1)"
    )
    enhance_parser.add_argument(
        "--protect-rate", type=float, default=0.33,
        help="Consonant protection rate (0-0.5)"
    )
    enhance_parser.add_argument(
        "--clean", action="store_true",
        help="Apply noise reduction"
    )

    # Batch enhance
    batch_parser = subparsers.add_parser(
        "batch", help="Enhance all files in a directory"
    )
    batch_parser.add_argument("input_dir", type=Path, help="Input directory")
    batch_parser.add_argument("output_dir", type=Path, help="Output directory")
    batch_parser.add_argument(
        "--model", "-m", required=True, help="RVC model name"
    )
    batch_parser.add_argument(
        "--pattern", default="*.wav", help="File pattern to match"
    )
    batch_parser.add_argument(
        "--f0-method", default="rmvpe",
        choices=["rmvpe", "crepe", "fcpe"],
        help="F0 extraction method"
    )

    # List models
    list_parser = subparsers.add_parser(
        "list-models", help="List available voice models"
    )

    # Download model
    download_parser = subparsers.add_parser(
        "download", help="Download a voice model"
    )
    download_parser.add_argument("url", help="Model URL or HuggingFace ID")
    download_parser.add_argument(
        "--name", help="Name to save the model as"
    )

    args = parser.parse_args()

    if args.command == "enhance":
        enhance_audio(
            input_path=args.input,
            output_path=args.output,
            model_name=args.model,
            f0_method=args.f0_method,
            index_rate=args.index_rate,
            protect_rate=args.protect_rate,
            clean_audio=args.clean,
        )

    elif args.command == "batch":
        enhance_batch(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model_name=args.model,
            pattern=args.pattern,
            f0_method=args.f0_method,
        )

    elif args.command == "list-models":
        models = list_available_models()
        if models:
            print("Available voice models:")
            for model in models:
                print(f"  - {model}")
        else:
            print("No voice models found.")
            print(f"Model directory: {get_voice_models_dir()}")
            print("\nDownload models from:")
            print("  https://huggingface.co/models?other=rvc")

    elif args.command == "download":
        download_model(args.url, args.name)


if __name__ == "__main__":
    main()
