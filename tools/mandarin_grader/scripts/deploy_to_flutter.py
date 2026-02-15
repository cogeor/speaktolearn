#!/usr/bin/env python3
"""Deploy trained V4 model to Flutter app.

This script automates the model deployment workflow:
1. Find latest checkpoint (or use --checkpoint arg)
2. Export to ONNX using existing export_onnx.py functions
3. Add grade thresholds to metadata JSON
4. Copy model.onnx and metadata.json to Flutter assets
5. Verify Flutter asset manifest
6. Report success with file sizes

Usage:
    # Deploy latest checkpoint from default directory
    python deploy_to_flutter.py

    # Deploy specific checkpoint
    python deploy_to_flutter.py --checkpoint checkpoints_v4/best_model.pt

    # Dry run (don't copy files)
    python deploy_to_flutter.py --dry-run

    # Use FP16 precision
    python deploy_to_flutter.py --fp16
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mandarin_grader.model.syllable_predictor_v4 import GRADE_THRESHOLDS
from scripts.export_onnx import (
    create_model_for_export,
    export_to_onnx,
    generate_model_metadata,
    validate_onnx_export,
)


def find_latest_checkpoint(base_dir: Path) -> Path | None:
    """Find the latest checkpoint from v4 checkpoint directories.

    Looks for checkpoints_v4* directories and finds best_model.pt or latest checkpoint.

    Args:
        base_dir: Base directory to search from (tools/mandarin_grader)

    Returns:
        Path to latest checkpoint or None if not found
    """
    # Find all v4 checkpoint directories
    v4_dirs = sorted(base_dir.glob("checkpoints_v4*"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not v4_dirs:
        return None

    # Check each directory for best_model.pt or latest checkpoint
    for checkpoint_dir in v4_dirs:
        best_model = checkpoint_dir / "best_model.pt"
        if best_model.exists():
            print(f"Found best model in: {checkpoint_dir.name}")
            return best_model

        # Fall back to latest checkpoint_epoch*.pt
        epoch_checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch*.pt"))
        if epoch_checkpoints:
            latest = epoch_checkpoints[-1]
            print(f"Found latest checkpoint in: {checkpoint_dir.name}")
            return latest

    return None


def verify_flutter_manifest(pubspec_path: Path) -> bool:
    """Verify that Flutter pubspec.yaml includes assets/models/ in assets.

    Args:
        pubspec_path: Path to pubspec.yaml

    Returns:
        True if manifest is correct, False otherwise
    """
    if not pubspec_path.exists():
        print(f"Warning: pubspec.yaml not found at {pubspec_path}")
        return False

    with open(pubspec_path, encoding="utf-8") as f:
        content = f.read()

    # Check if assets/models/ is in the assets list
    has_models_asset = "assets/models/" in content

    if not has_models_asset:
        print("\nWarning: assets/models/ not found in pubspec.yaml!")
        print("Add this to your pubspec.yaml under flutter.assets:")
        print("  - assets/models/")
        return False

    return True


def deploy_model(
    checkpoint_path: Path,
    flutter_assets_dir: Path,
    use_fp16: bool = False,
    validate: bool = True,
    dry_run: bool = False,
) -> bool:
    """Deploy model to Flutter assets directory.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        flutter_assets_dir: Path to Flutter assets/models/ directory
        use_fp16: Whether to use FP16 precision
        validate: Whether to validate ONNX export
        dry_run: If True, don't copy files (just report)

    Returns:
        True if deployment successful, False otherwise
    """
    import tempfile
    import torch

    print("=" * 60)
    print("V4 Model Deployment to Flutter")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Target: {flutter_assets_dir}")
    print(f"Precision: {'FP16' if use_fp16 else 'FP32'}")
    print(f"Dry run: {dry_run}")
    print("")

    # Validate checkpoint exists
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return False

    # Create temporary directory for export
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        onnx_path = temp_path / "model.onnx"
        metadata_path = temp_path / "metadata.json"

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}\n")

        try:
            # Create and load model
            print("Loading model from checkpoint...")
            model, config = create_model_for_export(checkpoint_path, device)

            # Export to ONNX
            print("\nExporting to ONNX...")
            export_to_onnx(model, onnx_path, config, opset_version=17, use_fp16=use_fp16)

            # Validate if requested
            if validate:
                print("\nValidating ONNX export...")
                validation_passed = validate_onnx_export(model, onnx_path, config, num_test_samples=15)
                if not validation_passed:
                    print("\nError: Validation failed! Aborting deployment.")
                    return False

            # Generate metadata with grade thresholds
            print("\nGenerating metadata with grade thresholds...")
            metadata = generate_model_metadata(
                onnx_path,
                config,
                checkpoint_path,
                use_fp16,
                opset_version=17,
            )

            # Add grade thresholds to metadata
            metadata["grade_thresholds"] = {
                "description": "Probability thresholds for mapping model confidence to user-facing grades",
                "thresholds": GRADE_THRESHOLDS,
                "mapping": {
                    "bad": f"prob < {GRADE_THRESHOLDS['almost']}",
                    "almost": f"{GRADE_THRESHOLDS['almost']} <= prob < {GRADE_THRESHOLDS['good']}",
                    "good": f"{GRADE_THRESHOLDS['good']} <= prob < {GRADE_THRESHOLDS['easy']}",
                    "easy": f"prob >= {GRADE_THRESHOLDS['easy']}",
                }
            }

            # Save metadata
            import json
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            print(f"  Metadata saved with grade thresholds: {list(GRADE_THRESHOLDS.keys())}")

            # Get file sizes
            onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)
            metadata_size_kb = metadata_path.stat().st_size / 1024

            print("\n" + "=" * 60)
            print("Deployment Files Ready")
            print("=" * 60)
            print(f"  model.onnx: {onnx_size_mb:.2f} MB")
            print(f"  metadata.json: {metadata_size_kb:.2f} KB")
            print("")

            if dry_run:
                print("Dry run mode - skipping file copy")
                print(f"\nWould copy to: {flutter_assets_dir}")
                return True

            # Create Flutter assets directory if needed
            flutter_assets_dir.mkdir(parents=True, exist_ok=True)

            # Copy files to Flutter assets
            dest_onnx = flutter_assets_dir / "model.onnx"
            dest_metadata = flutter_assets_dir / "metadata.json"

            print("Copying files to Flutter assets...")
            shutil.copy2(onnx_path, dest_onnx)
            shutil.copy2(metadata_path, dest_metadata)

            print(f"  Copied: {dest_onnx}")
            print(f"  Copied: {dest_metadata}")

            print("\n" + "=" * 60)
            print("Deployment Successful!")
            print("=" * 60)
            print(f"Model deployed to: {flutter_assets_dir}")
            print(f"  model.onnx ({onnx_size_mb:.2f} MB)")
            print(f"  metadata.json ({metadata_size_kb:.2f} KB)")
            print("\nNext steps:")
            print("  1. Run 'flutter pub get' to update asset manifest")
            print("  2. Rebuild your Flutter app")
            print("")

            return True

        except Exception as e:
            print(f"\nError during deployment: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Deploy V4 model to Flutter app",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Get script directory (tools/mandarin_grader/scripts)
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent
    repo_root = base_dir.parent.parent
    flutter_assets_dir = repo_root / "apps" / "mobile_flutter" / "assets" / "models"
    pubspec_path = repo_root / "apps" / "mobile_flutter" / "pubspec.yaml"

    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to checkpoint file (default: auto-detect latest v4 checkpoint)"
    )
    parser.add_argument(
        "--flutter-assets",
        type=Path,
        default=flutter_assets_dir,
        help=f"Path to Flutter assets/models/ directory (default: {flutter_assets_dir})"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Export in FP16 precision (smaller file size, may reduce accuracy)"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip ONNX validation (faster but not recommended)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run export but don't copy files (for testing)"
    )

    args = parser.parse_args()

    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if not checkpoint_path.is_absolute():
            checkpoint_path = base_dir / checkpoint_path
    else:
        print("Searching for latest V4 checkpoint...")
        checkpoint_path = find_latest_checkpoint(base_dir)
        if not checkpoint_path:
            print("Error: No V4 checkpoint found in checkpoints_v4* directories")
            print("Please specify checkpoint with --checkpoint flag")
            sys.exit(1)

    # Verify Flutter manifest
    if not args.dry_run:
        manifest_ok = verify_flutter_manifest(pubspec_path)
        if not manifest_ok:
            print("\nWarning: pubspec.yaml may need updates")
            response = input("Continue anyway? [y/N]: ")
            if response.lower() != 'y':
                print("Aborted.")
                sys.exit(1)

    # Deploy model
    success = deploy_model(
        checkpoint_path=checkpoint_path,
        flutter_assets_dir=args.flutter_assets,
        use_fp16=args.fp16,
        validate=not args.skip_validation,
        dry_run=args.dry_run,
    )

    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
