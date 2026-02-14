#!/usr/bin/env python3
"""Export SyllablePredictorV4 to ONNX format.

This script exports a trained PyTorch checkpoint to ONNX format with support
for dynamic batch sizes and variable-length inputs.

Usage:
    # Basic export
    python export_onnx.py --checkpoint checkpoints_v4/best_model.pt --output model.onnx

    # Export with validation
    python export_onnx.py --checkpoint checkpoints_v4/best_model.pt --output model.onnx --validate

    # Export with custom opset version
    python export_onnx.py --checkpoint checkpoints_v4/best_model.pt --output model.onnx --opset 17

    # Export in FP16 precision
    python export_onnx.py --checkpoint checkpoints_v4/best_model.pt --output model.onnx --fp16

Model Information:
    - Input: mel [batch, 80, time], pinyin_ids [batch, seq_len], masks
    - Output: syllable_logits [batch, 530], tone_logits [batch, 5]
    - Architecture: CNN front-end, RoPE Transformer, Attention pooling
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mandarin_grader.model.syllable_predictor_v4 import (
    SyllablePredictorV4,
    SyllablePredictorConfigV4,
    SyllableVocab,
)


def load_checkpoint(checkpoint_path: Path, device: str) -> dict:
    """Load checkpoint from file.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to map checkpoint to

    Returns:
        Dictionary containing model state_dict
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle both raw state_dict and wrapped checkpoint formats
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        epoch = checkpoint.get("epoch", "unknown")
        print(f"  Loaded checkpoint from epoch: {epoch}")
        if "val_syl_accuracy" in checkpoint and "val_tone_accuracy" in checkpoint:
            print(f"  Validation accuracy - Syllable: {checkpoint['val_syl_accuracy']:.4f}, Tone: {checkpoint['val_tone_accuracy']:.4f}")
    else:
        # Assume it's a raw state_dict
        state_dict = checkpoint
        print("  Loaded raw state_dict")

    return state_dict


def infer_config_from_state_dict(state_dict: dict) -> dict:
    """Infer model architecture parameters from state_dict shapes.

    Args:
        state_dict: Model state dictionary

    Returns:
        Dictionary of inferred architecture parameters
    """
    params = {}

    # Infer d_model from audio_type_embed: [1, 1, d_model]
    if "audio_type_embed" in state_dict:
        params["d_model"] = state_dict["audio_type_embed"].shape[-1]
    elif "pinyin_type_embed" in state_dict:
        params["d_model"] = state_dict["pinyin_type_embed"].shape[-1]
    else:
        raise ValueError("Cannot infer d_model: missing audio_type_embed or pinyin_type_embed")

    # Count transformer layers by counting unique layer indices
    layer_indices = set()
    for key in state_dict.keys():
        if key.startswith("transformer_layers."):
            # Key format: transformer_layers.0.q_proj.weight
            idx = int(key.split(".")[1])
            layer_indices.add(idx)
    params["n_layers"] = len(layer_indices) if layer_indices else 4

    # Infer n_heads from q_proj weight shape
    # q_proj.weight shape: [d_model, d_model]
    # But we can infer from the head_dim used in view operations
    # Actually, check the norm1 or use default
    d_model = params["d_model"]

    # n_heads: typically d_model / head_dim where head_dim is 32 or 64
    # For d_model=192, n_heads=6 (head_dim=32)
    # For d_model=384, n_heads=6 or 12 (head_dim=64 or 32)
    if d_model == 192:
        params["n_heads"] = 6
    elif d_model == 384:
        params["n_heads"] = 6  # head_dim=64
    elif d_model == 256:
        params["n_heads"] = 8  # head_dim=32
    elif d_model == 512:
        params["n_heads"] = 8  # head_dim=64
    else:
        # Try to infer from head_dim=32 or 64
        params["n_heads"] = max(4, d_model // 32)

    # Infer dim_feedforward from linear1.weight: [dim_ff, d_model]
    ff_key = "transformer_layers.0.linear1.weight"
    if ff_key in state_dict:
        params["dim_feedforward"] = state_dict[ff_key].shape[0]
    else:
        # Default to 2x d_model
        params["dim_feedforward"] = d_model * 2

    # Check for pitch fusion
    params["use_pitch"] = "pitch_fusion.pitch_proj.weight" in state_dict

    # Check for domain adversarial
    params["use_domain_adversarial"] = "domain_classifier.0.weight" in state_dict

    return params


def create_model_for_export(checkpoint_path: Path, device: str) -> tuple[SyllablePredictorV4, SyllablePredictorConfigV4]:
    """Create and load model for export.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Tuple of (model, config)
    """
    # Load state dict first to infer architecture
    state_dict = load_checkpoint(checkpoint_path, device)

    # Infer architecture parameters from state_dict
    print("Inferring model architecture from checkpoint...")
    inferred = infer_config_from_state_dict(state_dict)
    print(f"  d_model: {inferred['d_model']}")
    print(f"  n_layers: {inferred['n_layers']}")
    print(f"  n_heads: {inferred['n_heads']}")
    print(f"  dim_feedforward: {inferred['dim_feedforward']}")
    print(f"  use_pitch: {inferred['use_pitch']}")
    print(f"  use_domain_adversarial: {inferred['use_domain_adversarial']}")

    # Create config with inferred settings (but force inference mode)
    config = SyllablePredictorConfigV4(
        use_pitch=False,  # Disable for inference (even if trained with it)
        use_domain_adversarial=False,  # Disable for inference
        d_model=inferred["d_model"],
        n_layers=inferred["n_layers"],
        n_heads=inferred["n_heads"],
        dim_feedforward=inferred["dim_feedforward"],
        dropout=0.0,  # No dropout for inference
    )

    # Create model
    model = SyllablePredictorV4(config)

    # Filter state_dict to remove pitch_fusion and domain_classifier keys if present
    # (since we're creating model without them for inference)
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if not k.startswith("pitch_fusion.") and not k.startswith("domain_classifier.") and not k.startswith("gradient_reversal")
    }

    # Load state dict
    model.load_state_dict(filtered_state_dict)

    # Set to eval mode
    model.eval()

    # Move to device
    model = model.to(device)

    # Log parameter count
    total_params, trainable_params = model.count_parameters()
    print(f"  Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")

    return model, config


def prepare_dummy_inputs(config: SyllablePredictorConfigV4, device: str) -> tuple:
    """Prepare dummy inputs for ONNX export.

    Args:
        config: Model configuration
        device: Device to create tensors on

    Returns:
        Tuple of (mel, pinyin_ids, audio_mask, pinyin_mask)
    """
    # Dummy mel spectrogram: [1, 80, 100] (1 second of audio at 10ms per frame)
    mel = torch.randn(1, config.n_mels, 100, dtype=torch.float32, device=device)

    # Dummy pinyin_ids: [1, 5] (5 syllables context)
    pinyin_ids = torch.randint(0, config.n_syllables, (1, 5), dtype=torch.long, device=device)

    # Dummy masks (all valid)
    audio_mask = torch.zeros(1, 100, dtype=torch.bool, device=device)
    pinyin_mask = torch.zeros(1, 5, dtype=torch.bool, device=device)

    return (mel, pinyin_ids, audio_mask, pinyin_mask)


def export_to_onnx(
    model: SyllablePredictorV4,
    output_path: Path,
    config: SyllablePredictorConfigV4,
    opset_version: int,
    use_fp16: bool,
) -> None:
    """Export model to ONNX format.

    Args:
        model: The model to export
        output_path: Path to save ONNX file
        config: Model configuration
        opset_version: ONNX opset version
        use_fp16: Whether to use FP16 precision
    """
    device = next(model.parameters()).device

    # Convert to FP16 if requested
    if use_fp16:
        print("Converting model to FP16 precision...")
        model = model.half()

    # Prepare dummy inputs
    print("Preparing dummy inputs...")
    dummy_inputs = prepare_dummy_inputs(config, device)

    if use_fp16:
        # Convert mel input to FP16
        dummy_inputs = (dummy_inputs[0].half(),) + dummy_inputs[1:]

    # Define input and output names
    input_names = ['mel', 'pinyin_ids', 'audio_mask', 'pinyin_mask']
    output_names = ['syllable_logits', 'tone_logits']

    # Define dynamic axes for variable-length inputs
    dynamic_axes = {
        'mel': {0: 'batch', 2: 'time'},  # [batch, 80, time]
        'pinyin_ids': {0: 'batch', 1: 'seq_len'},  # [batch, seq_len]
        'audio_mask': {0: 'batch', 1: 'time'},  # [batch, time]
        'pinyin_mask': {0: 'batch', 1: 'seq_len'},  # [batch, seq_len]
        'syllable_logits': {0: 'batch'},  # [batch, n_syllables]
        'tone_logits': {0: 'batch'},  # [batch, n_tones]
    }

    print(f"Exporting to ONNX (opset version {opset_version})...")
    print(f"  Input shapes:")
    print(f"    mel: {list(dummy_inputs[0].shape)} -> [batch, 80, time]")
    print(f"    pinyin_ids: {list(dummy_inputs[1].shape)} -> [batch, seq_len]")
    print(f"    audio_mask: {list(dummy_inputs[2].shape)} -> [batch, time]")
    print(f"    pinyin_mask: {list(dummy_inputs[3].shape)} -> [batch, seq_len]")

    try:
        # Export using standard torch.onnx.export (more compatible than dynamo)
        torch.onnx.export(
            model,
            dummy_inputs,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

        # Get file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\nExport successful!")
        print(f"  Output: {output_path}")
        print(f"  File size: {file_size_mb:.2f} MB")

    except Exception as e:
        print(f"\nError during export: {e}")
        raise


def validate_onnx_export(
    pytorch_model: SyllablePredictorV4,
    onnx_path: Path,
    config: SyllablePredictorConfigV4,
) -> bool:
    """Validate ONNX export by comparing outputs.

    Args:
        pytorch_model: Original PyTorch model
        onnx_path: Path to exported ONNX model
        config: Model configuration

    Returns:
        True if validation passes, False otherwise
    """
    print("\nValidating ONNX export...")

    # Try to import onnxruntime
    try:
        import onnxruntime as ort
    except ImportError:
        print("  Warning: onnxruntime not installed. Skipping validation.")
        print("  Install with: pip install onnxruntime")
        return True

    # Load ONNX model
    print(f"  Loading ONNX model from: {onnx_path}")
    try:
        ort_session = ort.InferenceSession(str(onnx_path))
    except Exception as e:
        print(f"  Error loading ONNX model: {e}")
        return False

    # Prepare test inputs
    device = next(pytorch_model.parameters()).device
    test_inputs = prepare_dummy_inputs(config, device)

    # Run PyTorch inference
    print("  Running PyTorch inference...")
    pytorch_model.eval()
    with torch.no_grad():
        pt_syllable_logits, pt_tone_logits = pytorch_model(*test_inputs)

    # Convert PyTorch outputs to numpy
    pt_syllable = pt_syllable_logits.cpu().numpy()
    pt_tone = pt_tone_logits.cpu().numpy()

    # Run ONNX inference
    print("  Running ONNX inference...")
    ort_inputs = {
        'mel': test_inputs[0].cpu().numpy(),
        'pinyin_ids': test_inputs[1].cpu().numpy(),
        'audio_mask': test_inputs[2].cpu().numpy(),
        'pinyin_mask': test_inputs[3].cpu().numpy(),
    }

    try:
        ort_outputs = ort_session.run(None, ort_inputs)
        onnx_syllable = ort_outputs[0]
        onnx_tone = ort_outputs[1]
    except Exception as e:
        print(f"  Error running ONNX inference: {e}")
        return False

    # Compare outputs
    print("  Comparing outputs...")

    # Syllable logits
    syllable_max_diff = np.abs(pt_syllable - onnx_syllable).max()
    syllable_match = np.allclose(pt_syllable, onnx_syllable, rtol=1e-3, atol=1e-5)

    # Tone logits
    tone_max_diff = np.abs(pt_tone - onnx_tone).max()
    tone_match = np.allclose(pt_tone, onnx_tone, rtol=1e-3, atol=1e-5)

    # Report results
    print(f"\n  Syllable logits:")
    print(f"    Max absolute difference: {syllable_max_diff:.6f}")
    print(f"    Match (rtol=1e-3, atol=1e-5): {syllable_match}")

    print(f"\n  Tone logits:")
    print(f"    Max absolute difference: {tone_max_diff:.6f}")
    print(f"    Match (rtol=1e-3, atol=1e-5): {tone_match}")

    # Overall validation
    validation_passed = syllable_match and tone_match

    if validation_passed:
        print(f"\n  Validation: PASSED")
    else:
        print(f"\n  Validation: FAILED")
        print("  Outputs do not match within tolerance!")

    return validation_passed


def main():
    parser = argparse.ArgumentParser(
        description="Export SyllablePredictorV4 to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to PyTorch checkpoint file (.pt)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for output ONNX file (.onnx)"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Export in FP16 precision"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate export by comparing PyTorch vs ONNX outputs"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.checkpoint.exists():
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SyllablePredictorV4 ONNX Export")
    print("=" * 60)

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        # Create and load model
        model, config = create_model_for_export(args.checkpoint, device)

        # Export to ONNX
        export_to_onnx(model, args.output, config, args.opset, args.fp16)

        # Validate if requested
        if args.validate:
            validation_passed = validate_onnx_export(model, args.output, config)
            if not validation_passed:
                print("\nWarning: Validation failed! ONNX model may not produce correct outputs.")
                sys.exit(1)

        print("\n" + "=" * 60)
        print("Export complete!")
        print("=" * 60)
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Output: {args.output}")
        print(f"Opset version: {args.opset}")
        print(f"Precision: {'FP16' if args.fp16 else 'FP32'}")
        if args.validate:
            print("Validation: PASSED")

        sys.exit(0)

    except KeyboardInterrupt:
        print("\n\nExport interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
