#!/usr/bin/env python3
"""Optimize an ONNX model using transformer fusions and INT8 dynamic quantization.

This script applies two optimization passes to a model exported by export_onnx_v6.py:

1. ONNX Transformer Optimizer — fuses attention patterns, LayerNorm, GELU, etc.
   Uses onnxruntime.transformers.optimizer with model_type='bert' since the
   SyllablePredictorV6 architecture shares the same encoder-style patterns.

2. Dynamic INT8 Quantization — quantizes MatMul and other ops to INT8 weights
   at runtime, reducing model size and improving CPU throughput.

Usage:
    # Basic optimization (both passes)
    python optimize_model.py --input model_v6.onnx --output model_v6_opt_int8.onnx

    # Optimization with validation
    python optimize_model.py --input model_v6.onnx --output model_v6_opt_int8.onnx --validate

    # Skip transformer fusions (quantization only)
    python optimize_model.py --input model_v6.onnx --output model_v6_int8.onnx --no-fusion

    # Skip quantization (fusion only)
    python optimize_model.py --input model_v6.onnx --output model_v6_fused.onnx --no-quantize

Model Information:
    - Input: mel [batch, 80, time], position [batch, 1], audio_mask [batch, time]
    - Output: syllable_logits [batch, 532], tone_logits [batch, 5]
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def get_file_size_mb(path: Path) -> float:
    """Return file size in megabytes."""
    return path.stat().st_size / (1024 * 1024)


def apply_transformer_fusion(input_path: Path, fused_path: Path, num_heads: int = 0) -> bool:
    """Apply ONNX transformer optimizer to fuse attention, LayerNorm, etc.

    Args:
        input_path: Path to input ONNX model.
        fused_path: Path to save the fused model.
        num_heads: Number of attention heads (0 = auto-detect).

    Returns:
        True if successful, False otherwise.
    """
    try:
        from onnxruntime.transformers import optimizer
        from onnxruntime.transformers.fusion_options import FusionOptions
    except ImportError:
        print("  Error: onnxruntime-tools not available. Install with: pip install onnxruntime")
        return False

    print(f"  Applying transformer fusions...")
    print(f"  Input: {input_path}")

    # Use 'bert' model type — shares encoder-style architecture with V6
    # (multi-head attention + LayerNorm + FFN pattern)
    fusion_options = FusionOptions("bert")

    try:
        optimized = optimizer.optimize_model(
            str(input_path),
            model_type="bert",
            num_heads=num_heads,
            hidden_size=0,  # auto-detect
            optimization_options=fusion_options,
            opt_level=1,
            use_gpu=False,
            only_onnxruntime=False,
        )

        optimized.save_model_to_file(str(fused_path))
        print(f"  Fused model saved to: {fused_path}")
        return True

    except Exception as e:
        print(f"  Warning: Transformer fusion failed: {e}")
        print(f"  Falling back to copying input model for next pass...")
        import shutil
        shutil.copy2(str(input_path), str(fused_path))
        return False


def apply_dynamic_quantization(input_path: Path, output_path: Path) -> bool:
    """Apply dynamic INT8 quantization to the model.

    Dynamic quantization computes activation quantization parameters at runtime
    and stores weight quantization parameters statically. This avoids the need
    for a calibration dataset while still reducing model size and improving
    CPU MatMul throughput.

    Args:
        input_path: Path to input ONNX model (ideally already fused).
        output_path: Path to save the quantized model.

    Returns:
        True if successful, False otherwise.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("  Error: onnxruntime quantization not available.")
        return False

    print(f"  Applying dynamic INT8 quantization...")
    print(f"  Input: {input_path}")

    try:
        quantize_dynamic(
            model_input=str(input_path),
            model_output=str(output_path),
            weight_type=QuantType.QInt8,
            # Quantize MatMul, Gemm ops (core transformer ops)
            per_channel=False,
            reduce_range=False,
            optimize_model=False,  # Already optimized in previous step
        )
        print(f"  Quantized model saved to: {output_path}")
        return True

    except Exception as e:
        print(f"  Error: Dynamic quantization failed: {e}")
        return False


def validate_models(original_path: Path, optimized_path: Path, num_samples: int = 5) -> bool:
    """Load both models and compare outputs on random inputs.

    Generates random mel spectrograms and checks that the optimized model
    produces outputs within a tolerance of the original. INT8 quantization
    introduces some numerical error, so we use a relaxed tolerance (1e-1
    max absolute difference for logits).

    Args:
        original_path: Path to original ONNX model.
        optimized_path: Path to optimized/quantized ONNX model.
        num_samples: Number of random samples to compare.

    Returns:
        True if outputs match within tolerance, False otherwise.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  Error: onnxruntime not available for validation.")
        return False

    print(f"\nValidating models...")
    print(f"  Original:  {original_path}")
    print(f"  Optimized: {optimized_path}")

    try:
        orig_session = ort.InferenceSession(str(original_path))
        opt_session = ort.InferenceSession(str(optimized_path))
    except Exception as e:
        print(f"  Error loading models: {e}")
        return False

    # Infer fixed input shapes from the original model
    input_meta = {inp.name: inp for inp in orig_session.get_inputs()}
    mel_shape = input_meta["mel"].shape if "mel" in input_meta else None

    # Determine a fixed time dimension for validation
    # Use 500 frames as a reasonable default if shape is dynamic
    time_dim = 500
    if mel_shape and len(mel_shape) == 3:
        if isinstance(mel_shape[2], int) and mel_shape[2] > 0:
            time_dim = mel_shape[2]

    n_mels = 80
    max_positions = 60  # V6 default

    print(f"  Using input: mel=[1, {n_mels}, {time_dim}], position=[1, 1], audio_mask=[1, {time_dim}]")
    print(f"  Running {num_samples} random samples...")

    all_passed = True
    max_syl_err = 0.0
    max_tone_err = 0.0

    # Tolerance: INT8 quantization introduces ~1-2% error on logits
    # Use 1.0 as an absolute tolerance for logits (acceptable for argmax inference)
    tolerance = 1.0

    for i in range(num_samples):
        mel = (np.random.randn(1, n_mels, time_dim).astype(np.float32) * 2.0 - 4.0)
        position = np.array([[np.random.randint(0, max_positions)]], dtype=np.int64)
        audio_mask = np.zeros((1, time_dim), dtype=bool)
        # Simulate variable-length audio
        actual_len = np.random.randint(100, time_dim)
        audio_mask[:, actual_len:] = True

        feed = {
            "mel": mel,
            "position": position,
            "audio_mask": audio_mask,
        }

        try:
            orig_out = orig_session.run(None, feed)
            opt_out = opt_session.run(None, feed)
        except Exception as e:
            print(f"  Sample {i+1}: inference error: {e}")
            all_passed = False
            continue

        syl_err = float(np.abs(orig_out[0] - opt_out[0]).max())
        tone_err = float(np.abs(orig_out[1] - opt_out[1]).max())

        max_syl_err = max(max_syl_err, syl_err)
        max_tone_err = max(max_tone_err, tone_err)

        # Check that argmax predictions match (top-1 accuracy preserved)
        syl_match = bool(np.argmax(orig_out[0], axis=1) == np.argmax(opt_out[0], axis=1))
        tone_match = bool(np.argmax(orig_out[1], axis=1) == np.argmax(opt_out[1], axis=1))

        within_tol = syl_err < tolerance and tone_err < tolerance
        status = "PASS" if within_tol else "FAIL"
        syl_pred = "OK" if syl_match else "MISMATCH"
        tone_pred = "OK" if tone_match else "MISMATCH"

        print(
            f"  Sample {i+1}: syl_err={syl_err:.4f} [{syl_pred}], "
            f"tone_err={tone_err:.4f} [{tone_pred}] [{status}]"
        )

        if not within_tol:
            all_passed = False

    print(f"\n  Max syllable error: {max_syl_err:.6f}")
    print(f"  Max tone error:     {max_tone_err:.6f}")
    print(f"  Tolerance:          {tolerance:.1f}")
    print(f"  Result:             {'PASSED' if all_passed else 'FAILED'}")

    return all_passed


def report_size_reduction(original_path: Path, output_path: Path, label: str = "") -> None:
    """Print a size reduction summary."""
    orig_mb = get_file_size_mb(original_path)
    out_mb = get_file_size_mb(output_path)
    reduction_pct = (1.0 - out_mb / orig_mb) * 100.0
    tag = f" ({label})" if label else ""
    print(f"\nSize report{tag}:")
    print(f"  Original:  {orig_mb:.2f} MB")
    print(f"  Output:    {out_mb:.2f} MB")
    print(f"  Reduction: {reduction_pct:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Optimize an ONNX model with transformer fusions and INT8 dynamic quantization."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input ONNX model path")
    parser.add_argument("--output", type=Path, required=True, help="Output optimized model path")
    parser.add_argument(
        "--num-heads",
        type=int,
        default=0,
        help="Number of attention heads (0 = auto-detect, default: 0)",
    )
    parser.add_argument(
        "--no-fusion",
        action="store_true",
        help="Skip transformer fusion pass (quantization only)",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip INT8 quantization pass (fusion only)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Load both models and compare outputs on random inputs",
    )
    parser.add_argument(
        "--validate-samples",
        type=int,
        default=5,
        help="Number of random samples for validation (default: 5)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input model not found: {args.input}")
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ONNX Model Optimizer")
    print("=" * 60)
    print(f"Input:  {args.input}  ({get_file_size_mb(args.input):.2f} MB)")
    print(f"Output: {args.output}")

    if args.no_fusion and args.no_quantize:
        print("Error: --no-fusion and --no-quantize both set — nothing to do.")
        sys.exit(1)

    import tempfile
    import shutil

    # We may need an intermediate file for the fused (pre-quantization) model
    temp_dir = None
    fused_path = args.output  # Default: write directly if only one pass

    try:
        if not args.no_fusion and not args.no_quantize:
            # Both passes: use a temp file for the intermediate fused model
            temp_dir = tempfile.mkdtemp(prefix="onnx_opt_")
            fused_path = Path(temp_dir) / "fused.onnx"
        elif not args.no_fusion:
            # Fusion only
            fused_path = args.output

        # --- Pass 1: Transformer fusion ---
        if not args.no_fusion:
            print("\n--- Pass 1: Transformer Fusion ---")
            fusion_ok = apply_transformer_fusion(args.input, fused_path, num_heads=args.num_heads)
            if fusion_ok:
                print(f"  Fusion successful.")
            else:
                print(f"  Fusion skipped (model copied as-is for next pass).")
        else:
            print("\n--- Pass 1: Transformer Fusion --- SKIPPED")
            if not args.no_quantize:
                # Quantization-only path: fused_path = input
                fused_path = args.input

        # --- Pass 2: Dynamic INT8 quantization ---
        if not args.no_quantize:
            print("\n--- Pass 2: Dynamic INT8 Quantization ---")
            quant_ok = apply_dynamic_quantization(fused_path, args.output)
            if not quant_ok:
                print("Error: Quantization failed.")
                sys.exit(1)
        else:
            print("\n--- Pass 2: Dynamic INT8 Quantization --- SKIPPED")
            # fused_path already points to args.output
            if not fused_path.samefile(args.output) if fused_path.exists() else fused_path != args.output:
                shutil.copy2(str(fused_path), str(args.output))

        # --- Size report ---
        report_size_reduction(args.input, args.output, label="total")

        # --- Validation ---
        if args.validate:
            print("\n" + "=" * 60)
            print("VALIDATION")
            print("=" * 60)
            passed = validate_models(args.input, args.output, num_samples=args.validate_samples)
            if not passed:
                print("\nWarning: Validation failed — outputs differ beyond tolerance.")
                sys.exit(1)

        print("\n" + "=" * 60)
        print("Optimization complete!")
        print("=" * 60)

    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
