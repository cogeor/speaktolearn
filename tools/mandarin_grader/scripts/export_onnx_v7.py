#!/usr/bin/env python3
"""Export SyllablePredictorV7 to ONNX format.

This script exports a trained V7 PyTorch checkpoint to ONNX format.
V7 uses CTC-based per-frame predictions for syllables and tones.

Key V7 Architecture:
- CNN downsampling: 4x (2 Conv1d layers with stride=2)
- Per-frame output: [batch, time//4, n_syllables+1], [batch, time//4, n_tones+1]
- CTC blank token at index 0 for both outputs
- No position input (predicts all frames in single pass)

Usage:
    # Basic export
    python export_onnx_v7.py --checkpoint checkpoints_v7/best_model.pt --output model_v7.onnx

    # Export with validation
    python export_onnx_v7.py --checkpoint checkpoints_v7/best_model.pt --output model_v7.onnx --validate

    # Export with metadata generation
    python export_onnx_v7.py --checkpoint checkpoints_v7/best_model.pt --output model_v7.onnx --metadata

Model Information:
    - Input: mel [batch, 80, time], audio_mask [batch, time]
    - Output: syllable_logits [batch, time//4, n_syllables+1], tone_logits [batch, time//4, n_tones+1]
    - CTC decoding is done post-ONNX (blank=0, greedy collapse)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from mandarin_grader.model.syllable_predictor_v7 import (
    SyllablePredictorV7,
    SyllablePredictorConfigV7,
)


def load_checkpoint(checkpoint_path: Path, device: str) -> dict:
    """Load checkpoint from file."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        epoch = checkpoint.get("epoch", "unknown")
        print(f"  Loaded checkpoint from epoch: {epoch}")
        if "val_syl_error_rate" in checkpoint:
            print(f"  Validation error rates - Syllable: {checkpoint['val_syl_error_rate']:.4f}, "
                  f"Tone: {checkpoint['val_tone_error_rate']:.4f}")
    else:
        state_dict = checkpoint
        print("  Loaded raw state_dict")

    return state_dict


def infer_config_from_state_dict(state_dict: dict) -> dict:
    """Infer model architecture parameters from state_dict shapes.

    V7-specific inference:
    - d_model: from output of audio_cnn (weight shapes)
    - n_layers: from transformer_layers.* keys
    - n_heads, dim_feedforward: from attention/FFN weights
    - n_syllables: from syllable_head output dimension - 1 (for blank)
    - n_tones: from tone_head output dimension - 1 (for blank)
    """
    params = {}

    # Infer d_model from CNN output or transformer layer
    if "audio_cnn.3.weight" in state_dict:  # Second Conv1d BatchNorm weight
        params["d_model"] = state_dict["audio_cnn.3.weight"].shape[0]
    elif "transformer_layers.0.norm1.weight" in state_dict:
        params["d_model"] = state_dict["transformer_layers.0.norm1.weight"].shape[0]
    else:
        raise ValueError("Cannot infer d_model from state_dict")

    # Infer n_layers from transformer layer indices
    layer_indices = set()
    for key in state_dict.keys():
        if key.startswith("transformer_layers."):
            idx = int(key.split(".")[1])
            layer_indices.add(idx)
    params["n_layers"] = len(layer_indices) if layer_indices else 4

    # Infer n_heads based on d_model
    d_model = params["d_model"]
    if d_model == 192:
        params["n_heads"] = 6
    elif d_model == 384:
        params["n_heads"] = 6
    else:
        for n_heads in [6, 8, 4, 12]:
            if d_model % n_heads == 0 and d_model // n_heads >= 32:
                params["n_heads"] = n_heads
                break
        else:
            params["n_heads"] = 6

    # Infer dim_feedforward from FFN weights
    ff_key = "transformer_layers.0.linear1.weight"
    if ff_key in state_dict:
        params["dim_feedforward"] = state_dict[ff_key].shape[0]
    else:
        params["dim_feedforward"] = d_model * 2

    # Infer vocabulary sizes from output heads (V7 has +1 for CTC blank)
    # syllable_head is Sequential: Linear(d_model, d_model), GELU, Dropout, Linear(d_model, n_syllables+1)
    # The final layer is syllable_head.3
    if "syllable_head.3.weight" in state_dict:
        params["n_syllables"] = state_dict["syllable_head.3.weight"].shape[0] - 1  # Minus blank
    else:
        params["n_syllables"] = 530

    if "tone_head.3.weight" in state_dict:
        params["n_tones"] = state_dict["tone_head.3.weight"].shape[0] - 1  # Minus blank
    else:
        params["n_tones"] = 5

    # Infer max_audio_frames from RoPE cache
    if "rope.cos_cached" in state_dict:
        rope_seq_len = state_dict["rope.cos_cached"].shape[1]
        # max_seq_len = max_audio_frames // 4 + 10
        params["max_audio_frames"] = (rope_seq_len - 10) * 4
    else:
        params["max_audio_frames"] = 1500

    # Attention window cannot be inferred from state_dict
    params["attention_window"] = 32

    return params


def create_model_for_export(checkpoint_path: Path, device: str) -> tuple[SyllablePredictorV7, SyllablePredictorConfigV7, dict]:
    """Create and load model for export."""
    state_dict = load_checkpoint(checkpoint_path, device)

    print("Inferring model architecture from checkpoint...")
    inferred = infer_config_from_state_dict(state_dict)
    print(f"  d_model: {inferred['d_model']}")
    print(f"  n_layers: {inferred['n_layers']}")
    print(f"  n_heads: {inferred['n_heads']}")
    print(f"  dim_feedforward: {inferred['dim_feedforward']}")
    print(f"  n_syllables: {inferred['n_syllables']} (+1 blank = {inferred['n_syllables'] + 1})")
    print(f"  n_tones: {inferred['n_tones']} (+1 blank = {inferred['n_tones'] + 1})")
    print(f"  max_audio_frames: {inferred.get('max_audio_frames', 1500)}")

    config = SyllablePredictorConfigV7(
        d_model=inferred["d_model"],
        n_layers=inferred["n_layers"],
        n_heads=inferred["n_heads"],
        dim_feedforward=inferred["dim_feedforward"],
        n_syllables=inferred["n_syllables"],
        n_tones=inferred["n_tones"],
        max_audio_frames=inferred.get("max_audio_frames", 1500),
        attention_window=inferred.get("attention_window", 32),
        dropout=0.0,
    )

    model = SyllablePredictorV7(config)
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)

    total_params, trainable_params = model.count_parameters()
    print(f"  Model parameters: {total_params:,} total")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")

    return model, config, inferred


def prepare_dummy_inputs(config: SyllablePredictorConfigV7, device: str, time_frames: int = 500) -> tuple:
    """Prepare dummy inputs for ONNX export.

    V7 inputs:
    - mel: [batch, n_mels, time]
    - audio_mask: [batch, time] (bool) - optional but useful for inference
    """
    mel = torch.randn(1, config.n_mels, time_frames, dtype=torch.float32, device=device)
    audio_mask = torch.zeros(1, time_frames, dtype=torch.bool, device=device)

    return (mel, audio_mask)


def export_to_onnx(
    model: SyllablePredictorV7,
    output_path: Path,
    config: SyllablePredictorConfigV7,
    opset_version: int,
    use_fp16: bool,
) -> None:
    """Export model to ONNX format."""
    device = next(model.parameters()).device

    print("Preparing dummy inputs...")
    # Use a moderate size for export
    dummy_inputs = prepare_dummy_inputs(config, device, time_frames=500)

    input_names = ['mel', 'audio_mask']
    output_names = ['syllable_logits', 'tone_logits']

    # V7 has dynamic time dimension (variable length audio)
    dynamic_axes = {
        'mel': {0: 'batch', 2: 'time'},
        'audio_mask': {0: 'batch', 1: 'time'},
        'syllable_logits': {0: 'batch', 1: 'time_out'},  # time_out = time // 4
        'tone_logits': {0: 'batch', 1: 'time_out'},
    }

    print(f"Exporting to ONNX (opset version {opset_version})...")
    print(f"  Input shapes:")
    print(f"    mel: {list(dummy_inputs[0].shape)} -> [batch, 80, time]")
    print(f"    audio_mask: {list(dummy_inputs[1].shape)} -> [batch, time]")
    print(f"  Output shapes (for time=500):")
    print(f"    syllable_logits: [batch, 125, {config.n_syllables + 1}] (time//4, includes blank)")
    print(f"    tone_logits: [batch, 125, {config.n_tones + 1}] (time//4, includes blank)")

    try:
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

        if use_fp16:
            print("\nConverting internal ops to FP16...")
            try:
                import onnx
                from onnxruntime.transformers import float16

                model_fp32 = onnx.load(str(output_path))
                model_fp16 = float16.convert_float_to_float16(
                    model_fp32,
                    keep_io_types=True,
                )
                onnx.save(model_fp16, str(output_path))
                print("  FP16 conversion successful")
            except ImportError:
                print("  Warning: FP16 conversion requires onnxruntime-tools")

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\nExport successful!")
        print(f"  Output: {output_path}")
        print(f"  File size: {file_size_mb:.2f} MB")

    except Exception as e:
        print(f"\nError during export: {e}")
        raise


def generate_model_metadata(
    onnx_path: Path,
    config: SyllablePredictorConfigV7,
    checkpoint_path: Path,
    use_fp16: bool,
    opset_version: int,
    inferred_params: dict | None = None,
) -> dict:
    """Generate model metadata JSON for V7."""
    model_size_bytes = onnx_path.stat().st_size
    model_size_mb = model_size_bytes / (1024 * 1024)

    metadata = {
        "model_info": {
            "name": "SyllablePredictorV7",
            "version": "7.0",
            "architecture": "CNN (4x downsampling) + RoPE Transformer + CTC per-frame outputs",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "checkpoint_source": str(checkpoint_path.name),
            "precision": "fp16" if use_fp16 else "fp32",
            "opset_version": opset_version,
            "model_size_bytes": model_size_bytes,
            "model_size_mb": round(model_size_mb, 2),
        },
        "input_specs": {
            "mel": {
                "shape": ["batch", 80, "time"],
                "dtype": "float32",
                "description": "Full sentence log-mel spectrogram",
                "notes": "Variable time dimension; typically up to 1500 frames (15s at 10ms/frame)"
            },
            "audio_mask": {
                "shape": ["batch", "time"],
                "dtype": "bool",
                "description": "Padding mask for audio frames (True = padded/ignored)",
                "notes": "Must match time dimension of mel input"
            }
        },
        "output_specs": {
            "syllable_logits": {
                "shape": ["batch", "time_out", config.n_syllables + 1],
                "dtype": "float32",
                "description": "Per-frame syllable logits (CTC format)",
                "num_classes": config.n_syllables + 1,
                "notes": "time_out = time // 4 (after CNN downsampling). Index 0 = CTC blank."
            },
            "tone_logits": {
                "shape": ["batch", "time_out", config.n_tones + 1],
                "dtype": "float32",
                "description": "Per-frame tone logits (CTC format)",
                "num_classes": config.n_tones + 1,
                "notes": "time_out = time // 4. Index 0 = CTC blank, 1-5 = tones."
            }
        },
        "preprocessing": {
            "audio": {
                "sample_rate": config.sample_rate,
                "n_mels": config.n_mels,
                "hop_length": config.hop_length,
                "win_length": config.win_length,
                "n_fft": 512,
                "fmin": 0,
                "fmax": 8000,
            },
            "normalization": {
                "method": "log_mel",
                "description": "Apply log to mel spectrogram: log(mel + 1e-9)"
            }
        },
        "ctc_decoding": {
            "blank_index": 0,
            "algorithm": "greedy",
            "description": "Apply softmax, take argmax per frame, collapse repeats, remove blanks",
            "syllable_index_offset": 1,
            "tone_index_offset": 1,
            "notes": "Decoded syllable IDs are 1-indexed (add 1 back for vocab lookup). Tone IDs: 1=tone1, 2=tone2, ..., 5=neutral."
        },
        "pronunciation_grading": {
            "method": "max_probability_alignment",
            "description": "For each target syllable, find the frame with max probability for that syllable",
            "algorithm": [
                "1. Apply softmax to get per-frame probabilities",
                "2. For each target syllable ID in the expected sequence:",
                "   a. Extract probability of target ID at each frame",
                "   b. Take max probability as the score for that syllable",
                "3. Optionally use alignment-aware scoring (monotonic frame order)",
                "4. Combined score = 0.7 * syllable_score + 0.3 * tone_score"
            ],
            "score_range": [0.0, 1.0],
            "notes": "Higher probability = better pronunciation. Use alignment scoring for ordered sequences."
        },
        "vocabulary": {
            "n_syllables": config.n_syllables,
            "n_tones": config.n_tones,
            "syllable_blank": 0,
            "tone_blank": 0,
        },
        "model_architecture": {
            "d_model": config.d_model,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "dim_feedforward": config.dim_feedforward,
            "max_audio_frames": config.max_audio_frames,
            "cnn_downsampling_factor": 4,
            "attention_window": config.attention_window,
            "attention_type": "sliding_window",
        }
    }

    return metadata


def save_model_metadata(metadata: dict, onnx_path: Path) -> Path:
    """Save model metadata JSON alongside ONNX file."""
    metadata_path = onnx_path.parent / (onnx_path.stem + '_metadata.json')

    print(f"\nGenerating model metadata...")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  Metadata saved to: {metadata_path}")
    print(f"  Model version: V7 (CTC per-frame predictions)")

    return metadata_path


def validate_onnx_export(
    pytorch_model: SyllablePredictorV7,
    onnx_path: Path,
    config: SyllablePredictorConfigV7,
    num_test_samples: int = 10,
) -> bool:
    """Validate ONNX export by comparing outputs."""
    print("\n" + "=" * 60)
    print("VALIDATING ONNX EXPORT")
    print("=" * 60)

    try:
        import onnxruntime as ort
    except ImportError:
        print("  Warning: onnxruntime not installed. Skipping validation.")
        return True

    print(f"Loading ONNX model from: {onnx_path}")
    try:
        ort_session = ort.InferenceSession(str(onnx_path))
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return False

    device = next(pytorch_model.parameters()).device
    pytorch_model.eval()

    print(f"\nRunning validation on {num_test_samples} samples...")

    all_passed = True
    max_syllable_error = 0.0
    max_tone_error = 0.0

    for i in range(num_test_samples):
        # Variable length input
        time_frames = np.random.randint(200, 800)
        mel = torch.randn(1, config.n_mels, time_frames, dtype=torch.float32, device=device) * 2.0 - 4.0
        audio_mask = torch.zeros(1, time_frames, dtype=torch.bool, device=device)
        # Simulate some padding
        actual_len = np.random.randint(int(time_frames * 0.7), time_frames)
        audio_mask[:, actual_len:] = True

        # PyTorch inference
        with torch.no_grad():
            pt_syl, pt_tone = pytorch_model(mel, audio_mask)

        # ONNX inference
        ort_inputs = {
            'mel': mel.cpu().numpy(),
            'audio_mask': audio_mask.cpu().numpy(),
        }

        try:
            ort_outputs = ort_session.run(None, ort_inputs)
        except Exception as e:
            print(f"  Sample {i+1}: ONNX inference failed: {e}")
            all_passed = False
            continue

        # Compare
        syl_error = np.abs(pt_syl.cpu().numpy() - ort_outputs[0]).max()
        tone_error = np.abs(pt_tone.cpu().numpy() - ort_outputs[1]).max()

        max_syllable_error = max(max_syllable_error, syl_error)
        max_tone_error = max(max_tone_error, tone_error)

        passed = syl_error < 1e-3 and tone_error < 1e-3
        status = "PASS" if passed else "FAIL"
        out_frames = (time_frames + 3) // 4
        print(f"  Sample {i+1}: time={time_frames}f->out={out_frames}f, "
              f"syl_err={syl_error:.6f}, tone_err={tone_error:.6f} [{status}]")

        if not passed:
            all_passed = False

    print("\n" + "-" * 60)
    print(f"Max syllable error: {max_syllable_error:.8f}")
    print(f"Max tone error: {max_tone_error:.8f}")
    print(f"Overall: {'PASSED' if all_passed else 'FAILED'}")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Export SyllablePredictorV7 to ONNX")

    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--metadata", action="store_true")

    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SyllablePredictorV7 ONNX Export (CTC)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        model, config, inferred = create_model_for_export(args.checkpoint, device)

        export_to_onnx(model, args.output, config, args.opset, args.fp16)

        if args.metadata:
            metadata = generate_model_metadata(
                args.output, config, args.checkpoint, args.fp16, args.opset,
                inferred_params=inferred
            )
            save_model_metadata(metadata, args.output)

        if args.validate:
            if not validate_onnx_export(model, args.output, config):
                print("\nWarning: Validation failed!")
                sys.exit(1)

        print("\n" + "=" * 60)
        print("Export complete!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
