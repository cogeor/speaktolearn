#!/usr/bin/env python3
"""Export SyllablePredictorV5 to ONNX format.

This script exports a trained V5 PyTorch checkpoint to ONNX format.
V5 uses full sentence audio + position index instead of pinyin sequence.

Usage:
    # Basic export
    python export_onnx_v5.py --checkpoint checkpoints_v5/best_model.pt --output model_v5.onnx

    # Export with validation
    python export_onnx_v5.py --checkpoint checkpoints_v5/best_model.pt --output model_v5.onnx --validate

    # Export with metadata generation
    python export_onnx_v5.py --checkpoint checkpoints_v5/best_model.pt --output model_v5.onnx --metadata

Model Information:
    - Input: mel [batch, 80, time], position [batch, 1], audio_mask [batch, time]
    - Output: syllable_logits [batch, 532], tone_logits [batch, 5]
    - Architecture: CNN (8x downsampling) + Position Embedding + RoPE Transformer + Longformer Attention + PMA Pooling
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from mandarin_grader.model.syllable_predictor_v5 import (
    SyllablePredictorV5,
    SyllablePredictorConfigV5,
)


def load_checkpoint(checkpoint_path: Path, device: str) -> dict:
    """Load checkpoint from file."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        epoch = checkpoint.get("epoch", "unknown")
        print(f"  Loaded checkpoint from epoch: {epoch}")
        if "val_syl_accuracy" in checkpoint:
            print(f"  Validation accuracy - Syllable: {checkpoint['val_syl_accuracy']:.4f}, "
                  f"Tone: {checkpoint['val_tone_accuracy']:.4f}")
    else:
        state_dict = checkpoint
        print("  Loaded raw state_dict")

    return state_dict


def infer_config_from_state_dict(state_dict: dict) -> dict:
    """Infer model architecture parameters from state_dict shapes."""
    params = {}

    if "audio_type_embed" in state_dict:
        params["d_model"] = state_dict["audio_type_embed"].shape[-1]
    else:
        raise ValueError("Cannot infer d_model: missing audio_type_embed")

    layer_indices = set()
    for key in state_dict.keys():
        if key.startswith("transformer_layers."):
            idx = int(key.split(".")[1])
            layer_indices.add(idx)
    params["n_layers"] = len(layer_indices) if layer_indices else 4

    d_model = params["d_model"]
    if d_model == 192:
        params["n_heads"] = 6
    elif d_model == 384:
        params["n_heads"] = 6
    elif d_model == 480:
        params["n_heads"] = 6
    else:
        for n_heads in [6, 8, 4, 12]:
            if d_model % n_heads == 0 and d_model // n_heads >= 32:
                params["n_heads"] = n_heads
                break
        else:
            params["n_heads"] = 6

    ff_key = "transformer_layers.0.linear1.weight"
    if ff_key in state_dict:
        params["dim_feedforward"] = state_dict[ff_key].shape[0]
    else:
        params["dim_feedforward"] = d_model * 2

    # Infer max_positions from position embedding
    if "position_embed.weight" in state_dict:
        params["max_positions"] = state_dict["position_embed.weight"].shape[0]
    else:
        params["max_positions"] = 30

    # Infer CNN downsampling factor by counting Conv1d layers in audio_cnn
    # Each Conv1d layer with stride=2 halves the sequence length
    cnn_layer_indices = set()
    for key in state_dict.keys():
        if key.startswith("audio_cnn.") and key.endswith(".weight"):
            # audio_cnn.{idx}.weight - extract layer index
            parts = key.split(".")
            if len(parts) >= 3:
                try:
                    idx = int(parts[1])
                    # Check if this is a Conv1d (weight shape: [out_ch, in_ch, kernel])
                    if len(state_dict[key].shape) == 3:
                        cnn_layer_indices.add(idx)
                except ValueError:
                    pass

    # Each Conv1d layer applies stride=2, so total downsampling = 2^num_conv_layers
    num_cnn_layers = len(cnn_layer_indices) if cnn_layer_indices else 3  # Default: 3 layers = 8x
    params["cnn_downsampling_factor"] = 2 ** num_cnn_layers

    # Infer max_audio_frames from RoPE cache (cos_cached shape is [1, seq_len, d_model])
    if "rope.cos_cached" in state_dict:
        # RoPE cache is for downsampled audio: actual_frames = cache_len * downsampling_factor
        rope_seq_len = state_dict["rope.cos_cached"].shape[1]
        params["max_audio_frames"] = rope_seq_len * params["cnn_downsampling_factor"]
    else:
        params["max_audio_frames"] = 1000  # Default

    # Attention window cannot be inferred from state_dict (stored as scalar attribute, not tensor)
    # Use default value from config
    params["attention_window"] = 32

    # Global attention flag - also cannot be inferred, use default
    params["use_global_attention"] = True

    return params


def create_model_for_export(checkpoint_path: Path, device: str) -> tuple[SyllablePredictorV5, SyllablePredictorConfigV5, dict]:
    """Create and load model for export.

    Returns:
        Tuple of (model, config, inferred_params)
    """
    state_dict = load_checkpoint(checkpoint_path, device)

    print("Inferring model architecture from checkpoint...")
    inferred = infer_config_from_state_dict(state_dict)
    print(f"  d_model: {inferred['d_model']}")
    print(f"  n_layers: {inferred['n_layers']}")
    print(f"  n_heads: {inferred['n_heads']}")
    print(f"  dim_feedforward: {inferred['dim_feedforward']}")
    print(f"  max_positions: {inferred['max_positions']}")
    print(f"  max_audio_frames: {inferred.get('max_audio_frames', 1000)}")
    print(f"  cnn_downsampling_factor: {inferred.get('cnn_downsampling_factor', 8)}x")
    print(f"  attention_window: {inferred.get('attention_window', 32)}")

    config = SyllablePredictorConfigV5(
        d_model=inferred["d_model"],
        n_layers=inferred["n_layers"],
        n_heads=inferred["n_heads"],
        dim_feedforward=inferred["dim_feedforward"],
        max_positions=inferred["max_positions"],
        max_audio_frames=inferred.get("max_audio_frames", 1000),
        attention_window=inferred.get("attention_window", 32),
        dropout=0.0,
    )

    model = SyllablePredictorV5(config)
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)

    total_params, trainable_params = model.count_parameters()
    print(f"  Model parameters: {total_params:,} total")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")

    return model, config, inferred


def prepare_dummy_inputs(config: SyllablePredictorConfigV5, device: str) -> tuple:
    """Prepare dummy inputs for ONNX export."""
    # V5 uses full audio up to max_audio_frames
    mel = torch.randn(1, config.n_mels, config.max_audio_frames, dtype=torch.float32, device=device)

    # Position is a single integer [batch, 1]
    position = torch.tensor([[0]], dtype=torch.long, device=device)

    # Audio mask
    audio_mask = torch.zeros(1, config.max_audio_frames, dtype=torch.bool, device=device)

    return (mel, position, audio_mask)


def export_to_onnx(
    model: SyllablePredictorV5,
    output_path: Path,
    config: SyllablePredictorConfigV5,
    opset_version: int,
    use_fp16: bool,
) -> None:
    """Export model to ONNX format.

    Note on attention mask handling:
    The Longformer-style attention mask is created dynamically in the model's forward()
    based on config.attention_window and config.use_global_attention. During ONNX export,
    this mask is baked into the graph for the max_audio_frames sequence length.
    The mask creation uses simple tensor operations that are ONNX-compatible.
    """
    device = next(model.parameters()).device

    print("Preparing dummy inputs...")
    dummy_inputs = prepare_dummy_inputs(config, device)

    # V5 inputs: mel, position, audio_mask (no pinyin_ids or pinyin_mask)
    input_names = ['mel', 'position', 'audio_mask']
    output_names = ['syllable_logits', 'tone_logits']

    dynamic_axes = {
        'mel': {0: 'batch', 2: 'time'},
        'position': {0: 'batch'},
        'audio_mask': {0: 'batch', 1: 'time'},
        'syllable_logits': {0: 'batch'},
        'tone_logits': {0: 'batch'},
    }

    print(f"Exporting to ONNX (opset version {opset_version})...")
    print(f"  Input shapes:")
    print(f"    mel: {list(dummy_inputs[0].shape)} -> [batch, 80, time]")
    print(f"    position: {list(dummy_inputs[1].shape)} -> [batch, 1]")
    print(f"    audio_mask: {list(dummy_inputs[2].shape)} -> [batch, time]")
    print(f"  Note: Longformer attention mask is baked for max sequence length")

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
    config: SyllablePredictorConfigV5,
    checkpoint_path: Path,
    use_fp16: bool,
    opset_version: int,
    inferred_params: dict | None = None,
) -> dict:
    """Generate model metadata JSON."""
    model_size_bytes = onnx_path.stat().st_size
    model_size_mb = model_size_bytes / (1024 * 1024)

    # Get downsampling factor from inferred params or default
    cnn_downsampling_factor = 8
    if inferred_params and "cnn_downsampling_factor" in inferred_params:
        cnn_downsampling_factor = inferred_params["cnn_downsampling_factor"]

    metadata = {
        "model_info": {
            "name": "SyllablePredictorV5",
            "version": "5.0",
            "architecture": "CNN (8x downsampling) + Position Embedding + RoPE Transformer + Longformer Attention + PMA Pooling",
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
                "notes": "time up to ~1000 frames (10s audio at 10ms/frame)"
            },
            "position": {
                "shape": ["batch", 1],
                "dtype": "int64",
                "description": "Syllable position index to predict (0-indexed)",
                "notes": "Single integer indicating which syllable to score"
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
                "shape": ["batch", config.n_syllables],
                "dtype": "float32",
                "description": "Logits for syllable prediction",
                "num_classes": config.n_syllables
            },
            "tone_logits": {
                "shape": ["batch", config.n_tones],
                "dtype": "float32",
                "description": "Logits for tone prediction",
                "num_classes": config.n_tones
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
        "vocabulary": {
            "n_syllables": config.n_syllables,
            "n_tones": config.n_tones,
            "max_positions": config.max_positions,
        },
        "model_architecture": {
            "d_model": config.d_model,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "dim_feedforward": config.dim_feedforward,
            "max_audio_frames": config.max_audio_frames,
            "max_positions": config.max_positions,
            "cnn_downsampling_factor": cnn_downsampling_factor,
            "attention_window": config.attention_window,
            "use_global_attention": config.use_global_attention,
            "attention_type": "longformer",
            "attention_notes": "Local sliding window (size=2*window+1) with global attention on position 0",
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
    print(f"  Model version: V5 (full-sentence, position-based, Longformer attention)")

    return metadata_path


def validate_onnx_export(
    pytorch_model: SyllablePredictorV5,
    onnx_path: Path,
    config: SyllablePredictorConfigV5,
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

    # ONNX export uses fixed input size due to RoPE cache and attention mask being baked in
    fixed_mel_len = config.max_audio_frames
    print(f"  Using fixed mel length: {fixed_mel_len} (matching export)")
    print(f"  Attention window: {config.attention_window}")

    for i in range(num_test_samples):
        # Generate random test input with FIXED size
        mel = torch.randn(1, config.n_mels, fixed_mel_len, dtype=torch.float32, device=device) * 2.0 - 4.0
        position = torch.tensor([[np.random.randint(0, config.max_positions)]], dtype=torch.long, device=device)
        audio_mask = torch.zeros(1, fixed_mel_len, dtype=torch.bool, device=device)
        # Simulate shorter audio with mask
        actual_len = np.random.randint(100, fixed_mel_len)
        audio_mask[:, actual_len:] = True

        # PyTorch inference
        with torch.no_grad():
            pt_syl, pt_tone = pytorch_model(mel, position, audio_mask)

        # ONNX inference
        ort_inputs = {
            'mel': mel.cpu().numpy(),
            'position': position.cpu().numpy(),
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
        print(f"  Sample {i+1}: actual={actual_len}f, pos={position[0,0].item()}, "
              f"syl_err={syl_error:.6f}, tone_err={tone_error:.6f} [{status}]")

        if not passed:
            all_passed = False

    print("\n" + "-" * 60)
    print(f"Max syllable error: {max_syllable_error:.8f}")
    print(f"Max tone error: {max_tone_error:.8f}")
    print(f"Overall: {'PASSED' if all_passed else 'FAILED'}")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Export SyllablePredictorV5 to ONNX")

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
    print("SyllablePredictorV5 ONNX Export")
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
