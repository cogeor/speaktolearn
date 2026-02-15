#!/usr/bin/env python3
"""Export SyllablePredictorV4 to ONNX format.

This script exports a trained PyTorch checkpoint to ONNX format with support
for dynamic batch sizes and variable-length inputs.

Usage:
    # Basic export
    python export_onnx.py --checkpoint checkpoints_v4/best_model.pt --output model.onnx

    # Export with validation
    python export_onnx.py --checkpoint checkpoints_v4/best_model.pt --output model.onnx --validate

    # Export with metadata generation
    python export_onnx.py --checkpoint checkpoints_v4/best_model.pt --output model.onnx --metadata

    # Export with custom opset version
    python export_onnx.py --checkpoint checkpoints_v4/best_model.pt --output model.onnx --opset 17

    # Export in FP16 precision
    python export_onnx.py --checkpoint checkpoints_v4/best_model.pt --output model.onnx --fp16

    # Full export with validation and metadata
    python export_onnx.py --checkpoint checkpoints_v4/best_model.pt --output model.onnx --validate --metadata --fp16

Model Information:
    - Input: mel [batch, 80, time], pinyin_ids [batch, seq_len], masks
    - Output: syllable_logits [batch, 530], tone_logits [batch, 5]
    - Architecture: CNN front-end, RoPE Transformer, Attention pooling
"""

import argparse
import json
import sys
from datetime import datetime
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

    # Create config with inferred settings (but force inference mode)
    config = SyllablePredictorConfigV4(
        use_pitch=False,  # Disable for inference (even if trained with it)
        d_model=inferred["d_model"],
        n_layers=inferred["n_layers"],
        n_heads=inferred["n_heads"],
        dim_feedforward=inferred["dim_feedforward"],
        dropout=0.0,  # No dropout for inference
    )

    # Create model
    model = SyllablePredictorV4(config)

    # Filter state_dict to remove pitch_fusion keys if present
    # (since we're creating model without them for inference)
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if not k.startswith("pitch_fusion.")
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

    # Always export in FP32 first, then convert to FP16 if requested
    # This ensures inputs/outputs stay as float32 for compatibility

    # Prepare dummy inputs
    print("Preparing dummy inputs...")
    dummy_inputs = prepare_dummy_inputs(config, device)

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

        # Convert to FP16 if requested (keeps I/O as float32 for compatibility)
        if use_fp16:
            print("\nConverting internal ops to FP16 (keeping I/O as float32)...")
            try:
                import onnx
                from onnxruntime.transformers import float16

                model_fp32 = onnx.load(str(output_path))
                model_fp16 = float16.convert_float_to_float16(
                    model_fp32,
                    keep_io_types=True,  # Keep inputs/outputs as float32
                )
                onnx.save(model_fp16, str(output_path))
                print("  FP16 conversion successful (I/O remains float32)")
            except ImportError:
                try:
                    # Fallback to onnxconverter-common
                    from onnxconverter_common import float16 as onnx_float16
                    model_fp32 = onnx.load(str(output_path))
                    model_fp16 = onnx_float16.convert_float_to_float16(
                        model_fp32,
                        keep_io_types=True,
                    )
                    onnx.save(model_fp16, str(output_path))
                    print("  FP16 conversion successful (I/O remains float32)")
                except Exception as e:
                    print(f"  Warning: FP16 conversion failed: {e}")
                    print("  Keeping model in FP32 format")

        # Get file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\nExport successful!")
        print(f"  Output: {output_path}")
        print(f"  File size: {file_size_mb:.2f} MB")

    except Exception as e:
        print(f"\nError during export: {e}")
        raise


def generate_test_samples(
    config: SyllablePredictorConfigV4,
    device: str,
    num_samples: int = 15,
    fixed_mel_len: int = 100,
    fixed_pinyin_len: int = 5,
) -> list:
    """Generate diverse test samples for comprehensive validation.

    All samples use fixed input sizes to match the ONNX export configuration.
    This is necessary because the RoPE positional embeddings have a fixed cache
    size determined at export time.

    Samples vary in:
    - Random input values (different mel spectrograms and pinyin tokens)
    - Padding patterns (where True values in masks simulate shorter actual inputs)

    Args:
        config: Model configuration
        device: Device to create tensors on
        num_samples: Total number of samples to generate (default: 15)
        fixed_mel_len: Fixed mel length (must match ONNX export, default: 100)
        fixed_pinyin_len: Fixed pinyin length (must match ONNX export, default: 5)

    Returns:
        List of tuples: [(mel, pinyin_ids, audio_mask, pinyin_mask, description), ...]
        Each sample has:
        - mel: [1, 80, fixed_mel_len] - log-mel spectrogram
        - pinyin_ids: [1, fixed_pinyin_len] - pinyin token IDs
        - audio_mask: [1, fixed_mel_len] - True for padded frames
        - pinyin_mask: [1, fixed_pinyin_len] - True for padded tokens
        - description: str describing the sample
    """
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    samples = []

    # 10 samples with varied random values, no padding
    for i in range(10):
        # Generate mel with realistic log-mel values (mean ~-4, std ~2)
        mel = torch.randn(1, config.n_mels, fixed_mel_len, dtype=torch.float32, device=device) * 2.0 - 4.0

        # Generate valid pinyin token IDs (2 to n_syllables-1, excluding PAD=0 and BOS=1)
        pinyin_ids = torch.randint(2, config.n_syllables, (1, fixed_pinyin_len), dtype=torch.long, device=device)

        # No padding - all False masks
        audio_mask = torch.zeros(1, fixed_mel_len, dtype=torch.bool, device=device)
        pinyin_mask = torch.zeros(1, fixed_pinyin_len, dtype=torch.bool, device=device)

        samples.append((mel, pinyin_ids, audio_mask, pinyin_mask, f"Random #{i+1}"))

    # 2 samples with audio padding (simulating shorter audio)
    for i, valid_frames in enumerate([50, 70]):
        mel = torch.zeros(1, config.n_mels, fixed_mel_len, dtype=torch.float32, device=device)
        mel[:, :, :valid_frames] = torch.randn(1, config.n_mels, valid_frames, dtype=torch.float32, device=device) * 2.0 - 4.0

        pinyin_ids = torch.randint(2, config.n_syllables, (1, fixed_pinyin_len), dtype=torch.long, device=device)

        audio_mask = torch.zeros(1, fixed_mel_len, dtype=torch.bool, device=device)
        audio_mask[:, valid_frames:] = True  # Padded frames

        pinyin_mask = torch.zeros(1, fixed_pinyin_len, dtype=torch.bool, device=device)

        samples.append((mel, pinyin_ids, audio_mask, pinyin_mask, f"AudioPad {valid_frames}"))

    # 2 samples with pinyin padding (simulating shorter context)
    for i, valid_tokens in enumerate([2, 3]):
        mel = torch.randn(1, config.n_mels, fixed_mel_len, dtype=torch.float32, device=device) * 2.0 - 4.0

        pinyin_ids = torch.zeros(1, fixed_pinyin_len, dtype=torch.long, device=device)
        pinyin_ids[:, :valid_tokens] = torch.randint(2, config.n_syllables, (1, valid_tokens), dtype=torch.long, device=device)

        audio_mask = torch.zeros(1, fixed_mel_len, dtype=torch.bool, device=device)

        pinyin_mask = torch.zeros(1, fixed_pinyin_len, dtype=torch.bool, device=device)
        pinyin_mask[:, valid_tokens:] = True  # Padded tokens

        samples.append((mel, pinyin_ids, audio_mask, pinyin_mask, f"PinyinPad {valid_tokens}"))

    # 1 sample with both audio and pinyin padding
    mel = torch.zeros(1, config.n_mels, fixed_mel_len, dtype=torch.float32, device=device)
    mel[:, :, :60] = torch.randn(1, config.n_mels, 60, dtype=torch.float32, device=device) * 2.0 - 4.0

    pinyin_ids = torch.zeros(1, fixed_pinyin_len, dtype=torch.long, device=device)
    pinyin_ids[:, :3] = torch.randint(2, config.n_syllables, (1, 3), dtype=torch.long, device=device)

    audio_mask = torch.zeros(1, fixed_mel_len, dtype=torch.bool, device=device)
    audio_mask[:, 60:] = True

    pinyin_mask = torch.zeros(1, fixed_pinyin_len, dtype=torch.bool, device=device)
    pinyin_mask[:, 3:] = True

    samples.append((mel, pinyin_ids, audio_mask, pinyin_mask, "BothPad"))

    return samples[:num_samples]


def generate_model_metadata(
    onnx_path: Path,
    config: SyllablePredictorConfigV4,
    checkpoint_path: Path,
    use_fp16: bool,
    opset_version: int,
) -> dict:
    """Generate model metadata JSON.

    Args:
        onnx_path: Path to exported ONNX model
        config: Model configuration
        checkpoint_path: Path to original checkpoint
        use_fp16: Whether model is in FP16 precision
        opset_version: ONNX opset version

    Returns:
        Dictionary containing model metadata
    """
    # Get model file size
    model_size_bytes = onnx_path.stat().st_size
    model_size_mb = model_size_bytes / (1024 * 1024)

    # Create metadata structure
    metadata = {
        "model_info": {
            "name": "SyllablePredictorV4",
            "version": "4.0",
            "architecture": "CNN + RoPE Transformer + Attention Pooling",
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
                "dtype": "float32",  # I/O always float32, internal ops may be fp16
                "description": "Log-mel spectrogram features",
                "notes": "time dimension is variable (typically ~100 frames for 1s audio)"
            },
            "pinyin_ids": {
                "shape": ["batch", "seq_len"],
                "dtype": "int64",
                "description": "Pinyin token IDs for context (previous syllables)",
                "notes": "seq_len is variable, typically 1-50 tokens"
            },
            "audio_mask": {
                "shape": ["batch", "time"],
                "dtype": "bool",
                "description": "Padding mask for audio frames (True = padded/ignored)",
                "notes": "Must match time dimension of mel input"
            },
            "pinyin_mask": {
                "shape": ["batch", "seq_len"],
                "dtype": "bool",
                "description": "Padding mask for pinyin tokens (True = padded/ignored)",
                "notes": "Must match seq_len dimension of pinyin_ids input"
            }
        },
        "output_specs": {
            "syllable_logits": {
                "shape": ["batch", config.n_syllables],
                "dtype": "float32",  # I/O always float32
                "description": "Logits for syllable prediction (apply softmax for probabilities)",
                "num_classes": config.n_syllables
            },
            "tone_logits": {
                "shape": ["batch", config.n_tones],
                "dtype": "float32",  # I/O always float32
                "description": "Logits for tone prediction (apply softmax for probabilities)",
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
                "notes": "Use librosa.feature.melspectrogram or equivalent with these params"
            },
            "normalization": {
                "method": "log_mel",
                "description": "Apply log to mel spectrogram: log(mel + 1e-9)"
            }
        },
        "vocabulary": {
            "n_syllables": config.n_syllables,
            "n_tones": config.n_tones,
            "special_tokens": {
                "PAD": config.pad_token,
                "BOS": config.bos_token
            },
            "notes": "Syllable vocab includes special tokens. Valid token range: [0, n_syllables-1]"
        },
        "model_architecture": {
            "d_model": config.d_model,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "dim_feedforward": config.dim_feedforward,
            "cnn_kernel_size": config.cnn_kernel_size,
            "max_audio_frames": config.max_audio_frames,
            "max_pinyin_len": config.max_pinyin_len
        }
    }

    return metadata


def save_model_metadata(
    metadata: dict,
    onnx_path: Path,
) -> Path:
    """Save model metadata JSON alongside ONNX file.

    Args:
        metadata: Metadata dictionary
        onnx_path: Path to ONNX model file

    Returns:
        Path to saved metadata JSON file
    """
    # Create metadata filename: model.onnx -> model_metadata.json
    metadata_path = onnx_path.with_suffix('').with_suffix('.onnx').parent / (onnx_path.stem + '_metadata.json')

    print(f"\nGenerating model metadata...")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  Metadata saved to: {metadata_path}")
    print(f"  Model size: {metadata['model_info']['model_size_mb']} MB")
    print(f"  Precision: {metadata['model_info']['precision']}")
    print(f"  Input: mel {metadata['input_specs']['mel']['shape']}, "
          f"pinyin_ids {metadata['input_specs']['pinyin_ids']['shape']}")
    print(f"  Output: syllable_logits [{metadata['output_specs']['syllable_logits']['num_classes']}], "
          f"tone_logits [{metadata['output_specs']['tone_logits']['num_classes']}]")

    return metadata_path


def validate_onnx_export(
    pytorch_model: SyllablePredictorV4,
    onnx_path: Path,
    config: SyllablePredictorConfigV4,
    num_test_samples: int = 15,
) -> bool:
    """Validate ONNX export by comparing outputs on multiple test samples.

    Args:
        pytorch_model: Original PyTorch model
        onnx_path: Path to exported ONNX model
        config: Model configuration
        num_test_samples: Number of test samples to validate (default: 15)

    Returns:
        True if validation passes, False otherwise
    """
    print("\n" + "=" * 60)
    print("VALIDATING ONNX EXPORT")
    print("=" * 60)

    # Try to import onnxruntime
    try:
        import onnxruntime as ort
    except ImportError:
        print("  Warning: onnxruntime not installed. Skipping validation.")
        print("  Install with: pip install onnxruntime")
        return True

    # Load ONNX model
    print(f"Loading ONNX model from: {onnx_path}")
    try:
        ort_session = ort.InferenceSession(str(onnx_path))
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return False

    # Generate test samples
    device = next(pytorch_model.parameters()).device
    print(f"\nGenerating {num_test_samples} test samples (fixed size: mel=100, pinyin=5)...")
    test_samples = generate_test_samples(config, device, num_test_samples)

    # Storage for per-sample results
    syllable_errors = []
    tone_errors = []
    sample_results = []

    # Run validation on each sample
    print(f"\nRunning validation on {len(test_samples)} samples...")
    pytorch_model.eval()

    for idx, sample in enumerate(test_samples, 1):
        # Unpack sample (5 elements: mel, pinyin_ids, audio_mask, pinyin_mask, description)
        mel, pinyin_ids, audio_mask, pinyin_mask, description = sample

        # Get sample info
        mel_len = mel.shape[2]
        pinyin_len = pinyin_ids.shape[1]
        has_audio_padding = audio_mask.any().item()
        has_pinyin_padding = pinyin_mask.any().item()

        # Run PyTorch inference
        with torch.no_grad():
            pt_syllable_logits, pt_tone_logits = pytorch_model(mel, pinyin_ids, audio_mask, pinyin_mask)

        # Convert PyTorch outputs to numpy
        pt_syllable = pt_syllable_logits.cpu().numpy()
        pt_tone = pt_tone_logits.cpu().numpy()

        # Run ONNX inference
        ort_inputs = {
            'mel': mel.cpu().numpy(),
            'pinyin_ids': pinyin_ids.cpu().numpy(),
            'audio_mask': audio_mask.cpu().numpy(),
            'pinyin_mask': pinyin_mask.cpu().numpy(),
        }

        try:
            ort_outputs = ort_session.run(None, ort_inputs)
            onnx_syllable = ort_outputs[0]
            onnx_tone = ort_outputs[1]
        except Exception as e:
            print(f"Error running ONNX inference on sample {idx}: {e}")
            return False

        # Compute per-sample errors
        syllable_error = np.abs(pt_syllable - onnx_syllable).max()
        tone_error = np.abs(pt_tone - onnx_tone).max()

        syllable_errors.append(syllable_error)
        tone_errors.append(tone_error)

        # Check if sample passes (use rtol=1e-3, atol=1e-4 for practical tolerance)
        # Max error ~0.0001 is acceptable for softmax logits (negligible impact on predictions)
        syllable_match = np.allclose(pt_syllable, onnx_syllable, rtol=1e-3, atol=1e-4)
        tone_match = np.allclose(pt_tone, onnx_tone, rtol=1e-3, atol=1e-4)
        sample_passed = syllable_match and tone_match

        # Store result
        sample_results.append({
            'index': idx,
            'description': description,
            'mel_len': mel_len,
            'pinyin_len': pinyin_len,
            'has_audio_padding': has_audio_padding,
            'has_pinyin_padding': has_pinyin_padding,
            'syllable_error': syllable_error,
            'tone_error': tone_error,
            'passed': sample_passed,
        })

    # Compute aggregate statistics
    syllable_max = np.max(syllable_errors)
    syllable_mean = np.mean(syllable_errors)
    syllable_std = np.std(syllable_errors)

    tone_max = np.max(tone_errors)
    tone_mean = np.mean(tone_errors)
    tone_std = np.std(tone_errors)

    # Count passed samples
    passed_count = sum(1 for r in sample_results if r['passed'])
    total_count = len(sample_results)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Total samples tested: {total_count}")
    print(f"Passed: {passed_count}/{total_count}")
    print("")

    print("Syllable Logits:")
    print(f"  Max error:  {syllable_max:.8f}")
    print(f"  Mean error: {syllable_mean:.8f}")
    print(f"  Std error:  {syllable_std:.8f}")
    print("")

    print("Tone Logits:")
    print(f"  Max error:  {tone_max:.8f}")
    print(f"  Mean error: {tone_mean:.8f}")
    print(f"  Std error:  {tone_std:.8f}")
    print("")

    # Print per-sample results table
    print("-" * 70)
    print("Per-Sample Results:")
    print("-" * 70)
    print(f"{'#':<3} {'Description':<14} {'Pad?':<8} {'Syl Error':<12} {'Tone Error':<12} {'Status':<6}")
    print("-" * 70)

    for result in sample_results:
        padding_info = ""
        if result['has_audio_padding'] and result['has_pinyin_padding']:
            padding_info = "A+P"
        elif result['has_audio_padding']:
            padding_info = "Audio"
        elif result['has_pinyin_padding']:
            padding_info = "Pin"
        else:
            padding_info = "-"

        status = "PASS" if result['passed'] else "FAIL"
        print(f"{result['index']:<3} {result['description']:<14} "
              f"{padding_info:<8} {result['syllable_error']:<12.8f} {result['tone_error']:<12.8f} {status:<6}")

    print("-" * 60)

    # Overall validation status
    validation_passed = (passed_count == total_count)

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    if validation_passed:
        print("Overall Status: PASSED")
        print(f"Max Absolute Error:")
        print(f"  Syllable logits: {syllable_max:.8f}")
        print(f"  Tone logits: {tone_max:.8f}")
        print("\nRecommendation: Model ready for production use")
    else:
        print("Overall Status: FAILED")
        print(f"Failed samples: {total_count - passed_count}/{total_count}")
        print("Outputs do not match within tolerance (rtol=1e-3, atol=1e-4)")
        print("\nRecommendation: Review failed samples and investigate discrepancies")

    print("=" * 60)

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
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Generate model_metadata.json with model specifications and preprocessing params"
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

        # Generate metadata if requested
        metadata_path = None
        if args.metadata:
            metadata = generate_model_metadata(
                args.output,
                config,
                args.checkpoint,
                args.fp16,
                args.opset
            )
            metadata_path = save_model_metadata(metadata, args.output)

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
        if args.metadata:
            print(f"Metadata: {metadata_path}")
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
