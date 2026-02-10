# Data Manifest - Mandarin Grader

This document describes all data locations for the mandarin_grader ML pipeline.

## Data Directories (gitignored)

### `data/`
Training and evaluation data.

| Folder | Description |
|--------|-------------|
| `samples/` | Raw audio samples for testing |
| `syllables_v2/` | Syllable vocabulary and metadata |
| `synthetic/` | Generated synthetic audio data |
| `synthetic_train/` | Processed training data from synthetic audio |

### `checkpoints*/`
Model checkpoints from training runs.

| Folder | Model | Status |
|--------|-------|--------|
| `checkpoints/` | SyllableToneModel (original) | Base training |
| `checkpoints_v2/` | ToneClassifier | Deprecated |
| `checkpoints_v3/` | SyllablePredictorV3 | **Current/Latest** |
| `checkpoints_v3_run1/` | SyllablePredictorV3 | Alternative run |
| `checkpoints_v3_normalized/` | SyllablePredictorV3 | With normalization |
| `checkpoints_v3_cmvn/` | SyllablePredictorV3 | With CMVN |
| `checkpoints_v3_volaug/` | SyllablePredictorV3 | With volume augmentation |
| `checkpoints_transformer/` | ToneTransformer | Deprecated |
| `checkpoints_transformer_v2/` | ToneTransformer | Deprecated |

### Other gitignored files
- `*.npy` - NumPy array files (debug examples)
- `mandarin_grader.egg-info/` - Build artifacts
- `uv.lock` - Dependency lockfile

## Current Model Architecture

The latest model is **SyllablePredictorV3** in `mandarin_grader/model/syllable_predictor_v3.py`:
- Autoregressive syllable+tone predictor
- Transformer-based architecture
- Target: <5M parameters for mobile deployment
- Training script: `scripts/train_v3.py`

## How to Regenerate Data

1. **Generate syllable lexicon**: `uv run python scripts/generate_syllable_lexicon_v2.py`
2. **Generate training data**: `uv run python scripts/generate_training_data.py`
3. **Train model**: `uv run python scripts/train_v3.py`
