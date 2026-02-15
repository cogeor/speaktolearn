# Archived Training Scripts

This directory contains historical training scripts that have been superseded by newer versions.

## Files

- **train.py** - Original training script for the initial model architecture
- **train_v3.py** - Training script for SyllablePredictorV3

## Current Training Script

**Use `train_v4.py` in the parent directory** for training the current V4 model architecture.

The V4 model includes:
- CNN front-end with 4x downsampling
- Attention pooling (PMA)
- Rotary Position Embeddings (RoPE)
- Improved architecture with <5M parameters

## Why These Were Archived

These scripts are kept for historical reference but are no longer maintained:
- The V3 model has been superseded by V4 with better architecture
- The original training script is outdated and incompatible with current data sources

For all new training runs, use `train_v4.py`.
