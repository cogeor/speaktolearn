#!/usr/bin/env python3
"""Validate pulled audio files."""
import struct
from pathlib import Path
import numpy as np

RECORDINGS_DIR = Path(__file__).parent.parent / "pulled_recordings"

for wav_path in sorted(RECORDINGS_DIR.glob("*.wav")):
    print(f"\n{'='*60}")
    print(f"File: {wav_path.name}")
    print(f"Size: {wav_path.stat().st_size} bytes")

    with open(wav_path, "rb") as f:
        # Read RIFF header
        riff = f.read(4)
        size = struct.unpack("<I", f.read(4))[0]
        wave = f.read(4)

        print(f"RIFF: {riff}, Size: {size}, WAVE: {wave}")

        # Read format chunk
        fmt = f.read(4)
        fmt_size = struct.unpack("<I", f.read(4))[0]
        audio_format = struct.unpack("<H", f.read(2))[0]
        num_channels = struct.unpack("<H", f.read(2))[0]
        sample_rate = struct.unpack("<I", f.read(4))[0]
        byte_rate = struct.unpack("<I", f.read(4))[0]
        block_align = struct.unpack("<H", f.read(2))[0]
        bits_per_sample = struct.unpack("<H", f.read(2))[0]

        print(f"Format: {audio_format} (1=PCM)")
        print(f"Channels: {num_channels}")
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Bits per sample: {bits_per_sample}")

        # Skip to data chunk
        if fmt_size > 16:
            f.read(fmt_size - 16)

        # Find data chunk
        while True:
            chunk_id = f.read(4)
            if not chunk_id:
                print("ERROR: No data chunk found!")
                break
            chunk_size = struct.unpack("<I", f.read(4))[0]
            if chunk_id == b"data":
                print(f"Data size: {chunk_size} bytes")
                num_samples = chunk_size // (bits_per_sample // 8) // num_channels
                duration = num_samples / sample_rate
                print(f"Samples: {num_samples}")
                print(f"Duration: {duration:.2f}s")

                # Read a few samples to check they're not silent/corrupted
                raw = f.read(min(chunk_size, 1000))
                samples = np.frombuffer(raw, dtype=np.int16)
                print(f"Sample range: [{samples.min()}, {samples.max()}]")
                print(f"RMS: {np.sqrt(np.mean(samples.astype(float)**2)):.1f}")
                break
            else:
                f.read(chunk_size)
