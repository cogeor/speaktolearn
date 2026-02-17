#!/usr/bin/env python3
"""Simple script to pull WAV files from emulator."""
import subprocess
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "pulled_recordings"
OUTPUT_DIR.mkdir(exist_ok=True)

APP = "com.speaktolearn.speak_to_learn"
FILES = ["ts_000001.wav", "ts_000006.wav", "ts_000008.wav", "ts_000009.wav"]

for f in FILES:
    print(f"Pulling {f}...")
    cmd = ["adb", "exec-out", "run-as", APP, "cat", f"app_flutter/recordings/{f}"]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode == 0 and len(result.stdout) > 100:
        out_path = OUTPUT_DIR / f
        out_path.write_bytes(result.stdout)
        print(f"  Saved {len(result.stdout)} bytes to {out_path}")
    else:
        print(f"  Failed: {result.stderr.decode()}")

print(f"\nFiles in {OUTPUT_DIR}:")
for f in OUTPUT_DIR.glob("*.wav"):
    print(f"  {f.name}: {f.stat().st_size} bytes")
