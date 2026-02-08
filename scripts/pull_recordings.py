#!/usr/bin/env python3
"""Pull audio recordings from Android emulator for Python scoring pipeline testing.

Usage:
    python scripts/pull_recordings.py              # Pull all recordings
    python scripts/pull_recordings.py ts_000001   # Pull specific recording
    python scripts/pull_recordings.py --list      # List available recordings
"""

import subprocess
import sys
from pathlib import Path

# Paths
PACKAGE_NAME = "com.speaktolearn.speak_to_learn"
DEVICE_RECORDINGS_PATH = f"/data/data/{PACKAGE_NAME}/app_flutter/recordings"
LOCAL_OUTPUT_DIR = Path(__file__).parent.parent / "tools" / "mandarin_grader" / "fixtures" / "audio"


def run_adb(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run adb command and return result."""
    cmd = ["adb"] + args
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def list_recordings() -> list[str]:
    """List recording files on emulator."""
    # Need to run as root to access app data on emulator
    run_adb(["root"], check=False)
    result = run_adb(["shell", "ls", DEVICE_RECORDINGS_PATH], check=False)
    if result.returncode != 0:
        print(f"No recordings found or path doesn't exist: {DEVICE_RECORDINGS_PATH}")
        return []
    files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    return files


def pull_recording(filename: str, output_dir: Path) -> Path | None:
    """Pull a single recording from emulator."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device_path = f"{DEVICE_RECORDINGS_PATH}/{filename}"
    local_path = output_dir / filename

    result = run_adb(["pull", device_path, str(local_path)], check=False)
    if result.returncode != 0:
        print(f"Failed to pull {filename}: {result.stderr}")
        return None

    print(f"Pulled: {local_path}")
    return local_path


def convert_to_wav(m4a_path: Path) -> Path | None:
    """Convert m4a to wav using ffmpeg (16kHz mono for scoring)."""
    wav_path = m4a_path.with_suffix(".wav")
    cmd = [
        "ffmpeg", "-y", "-i", str(m4a_path),
        "-ar", "16000",  # 16kHz sample rate
        "-ac", "1",      # Mono
        str(wav_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"FFmpeg conversion failed: {result.stderr}")
        return None
    print(f"Converted: {wav_path}")
    return wav_path


def main():
    args = sys.argv[1:]

    if not args or args[0] == "--list":
        print("Recordings on emulator:")
        for f in list_recordings():
            print(f"  {f}")
        return

    # Ensure adb root access for emulator
    run_adb(["root"], check=False)

    if args[0] == "--all":
        files = list_recordings()
    else:
        # Pull specific file(s)
        files = [f"{arg}.m4a" if not arg.endswith(".m4a") else arg for arg in args]

    for filename in files:
        local_path = pull_recording(filename, LOCAL_OUTPUT_DIR)
        if local_path and local_path.suffix == ".m4a":
            convert_to_wav(local_path)


if __name__ == "__main__":
    main()
