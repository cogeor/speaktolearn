#!/usr/bin/env python3
"""Pull recordings from connected Android device for ML evaluation.

This script automates the workflow:
1. Check for connected Android device via adb
2. Pull recordings from app's documents directory
3. Optionally validate and evaluate with V4 model

Usage:
    # Pull recordings to ./pulled_recordings/
    python pull_recordings.py

    # Pull and validate format
    python pull_recordings.py --validate

    # Pull to custom directory
    python pull_recordings.py --output ./my_recordings

    # Evaluate with model checkpoint
    python pull_recordings.py --evaluate --checkpoint checkpoints_v4/best_model.pt

Requirements:
    - adb (Android Debug Bridge) in PATH
    - USB debugging enabled on Android device
    - Device connected via USB
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# App package name (from android/app/build.gradle.kts)
APP_PACKAGE = "com.speaktolearn.speak_to_learn"

# Recordings directory inside app (relative to app data)
RECORDINGS_SUBDIR = "app_flutter/recordings"


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return result."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=check)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(cmd)}")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
        raise


def check_adb() -> bool:
    """Check if adb is available."""
    try:
        result = run_cmd(["adb", "version"], check=False)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def get_connected_devices() -> list[str]:
    """Get list of connected Android devices."""
    result = run_cmd(["adb", "devices"])
    lines = result.stdout.strip().split("\n")[1:]  # Skip header
    devices = []
    for line in lines:
        if line.strip() and "\tdevice" in line:
            device_id = line.split("\t")[0]
            devices.append(device_id)
    return devices


def get_device_recordings_path() -> str:
    """Get the full path to recordings on device."""
    # Run-as allows accessing app's private data directory
    return f"/data/data/{APP_PACKAGE}/{RECORDINGS_SUBDIR}"


def list_device_recordings(device_id: str | None = None) -> list[str]:
    """List recordings on device."""
    cmd = ["adb"]
    if device_id:
        cmd.extend(["-s", device_id])

    recordings_path = get_device_recordings_path()

    # Use run-as to access app's private directory
    cmd.extend(["shell", "run-as", APP_PACKAGE, "ls", "-la", f"files/recordings/"])

    result = run_cmd(cmd, check=False)
    if result.returncode != 0:
        # Try alternative path (some devices)
        cmd[-1] = recordings_path.replace(f"/data/data/{APP_PACKAGE}/", "")
        result = run_cmd(cmd, check=False)

    if result.returncode != 0:
        return []

    files = []
    for line in result.stdout.strip().split("\n"):
        if ".wav" in line.lower():
            # Parse ls -la output to get filename
            parts = line.split()
            if parts:
                filename = parts[-1]
                files.append(filename)
    return files


def pull_recordings(
    output_dir: Path,
    device_id: str | None = None,
) -> list[Path]:
    """Pull recordings from device to local directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd_base = ["adb"]
    if device_id:
        cmd_base.extend(["-s", device_id])

    # List files first
    recordings = list_device_recordings(device_id)

    if not recordings:
        print("No recordings found on device.")
        print(f"  Checked: {APP_PACKAGE}/files/recordings/")
        return []

    print(f"Found {len(recordings)} recording(s) on device:")
    for r in recordings:
        print(f"  - {r}")

    pulled_files = []

    # Pull each file using run-as + cat (workaround for adb pull permissions)
    for filename in recordings:
        local_path = output_dir / filename

        # Use run-as to read file content
        cmd = cmd_base + [
            "shell", "run-as", APP_PACKAGE,
            "cat", f"files/recordings/{filename}"
        ]

        print(f"Pulling: {filename}...", end=" ")
        result = run_cmd(cmd, check=False)

        if result.returncode == 0:
            # Write binary content to local file
            # Need to pull as binary, so use exec-out instead
            cmd_bin = cmd_base + [
                "exec-out", "run-as", APP_PACKAGE,
                "cat", f"files/recordings/{filename}"
            ]
            result_bin = subprocess.run(cmd_bin, capture_output=True, check=False)

            if result_bin.returncode == 0 and len(result_bin.stdout) > 0:
                local_path.write_bytes(result_bin.stdout)
                pulled_files.append(local_path)
                size_kb = len(result_bin.stdout) / 1024
                print(f"OK ({size_kb:.1f} KB)")
            else:
                print("FAILED (empty or error)")
        else:
            print(f"FAILED ({result.stderr.strip()})")

    return pulled_files


def validate_recordings(files: list[Path]) -> bool:
    """Validate pulled recordings using validate_flutter_audio.py."""
    script_dir = Path(__file__).parent
    validate_script = script_dir / "validate_flutter_audio.py"

    if not validate_script.exists():
        print(f"Warning: Validation script not found: {validate_script}")
        return False

    print(f"\nValidating {len(files)} recording(s)...")
    all_passed = True

    for f in files:
        print(f"\n{'='*60}")
        print(f"Validating: {f.name}")
        print("=" * 60)

        result = subprocess.run(
            [sys.executable, str(validate_script), str(f)],
            capture_output=False,
        )

        if result.returncode != 0:
            all_passed = False

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Pull recordings from connected Android device",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python pull_recordings.py
    python pull_recordings.py --validate
    python pull_recordings.py --output ./my_recordings --validate
        """,
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./pulled_recordings"),
        help="Output directory for pulled recordings (default: ./pulled_recordings)",
    )
    parser.add_argument(
        "--validate", "-v",
        action="store_true",
        help="Validate pulled recordings for ML compatibility",
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default=None,
        help="Specific device ID (if multiple devices connected)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List recordings on device without pulling",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Pull Recordings from Android Device")
    print("=" * 60)

    # Check adb
    if not check_adb():
        print("Error: adb not found in PATH")
        print("  Install Android SDK Platform Tools")
        print("  https://developer.android.com/tools/releases/platform-tools")
        sys.exit(1)

    print("adb: OK")

    # Check connected devices
    devices = get_connected_devices()
    if not devices:
        print("Error: No Android devices connected")
        print("  1. Enable USB debugging on your device")
        print("  2. Connect device via USB")
        print("  3. Accept USB debugging prompt on device")
        sys.exit(1)

    print(f"Connected device(s): {', '.join(devices)}")

    device_id = args.device or (devices[0] if len(devices) == 1 else None)
    if len(devices) > 1 and not args.device:
        print(f"Multiple devices connected. Use --device to specify.")
        print(f"  Available: {', '.join(devices)}")
        sys.exit(1)

    # List only
    if args.list:
        print(f"\nRecordings on device ({device_id or 'default'}):")
        recordings = list_device_recordings(device_id)
        if recordings:
            for r in recordings:
                print(f"  - {r}")
        else:
            print("  (none found)")
        sys.exit(0)

    # Pull recordings
    print(f"\nPulling recordings to: {args.output.absolute()}")
    pulled = pull_recordings(args.output, device_id)

    if not pulled:
        print("\nNo recordings pulled.")
        sys.exit(1)

    print(f"\nPulled {len(pulled)} recording(s) to {args.output}")

    # Validate if requested
    if args.validate:
        success = validate_recordings(pulled)
        if not success:
            print("\nSome recordings failed validation.")
            sys.exit(1)
        print("\nAll recordings validated successfully!")

    print("\nDone!")
    sys.exit(0)


if __name__ == "__main__":
    main()
