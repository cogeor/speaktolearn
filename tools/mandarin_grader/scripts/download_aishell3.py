#!/usr/bin/env python3
"""Download and extract AISHELL-3 dataset.

AISHELL-3 is a high-quality multi-speaker Mandarin TTS corpus:
- 85 hours of audio from 218 native speakers
- 88,035 utterances with pinyin and character transcripts
- Apache 2.0 license

Usage:
    # Download to default location (datasets/aishell3/)
    python download_aishell3.py

    # Download to specific directory
    python download_aishell3.py --output-dir /path/to/datasets/aishell3

    # Use specific mirror (eu, cn)
    python download_aishell3.py --mirror cn

    # Skip download, just extract existing archive
    python download_aishell3.py --extract-only --archive data_aishell3.tgz
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

# Mirror URLs for AISHELL-3
MIRRORS = {
    "eu1": "https://openslr.trmal.net/resources/93/data_aishell3.tgz",
    "eu2": "https://openslr.elda.org/resources/93/data_aishell3.tgz",
    "cn": "https://openslr.magicdatatech.com/resources/93/data_aishell3.tgz",
}

# Expected file size (~19GB)
EXPECTED_SIZE_GB = 19.0

# Default output directory
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "datasets" / "aishell3"


def download_with_progress(url: str, output_path: Path) -> bool:
    """Download file with progress indicator.

    Args:
        url: URL to download
        output_path: Where to save the file

    Returns:
        True if download succeeded
    """
    print(f"Downloading from: {url}")
    print(f"Output: {output_path}")
    print()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try using curl or wget for better progress
    if shutil.which("curl"):
        cmd = ["curl", "-L", "-o", str(output_path), "--progress-bar", url]
        result = subprocess.run(cmd)
        return result.returncode == 0
    elif shutil.which("wget"):
        cmd = ["wget", "-O", str(output_path), "--progress=bar:force", url]
        result = subprocess.run(cmd)
        return result.returncode == 0
    else:
        # Fallback to Python's urlretrieve
        def progress(count, block_size, total_size):
            downloaded = count * block_size
            percent = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
            gb_downloaded = downloaded / (1024 ** 3)
            gb_total = total_size / (1024 ** 3)
            sys.stdout.write(f"\r  {percent:.1f}% ({gb_downloaded:.2f}/{gb_total:.2f} GB)")
            sys.stdout.flush()

        try:
            urlretrieve(url, output_path, reporthook=progress)
            print()
            return True
        except Exception as e:
            print(f"\nError: {e}")
            return False


def extract_archive(archive_path: Path, output_dir: Path) -> bool:
    """Extract tar.gz archive.

    Args:
        archive_path: Path to .tgz file
        output_dir: Where to extract

    Returns:
        True if extraction succeeded
    """
    print(f"Extracting: {archive_path}")
    print(f"To: {output_dir}")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Try using system tar for better progress (especially on large files)
        if shutil.which("tar"):
            cmd = ["tar", "-xzf", str(archive_path), "-C", str(output_dir)]
            result = subprocess.run(cmd)
            if result.returncode == 0:
                return True
            # Fall through to Python extraction on failure

        # Python fallback
        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getmembers()
            total = len(members)
            for i, member in enumerate(members):
                if (i + 1) % 1000 == 0 or i == total - 1:
                    sys.stdout.write(f"\r  Extracting: {i + 1}/{total} files")
                    sys.stdout.flush()
                tar.extract(member, output_dir)
            print()

        return True
    except Exception as e:
        print(f"Error extracting: {e}")
        return False


def verify_structure(data_dir: Path) -> bool:
    """Verify AISHELL-3 directory structure.

    Args:
        data_dir: Root directory of extracted data

    Returns:
        True if structure looks correct
    """
    # Check for expected directories/files
    expected = [
        data_dir / "train" / "wav",
        data_dir / "train" / "content.txt",
        data_dir / "test" / "wav",
        data_dir / "test" / "content.txt",
    ]

    # After extraction, data is in data_aishell3/ subdirectory
    alt_data_dir = data_dir / "data_aishell3"
    if alt_data_dir.exists():
        expected = [
            alt_data_dir / "train" / "wav",
            alt_data_dir / "train" / "content.txt",
        ]

    missing = [p for p in expected if not p.exists()]

    if missing:
        print("Warning: Missing expected files/directories:")
        for p in missing:
            print(f"  - {p}")
        return False

    return True


def count_utterances(data_dir: Path) -> dict:
    """Count utterances in the dataset.

    Args:
        data_dir: Root directory of extracted data

    Returns:
        Dict with counts per split
    """
    # Handle nested structure
    if (data_dir / "data_aishell3").exists():
        data_dir = data_dir / "data_aishell3"

    counts = {}
    for split in ["train", "test"]:
        content_file = data_dir / split / "content.txt"
        if content_file.exists():
            with open(content_file, encoding="utf-8") as f:
                counts[split] = sum(1 for line in f if line.strip())
        else:
            counts[split] = 0

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract AISHELL-3 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download with default settings
    python download_aishell3.py

    # Use China mirror (faster in Asia)
    python download_aishell3.py --mirror cn

    # Custom output directory
    python download_aishell3.py --output-dir ~/datasets/aishell3

    # Extract existing archive
    python download_aishell3.py --extract-only --archive data_aishell3.tgz
        """
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--mirror", choices=["eu1", "eu2", "cn"], default="eu1",
        help="Mirror to use (default: eu1)"
    )
    parser.add_argument(
        "--extract-only", action="store_true",
        help="Skip download, just extract existing archive"
    )
    parser.add_argument(
        "--archive", type=Path,
        help="Path to existing archive (for --extract-only)"
    )
    parser.add_argument(
        "--keep-archive", action="store_true",
        help="Keep the archive after extraction"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("AISHELL-3 Dataset Download")
    print("=" * 60)
    print()
    print("Dataset info:")
    print("  - 85 hours of Mandarin speech")
    print("  - 218 native speakers")
    print("  - 88,035 utterances")
    print("  - ~19 GB download")
    print("  - License: Apache 2.0")
    print()

    archive_path = args.archive or (args.output_dir / "data_aishell3.tgz")

    # Download if needed
    if not args.extract_only:
        if archive_path.exists():
            size_gb = archive_path.stat().st_size / (1024 ** 3)
            print(f"Archive exists: {archive_path} ({size_gb:.2f} GB)")
            if size_gb < EXPECTED_SIZE_GB * 0.9:
                print("Warning: Archive seems incomplete. Re-downloading...")
                archive_path.unlink()
            else:
                print("Skipping download (use --extract-only to just extract)")

        if not archive_path.exists():
            url = MIRRORS[args.mirror]
            print(f"Mirror: {args.mirror}")
            print()

            if not download_with_progress(url, archive_path):
                print("Download failed!")
                sys.exit(1)

            print("Download complete!")
            print()

    # Extract
    if not archive_path.exists():
        print(f"Error: Archive not found: {archive_path}")
        sys.exit(1)

    print("Extracting archive...")
    if not extract_archive(archive_path, args.output_dir):
        print("Extraction failed!")
        sys.exit(1)

    print("Extraction complete!")
    print()

    # Verify
    print("Verifying structure...")
    if verify_structure(args.output_dir):
        print("Structure OK!")
    print()

    # Stats
    counts = count_utterances(args.output_dir)
    print("Utterance counts:")
    for split, count in counts.items():
        print(f"  {split}: {count:,}")
    print()

    # Cleanup
    if not args.keep_archive and archive_path.exists():
        print(f"Removing archive: {archive_path}")
        archive_path.unlink()

    # Final path info
    final_dir = args.output_dir
    if (args.output_dir / "data_aishell3").exists():
        final_dir = args.output_dir / "data_aishell3"

    print()
    print("=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print()
    print("To use in training:")
    print(f"  python train_v3.py --data-source aishell3 --data-dir {final_dir}")
    print()


if __name__ == "__main__":
    main()
