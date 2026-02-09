#!/usr/bin/env python3
"""Generate syllable lexicon from TTS for synthetic data generation.

This script extracts all unique syllables from the sentence dataset and
generates individual audio files for each using edge-tts.

Usage:
    python generate_syllable_lexicon.py --sentences path/to/sentences.zh.json
    python generate_syllable_lexicon.py --output data/syllables
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Literal

# Add parent package to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_SENTENCES = Path("apps/mobile_flutter/assets/datasets/sentences.zh.json")
DEFAULT_OUTPUT = Path("tools/mandarin_grader/data/syllables")

# TTS voice mappings
VOICES = {
    "female": "zh-CN-XiaoxiaoNeural",
    "male": "zh-CN-YunxiNeural",
}

# Pinyin with tone number to spoken form mapping
# The TTS expects pinyin with tone marks or numbers
TONE_MARKS = {
    "a": ["ā", "á", "ǎ", "à"],
    "e": ["ē", "é", "ě", "è"],
    "i": ["ī", "í", "ǐ", "ì"],
    "o": ["ō", "ó", "ǒ", "ò"],
    "u": ["ū", "ú", "ǔ", "ù"],
    "ü": ["ǖ", "ǘ", "ǚ", "ǜ"],
    "v": ["ǖ", "ǘ", "ǚ", "ǜ"],  # v is often used for ü
}


def add_tone_mark(pinyin: str, tone: int) -> str:
    """Add tone mark to pinyin.

    Rules for where to place the tone mark:
    1. If there's an 'a' or 'e', it takes the mark
    2. If there's 'ou', the 'o' takes the mark
    3. Otherwise, the last vowel takes the mark
    """
    if tone == 0 or tone > 4:
        return pinyin  # Neutral tone, no mark

    vowels = "aeiouüv"
    pinyin_lower = pinyin.lower()

    # Find where to place the mark
    mark_idx = -1

    # Rule 1: a or e gets the mark
    for i, char in enumerate(pinyin_lower):
        if char in "ae":
            mark_idx = i
            break

    # Rule 2: 'ou' -> mark on 'o'
    if mark_idx == -1:
        ou_idx = pinyin_lower.find("ou")
        if ou_idx != -1:
            mark_idx = ou_idx

    # Rule 3: last vowel gets the mark
    if mark_idx == -1:
        for i in range(len(pinyin_lower) - 1, -1, -1):
            if pinyin_lower[i] in vowels:
                mark_idx = i
                break

    if mark_idx == -1:
        return pinyin  # No vowel found

    # Apply the tone mark
    char = pinyin_lower[mark_idx]
    if char == "v":
        char = "ü"
    if char in TONE_MARKS:
        marked_char = TONE_MARKS[char][tone - 1]
        return pinyin[:mark_idx] + marked_char + pinyin[mark_idx + 1:]

    return pinyin


async def generate_syllable_audio(
    pinyin: str,
    tone: int,
    voice: str,
    output_path: Path,
) -> int:
    """Generate audio for a single syllable using edge-tts.

    Args:
        pinyin: Base pinyin without tone mark
        tone: Tone number 1-4 (0 for neutral)
        voice: TTS voice name
        output_path: Path to save the audio file

    Returns:
        Duration in milliseconds
    """
    import edge_tts
    import io
    import wave

    # Create pinyin with tone mark for TTS
    marked_pinyin = add_tone_mark(pinyin, tone)

    # Generate audio
    communicate = edge_tts.Communicate(marked_pinyin, voice)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect audio data
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]

    # edge-tts returns MP3, convert to WAV using scipy/numpy
    duration_ms = convert_mp3_to_wav(audio_data, output_path)

    return duration_ms


def convert_mp3_to_wav(mp3_data: bytes, wav_path: Path, target_sr: int = 16000) -> int:
    """Convert MP3 bytes to WAV file.

    Args:
        mp3_data: MP3 audio data as bytes
        wav_path: Output WAV path
        target_sr: Target sample rate

    Returns:
        Duration in milliseconds
    """
    import tempfile
    import subprocess
    import wave
    import struct

    # Find ffmpeg - check common locations
    ffmpeg_paths = [
        "ffmpeg",  # In PATH
        r"C:\Users\costa\AppData\Local\Overwolf\Extensions\ncfplpkmiejjaklknfnkgcpapnhkggmlcppckhcb\270.0.25\obs\bin\64bit\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
    ]

    ffmpeg_cmd = None
    for path in ffmpeg_paths:
        try:
            subprocess.run([path, "-version"], capture_output=True, check=True)
            ffmpeg_cmd = path
            break
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    if ffmpeg_cmd is None:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg and add to PATH.")

    # Write MP3 to temp file
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
        tmp_mp3.write(mp3_data)
        tmp_mp3_path = tmp_mp3.name

    # Create temp WAV path
    tmp_wav_path = tmp_mp3_path.replace(".mp3", ".wav")

    try:
        # Convert MP3 to WAV using ffmpeg
        subprocess.run([
            ffmpeg_cmd, "-y", "-i", tmp_mp3_path,
            "-ar", str(target_sr),
            "-ac", "1",
            "-sample_fmt", "s16",
            "-f", "wav",
            tmp_wav_path
        ], capture_output=True, check=True)

        # Read the WAV file using Python's wave module
        with wave.open(tmp_wav_path, 'rb') as wf:
            n_frames = wf.getnframes()
            raw_data = wf.readframes(n_frames)
            sr = wf.getframerate()

        # Convert bytes to numpy array of int16
        audio = np.array(struct.unpack(f'{n_frames}h', raw_data), dtype=np.int16)

        # Convert to float for processing
        audio_float = audio.astype(np.float32) / 32768.0

        # Simple silence trimming: find first/last sample above threshold
        threshold = 0.01
        nonsilent = np.abs(audio_float) > threshold

        if np.any(nonsilent):
            first_nonsilent = np.argmax(nonsilent)
            last_nonsilent = len(nonsilent) - np.argmax(nonsilent[::-1]) - 1

            # Add margin (20ms = 320 samples at 16kHz)
            margin = int(0.02 * target_sr)
            start = max(0, first_nonsilent - margin)
            end = min(len(audio_float), last_nonsilent + margin)

            audio_trimmed = audio_float[start:end]
        else:
            audio_trimmed = audio_float

        # Convert back to int16
        audio_int16 = (audio_trimmed * 32767).clip(-32768, 32767).astype(np.int16)

        # Save using wave module
        with wave.open(str(wav_path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(target_sr)
            wf.writeframes(audio_int16.tobytes())

        # Return duration in ms
        duration_ms = int(len(audio_trimmed) / target_sr * 1000)
        return duration_ms

    finally:
        # Clean up temp files
        import os
        if os.path.exists(tmp_mp3_path):
            os.unlink(tmp_mp3_path)
        if os.path.exists(tmp_wav_path):
            os.unlink(tmp_wav_path)


async def generate_lexicon(
    sentences_json: Path,
    output_dir: Path,
    voices: list[str] | None = None,
) -> None:
    """Generate complete syllable lexicon.

    Args:
        sentences_json: Path to sentences.zh.json
        output_dir: Directory to save syllable audio files
        voices: List of voice IDs to generate (default: ["female", "male"])
    """
    from mandarin_grader.data.lexicon import (
        SyllableLexicon,
        SyllableEntry,
        extract_unique_syllables,
    )

    if voices is None:
        voices = ["female", "male"]

    # Extract unique syllables
    logger.info(f"Extracting syllables from {sentences_json}")
    syllables = extract_unique_syllables(sentences_json)
    logger.info(f"Found {len(syllables)} unique (pinyin, tone) pairs")

    # Create lexicon
    lexicon = SyllableLexicon.create_empty(output_dir)

    # Generate audio for each syllable and voice
    total = len(syllables) * len(voices)
    completed = 0

    for voice_id in voices:
        voice_name = VOICES.get(voice_id, voice_id)
        voice_dir = output_dir / voice_id
        voice_dir.mkdir(parents=True, exist_ok=True)

        for pinyin, tone in syllables:
            # Skip if already exists
            if lexicon.has(pinyin, tone, voice_id):
                completed += 1
                continue

            filename = f"{pinyin}{tone}.wav"
            audio_path = voice_dir / filename
            relative_path = f"{voice_id}/{filename}"

            try:
                duration_ms = await generate_syllable_audio(
                    pinyin=pinyin,
                    tone=tone,
                    voice=voice_name,
                    output_path=audio_path,
                )

                entry = SyllableEntry(
                    pinyin=pinyin,
                    tone=tone,
                    voice_id=voice_id,
                    audio_path=relative_path,
                    duration_ms=duration_ms,
                )
                lexicon.add(entry)

                completed += 1
                if completed % 10 == 0:
                    logger.info(f"Progress: {completed}/{total} ({100*completed/total:.1f}%)")

            except Exception as e:
                logger.error(f"Failed to generate {voice_id}/{pinyin}{tone}: {e}")

    # Save lexicon
    lexicon.save()
    logger.info(f"Saved lexicon with {len(lexicon)} entries to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate syllable lexicon for synthetic data"
    )
    parser.add_argument(
        "--sentences",
        type=Path,
        default=DEFAULT_SENTENCES,
        help="Path to sentences.zh.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory for syllable audio files",
    )
    parser.add_argument(
        "--voices",
        nargs="+",
        default=["female", "male"],
        help="Voice IDs to generate",
    )

    args = parser.parse_args()

    # Run async generation
    asyncio.run(generate_lexicon(
        sentences_json=args.sentences,
        output_dir=args.output,
        voices=args.voices,
    ))


if __name__ == "__main__":
    main()
