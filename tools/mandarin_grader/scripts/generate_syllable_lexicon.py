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
from pathlib import Path
from typing import Literal

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

    # Create pinyin with tone mark for TTS
    marked_pinyin = add_tone_mark(pinyin, tone)

    # Generate audio
    communicate = edge_tts.Communicate(marked_pinyin, voice)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to temporary mp3 first
    temp_mp3 = output_path.with_suffix(".mp3")
    await communicate.save(str(temp_mp3))

    # Convert to WAV and get duration
    duration_ms = await convert_and_trim(temp_mp3, output_path)

    # Remove temp file
    temp_mp3.unlink(missing_ok=True)

    return duration_ms


async def convert_and_trim(mp3_path: Path, wav_path: Path) -> int:
    """Convert MP3 to WAV and trim silence.

    Args:
        mp3_path: Input MP3 path
        wav_path: Output WAV path

    Returns:
        Duration in milliseconds
    """
    from pydub import AudioSegment
    from pydub.silence import detect_leading_silence

    # Load audio
    audio = AudioSegment.from_mp3(str(mp3_path))

    # Convert to mono 16kHz
    audio = audio.set_frame_rate(16000).set_channels(1)

    # Trim silence from start and end
    def detect_trailing_silence(sound, silence_threshold=-40.0, chunk_size=10):
        """Detect trailing silence by reversing."""
        trim_ms = detect_leading_silence(
            sound.reverse(), silence_threshold, chunk_size
        )
        return len(sound) - trim_ms

    start_trim = detect_leading_silence(audio, silence_threshold=-40.0)
    end_trim = detect_trailing_silence(audio, silence_threshold=-40.0)

    # Keep a small margin (20ms) on each side
    margin_ms = 20
    start_trim = max(0, start_trim - margin_ms)
    end_trim = min(len(audio), end_trim + margin_ms)

    trimmed = audio[start_trim:end_trim]

    # Export as WAV
    trimmed.export(str(wav_path), format="wav")

    return len(trimmed)


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
