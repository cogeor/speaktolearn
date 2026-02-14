"""AISHELL-3 data source - native speaker recordings.

AISHELL-3 is a high-quality multi-speaker Mandarin TTS corpus with 85 hours
of audio from 218 speakers. Unlike synthetic data, it has natural coarticulation
and prosody, but no exact syllable boundaries.

Dataset structure after extraction:
    data_aishell3/
        train/
            wav/
                SSB0005/
                    SSB00050001.wav
                    SSB00050002.wav
                    ...
                SSB0009/
                    ...
            content.txt  # Transcripts: "utterance_id\tpinyin\thanzi"
        test/
            wav/
                ...
            content.txt
        speaker.info  # Speaker metadata

Download from: https://www.openslr.org/93/
"""

from __future__ import annotations

import json
import re
import wave
from pathlib import Path
from typing import Iterator

import numpy as np

from .data_source import DataSource, SentenceInfo
from ..types import TargetSyllable
from .dataloader import parse_romanization


def parse_aishell3_pinyin(pinyin_str: str) -> list[tuple[str, int]]:
    """Parse AISHELL-3 pinyin format to (syllable, tone) pairs.

    AISHELL-3 uses format like: "ni3 hao3" (syllable followed by tone number)

    Args:
        pinyin_str: Space-separated pinyin with tone numbers

    Returns:
        List of (base_pinyin, tone) tuples
    """
    syllables = []
    # Split by space and parse each syllable
    for token in pinyin_str.strip().split():
        if not token:
            continue
        # Handle tone number at end (e.g., "ni3" -> ("ni", 3))
        match = re.match(r'^([a-zA-ZüÜ]+)(\d)?$', token)
        if match:
            base = match.group(1).lower()
            tone = int(match.group(2)) if match.group(2) else 0
            # Map tone 5 (neutral in AISHELL-3) to tone 0, clamp to 0-4
            if tone == 5:
                tone = 0
            elif tone > 4:
                tone = 0
            syllables.append((base, tone))
        else:
            # Fallback: just use as-is with tone 0
            syllables.append((token.lower(), 0))
    return syllables


def get_audio_duration_samples(wav_path: Path) -> int:
    """Get audio duration in samples."""
    try:
        with wave.open(str(wav_path), 'rb') as wf:
            return wf.getnframes()
    except Exception:
        return 0


class AISHELL3DataSource(DataSource):
    """Data source for AISHELL-3 native speaker recordings.

    AISHELL-3 provides high-quality native Mandarin speech but without
    syllable-level alignments. Boundaries are estimated uniformly based
    on audio duration and syllable count.

    This is ideal for training on natural speech with real coarticulation,
    complementing synthetic data which has exact boundaries but unnatural
    transitions.
    """

    name = "aishell3"
    description = "AISHELL-3 native speaker recordings (85h, 218 speakers)"

    def load(
        self,
        data_dir: Path,
        split: str = "train",
        max_sentences: int | None = None,
        speakers: list[str] | None = None,
    ) -> list[SentenceInfo]:
        """Load AISHELL-3 sentences.

        Args:
            data_dir: Root directory of extracted AISHELL-3 data
            split: "train" or "test"
            max_sentences: Maximum number of sentences to load
            speakers: Optional list of speaker IDs to filter

        Returns:
            List of SentenceInfo objects with estimated boundaries
        """
        split_dir = data_dir / split
        content_file = split_dir / "content.txt"
        wav_dir = split_dir / "wav"

        if not content_file.exists():
            return []

        # Load transcripts: utterance_id -> (pinyin, hanzi)
        # AISHELL-3 format: "SSB00050001.wav\t广 guang3 州 zhou1 女 nv3..."
        # Interleaved hanzi and pinyin pairs
        transcripts = {}
        with open(content_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Split on tab: filename and content
                parts = line.split('\t', 1)
                if len(parts) < 2:
                    continue

                utt_id = parts[0].replace('.wav', '')  # Remove .wav extension
                content = parts[1]

                # Parse interleaved hanzi + pinyin format
                # "广 guang3 州 zhou1" -> hanzi="广州", pinyin="guang3 zhou1"
                tokens = content.split()
                hanzi_chars = []
                pinyin_parts = []

                for i, token in enumerate(tokens):
                    # Odd indices are pinyin (with tone number), even are hanzi
                    if i % 2 == 0:
                        hanzi_chars.append(token)
                    else:
                        pinyin_parts.append(token)

                hanzi = ''.join(hanzi_chars)
                pinyin = ' '.join(pinyin_parts)
                transcripts[utt_id] = (pinyin, hanzi)

        # Find all wav files
        sentences = []
        count = 0

        for speaker_dir in sorted(wav_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue

            speaker_id = speaker_dir.name
            if speakers and speaker_id not in speakers:
                continue

            for wav_file in sorted(speaker_dir.glob("*.wav")):
                if max_sentences and count >= max_sentences:
                    break

                utt_id = wav_file.stem
                if utt_id not in transcripts:
                    continue

                pinyin_str, hanzi = transcripts[utt_id]

                # Parse syllables
                if pinyin_str:
                    pinyin_tones = parse_aishell3_pinyin(pinyin_str)
                    syllables = []
                    for i, (base, tone) in enumerate(pinyin_tones):
                        # Get corresponding hanzi character if available
                        char = hanzi[i] if i < len(hanzi) else ""
                        syllables.append(TargetSyllable(
                            index=i,
                            hanzi=char,
                            pinyin=base,
                            initial="",  # Could parse from pinyin
                            final="",
                            tone_underlying=tone,
                            tone_surface=tone,
                        ))
                else:
                    # Fallback: parse from text using existing logic
                    syllables = parse_romanization("", hanzi)

                if not syllables:
                    continue

                # Estimate syllable boundaries uniformly
                audio_samples = get_audio_duration_samples(wav_file)
                if audio_samples == 0:
                    continue

                n_syllables = len(syllables)
                samples_per_syl = audio_samples // n_syllables
                boundaries = [
                    (i * samples_per_syl, (i + 1) * samples_per_syl)
                    for i in range(n_syllables)
                ]
                # Adjust last boundary to include remainder
                if boundaries:
                    boundaries[-1] = (boundaries[-1][0], audio_samples)

                sentences.append(SentenceInfo(
                    id=utt_id,
                    audio_path=wav_file,
                    text=hanzi,
                    syllables=syllables,
                    syllable_boundaries=boundaries,
                    sample_rate=16000,
                    total_samples=audio_samples,
                ))
                count += 1

            if max_sentences and count >= max_sentences:
                break

        return sentences

    def is_available(self, data_dir: Path) -> bool:
        """Check if AISHELL-3 data exists."""
        train_content = data_dir / "train" / "content.txt"
        train_wav = data_dir / "train" / "wav"
        return train_content.exists() and train_wav.exists()

    def get_speakers(self, data_dir: Path, split: str = "train") -> list[str]:
        """Get list of available speaker IDs."""
        wav_dir = data_dir / split / "wav"
        if not wav_dir.exists():
            return []
        return sorted(d.name for d in wav_dir.iterdir() if d.is_dir())
