"""Syllable lexicon management for synthetic data generation.

This module provides infrastructure for managing individual syllable audio files
with metadata, used for generating synthetic training data via concatenation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, Literal

from ..types import Tone


@dataclass
class SyllableEntry:
    """Metadata for a single syllable audio file."""

    pinyin: str  # Base pinyin without tone mark (e.g., "ni", "hao")
    tone: Tone  # Tone number 0-4
    voice_id: str  # Voice identifier (e.g., "female", "male")
    audio_path: str  # Relative path to audio file
    duration_ms: int = 0  # Duration in milliseconds

    @property
    def syllable_key(self) -> str:
        """Return unique key for this syllable: pinyin + tone."""
        return f"{self.pinyin}{self.tone}"

    @property
    def full_key(self) -> str:
        """Return unique key including voice: voice/pinyin + tone."""
        return f"{self.voice_id}/{self.syllable_key}"


@dataclass
class SyllableLexicon:
    """Collection of syllable audio files with metadata.

    Manages individual syllable recordings for use in synthetic
    sentence generation via concatenation.

    Usage:
        lexicon = SyllableLexicon.load(Path("data/syllables"))
        entry = lexicon.get("ni", 3, "female")
        audio = load_audio(lexicon.base_path / entry.audio_path)
    """

    base_path: Path
    entries: dict[str, SyllableEntry] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[SyllableEntry]:
        return iter(self.entries.values())

    def add(self, entry: SyllableEntry) -> None:
        """Add a syllable entry to the lexicon."""
        self.entries[entry.full_key] = entry

    def get(
        self,
        pinyin: str,
        tone: Tone,
        voice: str = "female"
    ) -> SyllableEntry | None:
        """Get entry for a specific syllable.

        Args:
            pinyin: Base pinyin without tone (e.g., "ni")
            tone: Tone number 0-4
            voice: Voice identifier

        Returns:
            SyllableEntry if found, None otherwise
        """
        key = f"{voice}/{pinyin}{tone}"
        return self.entries.get(key)

    def has(self, pinyin: str, tone: Tone, voice: str = "female") -> bool:
        """Check if a syllable exists in the lexicon."""
        return self.get(pinyin, tone, voice) is not None

    def list_syllables(self, voice: str | None = None) -> list[SyllableEntry]:
        """List all syllable entries, optionally filtered by voice."""
        if voice is None:
            return list(self.entries.values())
        return [e for e in self.entries.values() if e.voice_id == voice]

    def list_voices(self) -> set[str]:
        """Return set of all voice IDs in the lexicon."""
        return {e.voice_id for e in self.entries.values()}

    def get_audio_path(
        self,
        pinyin: str,
        tone: Tone,
        voice: str = "female"
    ) -> Path | None:
        """Get full audio path for a syllable."""
        entry = self.get(pinyin, tone, voice)
        if entry is None:
            return None
        return self.base_path / entry.audio_path

    def save(self, path: Path | None = None) -> None:
        """Save lexicon to JSON file.

        Args:
            path: Path to save to. Defaults to base_path/lexicon.json
        """
        if path is None:
            path = self.base_path / "lexicon.json"

        data = {
            "version": "1.0",
            "entries": [asdict(e) for e in self.entries.values()]
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, base_path: Path) -> "SyllableLexicon":
        """Load lexicon from directory.

        Supports two formats:
        - v1: lexicon.json with entries list
        - v2: metadata.json with voice/tone{N}/syllable.wav structure

        Args:
            base_path: Directory containing lexicon files and audio

        Returns:
            SyllableLexicon instance
        """
        # Try v2 format first (metadata.json with directory structure)
        metadata_path = base_path / "metadata.json"
        if metadata_path.exists():
            return cls._load_v2(base_path, metadata_path)

        # Fall back to v1 format (lexicon.json)
        lexicon_path = base_path / "lexicon.json"
        if not lexicon_path.exists():
            return cls(base_path=base_path)

        with open(lexicon_path, encoding="utf-8") as f:
            data = json.load(f)

        entries = {}
        for entry_data in data.get("entries", []):
            entry = SyllableEntry(
                pinyin=entry_data["pinyin"],
                tone=entry_data["tone"],
                voice_id=entry_data["voice_id"],
                audio_path=entry_data["audio_path"],
                duration_ms=entry_data.get("duration_ms", 0),
            )
            entries[entry.full_key] = entry

        return cls(base_path=base_path, entries=entries)

    @classmethod
    def _load_v2(cls, base_path: Path, metadata_path: Path) -> "SyllableLexicon":
        """Load v2 format lexicon with voice/tone{N}/syllable.wav structure."""
        import wave

        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        voices = metadata.get("voices", {})
        tones = metadata.get("tones", [0, 1, 2, 3, 4])

        entries = {}

        for voice_name in voices.keys():
            voice_dir = base_path / voice_name
            if not voice_dir.exists():
                continue

            for tone in tones:
                tone_dir = voice_dir / f"tone{tone}"
                if not tone_dir.exists():
                    continue

                for wav_file in tone_dir.glob("*.wav"):
                    pinyin = wav_file.stem  # filename without extension

                    # Get duration
                    try:
                        with wave.open(str(wav_file), 'rb') as wf:
                            frames = wf.getnframes()
                            rate = wf.getframerate()
                            duration_ms = int(frames / rate * 1000)
                    except Exception:
                        duration_ms = 0

                    # Relative path from base
                    audio_path = f"{voice_name}/tone{tone}/{wav_file.name}"

                    entry = SyllableEntry(
                        pinyin=pinyin,
                        tone=tone,
                        voice_id=voice_name,
                        audio_path=audio_path,
                        duration_ms=duration_ms,
                    )
                    entries[entry.full_key] = entry

        return cls(base_path=base_path, entries=entries)

    @classmethod
    def create_empty(cls, base_path: Path) -> "SyllableLexicon":
        """Create an empty lexicon at the given path."""
        base_path.mkdir(parents=True, exist_ok=True)
        return cls(base_path=base_path)


def extract_unique_syllables(
    sentences_json: Path,
) -> list[tuple[str, Tone]]:
    """Extract all unique (pinyin, tone) pairs from sentence dataset.

    Args:
        sentences_json: Path to sentences.zh.json

    Returns:
        List of (pinyin_base, tone) tuples
    """
    from .dataloader import parse_romanization

    with open(sentences_json, encoding="utf-8") as f:
        data = json.load(f)

    unique: set[tuple[str, Tone]] = set()

    for item in data.get("items", []):
        romanization = item.get("romanization", "")
        text = item.get("text", "")

        if not romanization:
            continue

        syllables = parse_romanization(romanization, text)
        for syl in syllables:
            # Extract base pinyin (remove tone marks)
            base_pinyin = _remove_tone_marks(syl.pinyin)
            unique.add((base_pinyin, syl.tone_underlying))

    return sorted(unique)


def _remove_tone_marks(pinyin: str) -> str:
    """Remove tone marks from pinyin, keeping base letters."""
    TONE_MARKS = {
        "ā": "a", "á": "a", "ǎ": "a", "à": "a",
        "ē": "e", "é": "e", "ě": "e", "è": "e",
        "ī": "i", "í": "i", "ǐ": "i", "ì": "i",
        "ō": "o", "ó": "o", "ǒ": "o", "ò": "o",
        "ū": "u", "ú": "u", "ǔ": "u", "ù": "u",
        "ǖ": "ü", "ǘ": "ü", "ǚ": "ü", "ǜ": "ü",
    }
    result = ""
    for char in pinyin.lower():
        result += TONE_MARKS.get(char, char)
    # Remove trailing tone numbers
    if result and result[-1].isdigit():
        result = result[:-1]
    return result
