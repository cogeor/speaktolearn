"""Dataset loaders for TTS audio, user recordings, and external datasets."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

from ..types import Ms, TargetSyllable, Tone

# Pinyin initial consonants (ordered by length for greedy matching)
INITIALS = [
    "zh", "ch", "sh",  # 2-char initials first
    "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
    "j", "q", "x", "z", "c", "s", "r", "y", "w",
]

# Tone mark to number mapping
TONE_MARKS = {
    "ā": ("a", 1), "á": ("a", 2), "ǎ": ("a", 3), "à": ("a", 4),
    "ē": ("e", 1), "é": ("e", 2), "ě": ("e", 3), "è": ("e", 4),
    "ī": ("i", 1), "í": ("i", 2), "ǐ": ("i", 3), "ì": ("i", 4),
    "ō": ("o", 1), "ó": ("o", 2), "ǒ": ("o", 3), "ò": ("o", 4),
    "ū": ("u", 1), "ú": ("u", 2), "ǔ": ("u", 3), "ù": ("u", 4),
    "ǖ": ("ü", 1), "ǘ": ("ü", 2), "ǚ": ("ü", 3), "ǜ": ("ü", 4),
}


def parse_pinyin_syllable(pinyin: str) -> tuple[str, str, Tone]:
    """Parse a pinyin syllable into initial, final, and tone.

    Args:
        pinyin: Pinyin syllable with tone marks (e.g., "nǐ", "hǎo")

    Returns:
        Tuple of (initial, final, tone)
    """
    # Extract tone from tone marks
    tone: Tone = 0
    base_pinyin = ""
    for char in pinyin.lower():
        if char in TONE_MARKS:
            base_char, tone = TONE_MARKS[char]
            base_pinyin += base_char
        else:
            base_pinyin += char

    # If no tone mark found, check for tone number suffix
    if tone == 0 and base_pinyin and base_pinyin[-1].isdigit():
        tone_num = int(base_pinyin[-1])
        if 0 <= tone_num <= 4:
            tone = tone_num  # type: ignore
            base_pinyin = base_pinyin[:-1]

    # Split into initial and final
    initial = ""
    for init in INITIALS:
        if base_pinyin.startswith(init):
            initial = init
            break

    final = base_pinyin[len(initial):]

    return initial, final, tone


def parse_romanization(romanization: str, hanzi: str) -> list[TargetSyllable]:
    """Parse romanization string into list of TargetSyllable.

    Args:
        romanization: Space-separated pinyin with tone marks (e.g., "nǐ hǎo")
        hanzi: Chinese characters (e.g., "你好")

    Returns:
        List of TargetSyllable objects
    """
    syllables = romanization.strip().split()
    chars = list(hanzi)

    # Handle mismatch in length
    if len(syllables) != len(chars):
        # Try to match as best we can
        chars = chars[:len(syllables)] if len(chars) > len(syllables) else chars + [""] * (len(syllables) - len(chars))

    result = []
    for i, (syl, char) in enumerate(zip(syllables, chars)):
        initial, final, tone = parse_pinyin_syllable(syl)
        result.append(TargetSyllable(
            index=i,
            hanzi=char,
            pinyin=syl,
            initial=initial,
            final=final,
            tone_underlying=tone,
            tone_surface=tone,  # Will be updated by sandhi
        ))

    return result


@dataclass
class AudioSample:
    """A single audio sample with metadata."""

    id: str
    audio_path: Path
    text: str
    romanization: str
    syllables: list[TargetSyllable] = field(default_factory=list)
    source: Literal["tts", "user", "aishell"] = "tts"
    voice_id: str | None = None

    def __post_init__(self) -> None:
        """Parse syllables from romanization if not provided."""
        if not self.syllables and self.romanization:
            self.syllables = parse_romanization(self.romanization, self.text)


@dataclass
class SentenceDataset:
    """Dataset of sentences with audio."""

    samples: list[AudioSample] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[AudioSample]:
        return iter(self.samples)

    def __getitem__(self, idx: int) -> AudioSample:
        return self.samples[idx]

    def filter_by_ids(self, ids: set[str]) -> "SentenceDataset":
        """Return a new dataset with only specified IDs."""
        return SentenceDataset([s for s in self.samples if s.id in ids])

    @classmethod
    def from_app_assets(
        cls,
        sentences_json: Path,
        audio_dir: Path,
        voice: Literal["male", "female"] = "female",
    ) -> "SentenceDataset":
        """Load from Flutter app assets.

        Args:
            sentences_json: Path to sentences.zh.json
            audio_dir: Path to examples/ directory containing male/female subdirs
            voice: Which voice to load ("male" or "female")

        Returns:
            SentenceDataset with all available samples
        """
        with open(sentences_json, encoding="utf-8") as f:
            data = json.load(f)

        samples = []
        voice_dir = audio_dir / voice

        for item in data.get("items", []):
            audio_path = voice_dir / f"{item['id']}.mp3"
            if not audio_path.exists():
                continue

            samples.append(AudioSample(
                id=item["id"],
                audio_path=audio_path,
                text=item["text"],
                romanization=item.get("romanization", ""),
                source="tts",
                voice_id=f"{voice[0]}1",  # f1 or m1
            ))

        return cls(samples)

    @classmethod
    def from_user_recordings(
        cls,
        recordings_dir: Path,
        sentences_json: Path,
    ) -> "SentenceDataset":
        """Load user recordings pulled from emulator.

        Args:
            recordings_dir: Directory containing pulled recordings (*.wav or *.m4a)
            sentences_json: Path to sentences.zh.json for metadata

        Returns:
            SentenceDataset with available recordings
        """
        # Load sentence metadata
        with open(sentences_json, encoding="utf-8") as f:
            data = json.load(f)

        # Build lookup by ID
        metadata = {item["id"]: item for item in data.get("items", [])}

        samples = []
        for audio_path in recordings_dir.glob("*.wav"):
            # Extract ID from filename (e.g., ts_000001.wav -> ts_000001)
            sample_id = audio_path.stem

            if sample_id not in metadata:
                continue

            item = metadata[sample_id]
            samples.append(AudioSample(
                id=sample_id,
                audio_path=audio_path,
                text=item["text"],
                romanization=item.get("romanization", ""),
                source="user",
                voice_id=None,
            ))

        # Also check for m4a files
        for audio_path in recordings_dir.glob("*.m4a"):
            sample_id = audio_path.stem
            if sample_id not in metadata:
                continue
            if any(s.id == sample_id for s in samples):
                continue  # Prefer WAV

            item = metadata[sample_id]
            samples.append(AudioSample(
                id=sample_id,
                audio_path=audio_path,
                text=item["text"],
                romanization=item.get("romanization", ""),
                source="user",
                voice_id=None,
            ))

        return cls(samples)

    @classmethod
    def from_aishell(
        cls,
        aishell_root: Path,
        subset: Literal["train", "dev", "test"] = "train",
        max_samples: int | None = None,
    ) -> "SentenceDataset":
        """Load from AISHELL-1 format.

        Expected structure:
            aishell_root/
                data_aishell/
                    wav/
                        train/
                            S0001/
                                BAC009S0001W0001.wav
                    transcript/
                        aishell_transcript_v0.8.txt

        Args:
            aishell_root: Root directory of AISHELL dataset
            subset: Which subset to load
            max_samples: Maximum number of samples to load

        Returns:
            SentenceDataset with AISHELL samples
        """
        wav_dir = aishell_root / "data_aishell" / "wav" / subset
        transcript_path = aishell_root / "data_aishell" / "transcript" / "aishell_transcript_v0.8.txt"

        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript not found: {transcript_path}")

        # Load transcripts
        transcripts = {}
        with open(transcript_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    transcripts[parts[0]] = parts[1]

        samples = []
        count = 0

        for speaker_dir in sorted(wav_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue

            for wav_path in sorted(speaker_dir.glob("*.wav")):
                if max_samples and count >= max_samples:
                    return cls(samples)

                sample_id = wav_path.stem
                if sample_id not in transcripts:
                    continue

                text = transcripts[sample_id]
                samples.append(AudioSample(
                    id=sample_id,
                    audio_path=wav_path,
                    text=text,
                    romanization="",  # AISHELL doesn't include pinyin
                    source="aishell",
                    voice_id=speaker_dir.name,
                ))
                count += 1

        return cls(samples)


class ContourDataset:
    """Dataset of extracted contours for tone classifier training.

    This class wraps a SentenceDataset and provides methods to extract
    pitch contours from all samples using forced alignment boundaries.
    The extracted contours can be used to train a tone classifier.

    Usage:
        dataset = SentenceDataset.from_app_assets(...)
        contour_ds = ContourDataset(dataset)
        contours = contour_ds.extract_all()  # List of (contour, tone) tuples
    """

    def __init__(
        self,
        samples: SentenceDataset | list[AudioSample],
        k: int = 20,
    ) -> None:
        """Initialize ContourDataset.

        Args:
            samples: SentenceDataset or list of AudioSample objects.
            k: Number of points for resampled contours (default 20).
        """
        if isinstance(samples, SentenceDataset):
            self._samples = samples.samples
        else:
            self._samples = samples
        self._k = k

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self._samples)

    @property
    def samples(self) -> list[AudioSample]:
        """Return underlying samples."""
        return self._samples

    @property
    def k(self) -> int:
        """Return contour resampling points."""
        return self._k
