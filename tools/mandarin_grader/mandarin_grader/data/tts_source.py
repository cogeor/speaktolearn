"""TTS data source for domain adaptation training.

This module provides a data source for TTS-generated audio files, enabling
domain adversarial training to adapt models from native speech to TTS speech.

Expected directory structure:
    tts_data/
        sentences.json  # List of {id, text, syllables, audio_file}
        audio/
            001.wav
            002.wav
            ...

The sentences.json format:
{
    "sample_rate": 16000,
    "sentences": [
        {
            "id": "tts_001",
            "text": "...",           # Chinese characters
            "syllables": [           # List of syllables with tones
                {"pinyin": "ni3", "hanzi": "...", "tone": 3},
                ...
            ],
            "audio_file": "audio/001.wav"  # Relative path
        },
        ...
    ]
}
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

from .data_source import DataSource, SentenceInfo
from ..types import TargetSyllable


class TTSDataSource(DataSource):
    """Data source for TTS-generated speech.

    TTS data is used for domain adaptation training. The model learns to
    handle both native speech (AISHELL-3) and TTS-generated speech through
    domain adversarial training with gradient reversal.

    Domain label: 1 (TTS), vs 0 (AISHELL-3/native)
    """

    name = "tts"
    description = "TTS-generated speech from OpenAI or Azure TTS"

    def load(
        self,
        data_dir: Path,
        split: str = "train",
        max_sentences: int | None = None,
    ) -> list[SentenceInfo]:
        """Load TTS sentences from data directory.

        Args:
            data_dir: Directory containing sentences.json and audio/
            split: Unused (TTS data is typically all training data)
            max_sentences: Maximum sentences to load (None = all)

        Returns:
            List of SentenceInfo objects
        """
        sentences_path = data_dir / "sentences.json"
        if not sentences_path.exists():
            return []

        with open(sentences_path, encoding="utf-8") as f:
            data = json.load(f)

        sample_rate = data.get("sample_rate", 16000)
        items = data.get("sentences", [])

        sentences: list[SentenceInfo] = []
        count = 0

        for item in items:
            if max_sentences is not None and count >= max_sentences:
                break

            # Get audio path
            audio_rel = item.get("audio_file", "")
            if not audio_rel:
                audio_rel = f"audio/{item['id']}.wav"
            audio_path = data_dir / audio_rel

            if not audio_path.exists():
                continue

            # Parse syllables
            syllables = self._parse_syllables(item.get("syllables", []))
            if not syllables:
                continue

            # Get audio length for boundary estimation
            total_samples = self._get_audio_samples(audio_path)
            if total_samples <= 0:
                continue

            # Estimate uniform syllable boundaries
            n_syllables = len(syllables)
            samples_per_syl = total_samples // max(n_syllables, 1)
            boundaries = [
                (i * samples_per_syl, (i + 1) * samples_per_syl)
                for i in range(n_syllables)
            ]
            if boundaries:
                boundaries[-1] = (boundaries[-1][0], total_samples)

            sentences.append(SentenceInfo(
                id=item["id"],
                audio_path=audio_path,
                text=item.get("text", ""),
                syllables=syllables,
                syllable_boundaries=boundaries,
                sample_rate=sample_rate,
                total_samples=total_samples,
            ))
            count += 1

        return sentences

    def _parse_syllables(self, syllable_data: list[dict]) -> list[TargetSyllable]:
        """Parse syllable data into TargetSyllable objects."""
        syllables = []
        for i, syl in enumerate(syllable_data):
            # Handle pinyin with tone number (e.g., "ni3") or tone marks
            pinyin = syl.get("pinyin", "")
            tone = syl.get("tone", 0)

            # If tone not provided, try to extract from pinyin
            if tone == 0 and pinyin and pinyin[-1].isdigit():
                try:
                    tone = int(pinyin[-1])
                    pinyin = pinyin[:-1]
                except ValueError:
                    pass

            syllables.append(TargetSyllable(
                index=i,
                hanzi=syl.get("hanzi", ""),
                pinyin=pinyin,
                initial="",
                final="",
                tone_underlying=tone,
                tone_surface=tone,
            ))
        return syllables

    def _get_audio_samples(self, audio_path: Path) -> int:
        """Get number of samples in audio file."""
        try:
            with wave.open(str(audio_path), 'rb') as wf:
                return wf.getnframes()
        except Exception:
            return 0

    def is_available(self, data_dir: Path) -> bool:
        """Check if TTS data exists at given path."""
        sentences_path = data_dir / "sentences.json"
        return sentences_path.exists()
