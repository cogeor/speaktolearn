"""Synthetic data source - syllable-concatenated audio.

This wraps the existing synthetic data loading logic from autoregressive_dataset.py.
Synthetic data has exact syllable boundaries from the concatenation process.
"""

from __future__ import annotations

from pathlib import Path

from .data_source import DataSource, SentenceInfo
from .autoregressive_dataset import load_synthetic_metadata, SyntheticSentenceInfo


class SyntheticDataSource(DataSource):
    """Data source for synthetic (syllable-concatenated) data.

    This is the original training data format where sentences are created
    by concatenating individual syllable audio files. Provides exact
    syllable boundaries from the synthesis process.

    Expected structure:
        data_dir/
            metadata.json   # Sentence info with boundaries
            audio/
                sentence_001.wav
                sentence_002.wav
                ...
    """

    name = "synthetic"
    description = "Synthetic syllable-concatenated audio with exact boundaries"

    def load(self, data_dir: Path, **kwargs) -> list[SentenceInfo]:
        """Load synthetic sentences.

        Args:
            data_dir: Directory containing metadata.json and audio/
            **kwargs: Unused

        Returns:
            List of SentenceInfo objects
        """
        raw_sentences = load_synthetic_metadata(data_dir)
        return [self._convert(s) for s in raw_sentences]

    def _convert(self, s: SyntheticSentenceInfo) -> SentenceInfo:
        """Convert internal format to unified SentenceInfo."""
        return SentenceInfo(
            id=s.id,
            audio_path=s.audio_path,
            text=s.text,
            syllables=s.syllables,
            syllable_boundaries=s.syllable_boundaries,
            sample_rate=s.sample_rate,
            total_samples=None,
        )

    def is_available(self, data_dir: Path) -> bool:
        """Check if synthetic data exists."""
        metadata_path = data_dir / "metadata.json"
        audio_dir = data_dir / "audio"
        return metadata_path.exists() and audio_dir.exists()
