"""Abstract data source interface for training data.

This module defines the interface for pluggable data sources. Each data source
provides sentence audio with syllable annotations, allowing training on various
datasets (synthetic, AISHELL, Common Voice, etc.).

Usage:
    from mandarin_grader.data.data_source import DataSourceRegistry

    # List available sources
    print(DataSourceRegistry.list_sources())

    # Load a specific source
    sentences = DataSourceRegistry.load("synthetic", data_dir=Path("data/synthetic_train"))
    sentences = DataSourceRegistry.load("aishell3", data_dir=Path("datasets/aishell3"))
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from ..types import TargetSyllable


@dataclass
class SentenceInfo:
    """Unified sentence info across all data sources.

    All data sources must produce SentenceInfo objects with these fields.
    This is the common format used by AutoregressiveDataset.
    """
    id: str
    audio_path: Path
    text: str  # Chinese characters
    syllables: list[TargetSyllable]  # Parsed syllable info with tones
    syllable_boundaries: list[tuple[int, int]]  # [(start_sample, end_sample), ...]
    sample_rate: int = 16000
    total_samples: int | None = None


class DataSource(ABC):
    """Abstract base class for data sources.

    Implement this interface to add a new data source for training.
    Each data source handles its own format parsing and boundary estimation.
    """

    name: str = "base"
    description: str = "Base data source"

    @abstractmethod
    def load(self, data_dir: Path, **kwargs) -> list[SentenceInfo]:
        """Load sentences from the data directory.

        Args:
            data_dir: Root directory containing the dataset
            **kwargs: Source-specific options

        Returns:
            List of SentenceInfo objects
        """
        pass

    @abstractmethod
    def is_available(self, data_dir: Path) -> bool:
        """Check if this data source is available at the given path.

        Args:
            data_dir: Root directory to check

        Returns:
            True if data source exists and is properly structured
        """
        pass

    def get_info(self, data_dir: Path) -> dict:
        """Get information about the data source.

        Args:
            data_dir: Root directory containing the dataset

        Returns:
            Dict with keys: name, available, n_sentences, description
        """
        available = self.is_available(data_dir)
        n_sentences = 0
        if available:
            try:
                sentences = self.load(data_dir)
                n_sentences = len(sentences)
            except Exception:
                pass

        return {
            "name": self.name,
            "available": available,
            "n_sentences": n_sentences,
            "description": self.description,
        }


