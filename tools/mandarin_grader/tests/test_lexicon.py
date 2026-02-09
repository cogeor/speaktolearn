"""Tests for syllable lexicon management."""

import json
import tempfile
from pathlib import Path

import pytest

from mandarin_grader.data.lexicon import (
    SyllableEntry,
    SyllableLexicon,
    extract_unique_syllables,
    _remove_tone_marks,
)


class TestSyllableEntry:
    """Tests for SyllableEntry dataclass."""

    def test_syllable_key(self):
        """Test syllable_key property."""
        entry = SyllableEntry(
            pinyin="ni",
            tone=3,
            voice_id="female",
            audio_path="female/ni3.wav",
            duration_ms=250,
        )
        assert entry.syllable_key == "ni3"

    def test_full_key(self):
        """Test full_key property with voice."""
        entry = SyllableEntry(
            pinyin="hao",
            tone=3,
            voice_id="male",
            audio_path="male/hao3.wav",
            duration_ms=300,
        )
        assert entry.full_key == "male/hao3"


class TestSyllableLexicon:
    """Tests for SyllableLexicon class."""

    def test_create_empty(self, tmp_path):
        """Test creating empty lexicon."""
        lexicon = SyllableLexicon.create_empty(tmp_path / "syllables")
        assert len(lexicon) == 0
        assert lexicon.base_path == tmp_path / "syllables"

    def test_add_and_get(self, tmp_path):
        """Test adding and retrieving entries."""
        lexicon = SyllableLexicon.create_empty(tmp_path)

        entry = SyllableEntry(
            pinyin="ni",
            tone=3,
            voice_id="female",
            audio_path="female/ni3.wav",
            duration_ms=250,
        )
        lexicon.add(entry)

        assert len(lexicon) == 1
        assert lexicon.has("ni", 3, "female")
        assert not lexicon.has("ni", 3, "male")

        retrieved = lexicon.get("ni", 3, "female")
        assert retrieved is not None
        assert retrieved.pinyin == "ni"
        assert retrieved.tone == 3
        assert retrieved.duration_ms == 250

    def test_get_nonexistent(self, tmp_path):
        """Test getting non-existent entry returns None."""
        lexicon = SyllableLexicon.create_empty(tmp_path)
        assert lexicon.get("nonexistent", 1, "female") is None

    def test_list_syllables(self, tmp_path):
        """Test listing syllables."""
        lexicon = SyllableLexicon.create_empty(tmp_path)

        # Add entries for both voices
        lexicon.add(SyllableEntry("ni", 3, "female", "female/ni3.wav", 250))
        lexicon.add(SyllableEntry("ni", 3, "male", "male/ni3.wav", 260))
        lexicon.add(SyllableEntry("hao", 3, "female", "female/hao3.wav", 300))

        # All entries
        all_entries = lexicon.list_syllables()
        assert len(all_entries) == 3

        # Female only
        female_entries = lexicon.list_syllables(voice="female")
        assert len(female_entries) == 2

        # Male only
        male_entries = lexicon.list_syllables(voice="male")
        assert len(male_entries) == 1

    def test_list_voices(self, tmp_path):
        """Test listing available voices."""
        lexicon = SyllableLexicon.create_empty(tmp_path)
        lexicon.add(SyllableEntry("ni", 3, "female", "female/ni3.wav", 250))
        lexicon.add(SyllableEntry("ni", 3, "male", "male/ni3.wav", 260))

        voices = lexicon.list_voices()
        assert voices == {"female", "male"}

    def test_get_audio_path(self, tmp_path):
        """Test getting full audio path."""
        lexicon = SyllableLexicon.create_empty(tmp_path)
        lexicon.add(SyllableEntry("ni", 3, "female", "female/ni3.wav", 250))

        path = lexicon.get_audio_path("ni", 3, "female")
        assert path == tmp_path / "female/ni3.wav"

        # Non-existent
        assert lexicon.get_audio_path("nonexistent", 1, "female") is None

    def test_save_and_load(self, tmp_path):
        """Test saving and loading lexicon."""
        lexicon = SyllableLexicon.create_empty(tmp_path)

        # Add some entries
        lexicon.add(SyllableEntry("ni", 3, "female", "female/ni3.wav", 250))
        lexicon.add(SyllableEntry("hao", 3, "female", "female/hao3.wav", 300))
        lexicon.add(SyllableEntry("ni", 3, "male", "male/ni3.wav", 260))

        # Save
        lexicon.save()

        # Verify file exists
        assert (tmp_path / "lexicon.json").exists()

        # Load into new instance
        loaded = SyllableLexicon.load(tmp_path)

        assert len(loaded) == 3
        assert loaded.has("ni", 3, "female")
        assert loaded.has("hao", 3, "female")
        assert loaded.has("ni", 3, "male")

        # Check values preserved
        entry = loaded.get("ni", 3, "female")
        assert entry.duration_ms == 250

    def test_load_nonexistent(self, tmp_path):
        """Test loading from directory without lexicon.json."""
        lexicon = SyllableLexicon.load(tmp_path)
        assert len(lexicon) == 0

    def test_iteration(self, tmp_path):
        """Test iterating over lexicon."""
        lexicon = SyllableLexicon.create_empty(tmp_path)
        lexicon.add(SyllableEntry("ni", 3, "female", "female/ni3.wav", 250))
        lexicon.add(SyllableEntry("hao", 3, "female", "female/hao3.wav", 300))

        entries = list(lexicon)
        assert len(entries) == 2


class TestRemoveToneMarks:
    """Tests for _remove_tone_marks helper."""

    def test_remove_marks(self):
        """Test removing tone marks from pinyin."""
        assert _remove_tone_marks("nǐ") == "ni"
        assert _remove_tone_marks("hǎo") == "hao"
        assert _remove_tone_marks("xiè") == "xie"
        assert _remove_tone_marks("lǜ") == "lü"

    def test_numbered_tone(self):
        """Test removing numbered tones."""
        assert _remove_tone_marks("ni3") == "ni"
        assert _remove_tone_marks("hao3") == "hao"

    def test_no_tone(self):
        """Test pinyin without tone."""
        assert _remove_tone_marks("ni") == "ni"
        assert _remove_tone_marks("ma") == "ma"


class TestExtractUniqueSyllables:
    """Tests for extract_unique_syllables function."""

    def test_extract_from_sentences(self, tmp_path):
        """Test extracting unique syllables from sentence data."""
        # Create mock sentences.json
        sentences = {
            "items": [
                {
                    "text": "你好",
                    "romanization": "nǐ hǎo",
                },
                {
                    "text": "谢谢",
                    "romanization": "xiè xie",
                },
                {
                    "text": "你",
                    "romanization": "nǐ",
                },
            ]
        }
        json_path = tmp_path / "sentences.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(sentences, f)

        syllables = extract_unique_syllables(json_path)

        # Should have: (ni, 3), (hao, 3), (xie, 4), (xie, 0)
        assert len(syllables) == 4
        assert ("ni", 3) in syllables
        assert ("hao", 3) in syllables
        assert ("xie", 4) in syllables

    def test_empty_romanization(self, tmp_path):
        """Test handling sentences without romanization."""
        sentences = {
            "items": [
                {"text": "你好", "romanization": ""},
                {"text": "谢谢"},  # No romanization key
            ]
        }
        json_path = tmp_path / "sentences.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(sentences, f)

        syllables = extract_unique_syllables(json_path)
        assert len(syllables) == 0
