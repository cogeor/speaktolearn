"""Tests for data loading utilities."""

import pytest

from mandarin_grader.data.dataloader import (
    AudioSample,
    ContourDataset,
    SentenceDataset,
    parse_pinyin_syllable,
    parse_romanization,
)


class TestParsePinyinSyllable:
    """Tests for parse_pinyin_syllable function."""

    def test_tone_marks(self) -> None:
        """Test parsing syllables with tone marks."""
        assert parse_pinyin_syllable("nǐ") == ("n", "i", 3)
        assert parse_pinyin_syllable("hǎo") == ("h", "ao", 3)
        assert parse_pinyin_syllable("wǒ") == ("w", "o", 3)
        assert parse_pinyin_syllable("shì") == ("sh", "i", 4)

    def test_two_char_initials(self) -> None:
        """Test parsing syllables with zh, ch, sh initials."""
        assert parse_pinyin_syllable("zhōng") == ("zh", "ong", 1)
        assert parse_pinyin_syllable("chī") == ("ch", "i", 1)
        assert parse_pinyin_syllable("shū") == ("sh", "u", 1)

    def test_no_initial(self) -> None:
        """Test parsing syllables without initial consonant."""
        assert parse_pinyin_syllable("ài") == ("", "ai", 4)
        assert parse_pinyin_syllable("ér") == ("", "er", 2)

    def test_neutral_tone(self) -> None:
        """Test parsing neutral tone (no mark)."""
        assert parse_pinyin_syllable("de") == ("d", "e", 0)
        assert parse_pinyin_syllable("ma") == ("m", "a", 0)

    def test_tone_number_suffix(self) -> None:
        """Test parsing with tone number suffix."""
        assert parse_pinyin_syllable("ni3") == ("n", "i", 3)
        assert parse_pinyin_syllable("hao3") == ("h", "ao", 3)
        assert parse_pinyin_syllable("de0") == ("d", "e", 0)

    def test_umlaut(self) -> None:
        """Test parsing syllables with ü."""
        assert parse_pinyin_syllable("nǚ") == ("n", "ü", 3)
        assert parse_pinyin_syllable("lǜ") == ("l", "ü", 4)


class TestParseRomanization:
    """Tests for parse_romanization function."""

    def test_simple_phrase(self) -> None:
        """Test parsing a simple phrase."""
        syllables = parse_romanization("nǐ hǎo", "你好")
        assert len(syllables) == 2

        assert syllables[0].hanzi == "你"
        assert syllables[0].pinyin == "nǐ"
        assert syllables[0].initial == "n"
        assert syllables[0].final == "i"
        assert syllables[0].tone_underlying == 3

        assert syllables[1].hanzi == "好"
        assert syllables[1].pinyin == "hǎo"
        assert syllables[1].initial == "h"
        assert syllables[1].final == "ao"
        assert syllables[1].tone_underlying == 3

    def test_longer_phrase(self) -> None:
        """Test parsing a longer phrase."""
        syllables = parse_romanization("wǒ ài nǐ", "我爱你")
        assert len(syllables) == 3
        assert [s.hanzi for s in syllables] == ["我", "爱", "你"]
        assert [s.tone_underlying for s in syllables] == [3, 4, 3]

    def test_mixed_tones(self) -> None:
        """Test parsing with all tone types."""
        syllables = parse_romanization("zhōng guó rén", "中国人")
        assert len(syllables) == 3
        assert syllables[0].tone_underlying == 1  # zhōng
        assert syllables[1].tone_underlying == 2  # guó (actually 2nd tone)
        assert syllables[2].tone_underlying == 2  # rén

    def test_with_neutral_tone(self) -> None:
        """Test parsing with neutral tone."""
        syllables = parse_romanization("xiè xie", "谢谢")
        assert len(syllables) == 2
        assert syllables[0].tone_underlying == 4  # xiè
        assert syllables[1].tone_underlying == 0  # xie (neutral)


class TestAudioSample:
    """Tests for AudioSample dataclass."""

    def test_auto_parse_syllables(self, tmp_path) -> None:
        """Test that syllables are auto-parsed from romanization."""
        sample = AudioSample(
            id="test_001",
            audio_path=tmp_path / "test.wav",
            text="你好",
            romanization="nǐ hǎo",
            source="tts",
        )
        assert len(sample.syllables) == 2
        assert sample.syllables[0].hanzi == "你"
        assert sample.syllables[1].hanzi == "好"


class TestSentenceDataset:
    """Tests for SentenceDataset class."""

    def test_filter_by_ids(self, tmp_path) -> None:
        """Test filtering dataset by IDs."""
        samples = [
            AudioSample(id="ts_000001", audio_path=tmp_path / "1.wav", text="你好", romanization="nǐ hǎo"),
            AudioSample(id="ts_000002", audio_path=tmp_path / "2.wav", text="谢谢", romanization="xiè xie"),
            AudioSample(id="ts_000003", audio_path=tmp_path / "3.wav", text="再见", romanization="zài jiàn"),
        ]
        dataset = SentenceDataset(samples)

        filtered = dataset.filter_by_ids({"ts_000001", "ts_000003"})
        assert len(filtered) == 2
        assert {s.id for s in filtered} == {"ts_000001", "ts_000003"}

    def test_len_and_iter(self, tmp_path) -> None:
        """Test __len__ and __iter__."""
        samples = [
            AudioSample(id=f"ts_{i:06d}", audio_path=tmp_path / f"{i}.wav", text="测试", romanization="cè shì")
            for i in range(5)
        ]
        dataset = SentenceDataset(samples)

        assert len(dataset) == 5
        assert list(dataset) == samples


class TestContourDataset:
    """Tests for ContourDataset class."""

    def test_init_with_sentence_dataset(self, tmp_path) -> None:
        """Test initialization with SentenceDataset."""
        samples = [
            AudioSample(id="ts_000001", audio_path=tmp_path / "1.wav", text="你好", romanization="nǐ hǎo"),
            AudioSample(id="ts_000002", audio_path=tmp_path / "2.wav", text="谢谢", romanization="xiè xie"),
        ]
        sentence_ds = SentenceDataset(samples)
        contour_ds = ContourDataset(sentence_ds)

        assert len(contour_ds) == 2
        assert contour_ds.samples == samples

    def test_init_with_sample_list(self, tmp_path) -> None:
        """Test initialization with list of AudioSample."""
        samples = [
            AudioSample(id="ts_000001", audio_path=tmp_path / "1.wav", text="你好", romanization="nǐ hǎo"),
        ]
        contour_ds = ContourDataset(samples)

        assert len(contour_ds) == 1
        assert contour_ds.samples == samples

    def test_default_k_value(self, tmp_path) -> None:
        """Test default k value is 20."""
        samples = [
            AudioSample(id="ts_000001", audio_path=tmp_path / "1.wav", text="你好", romanization="nǐ hǎo"),
        ]
        contour_ds = ContourDataset(samples)

        assert contour_ds.k == 20

    def test_custom_k_value(self, tmp_path) -> None:
        """Test custom k value."""
        samples = [
            AudioSample(id="ts_000001", audio_path=tmp_path / "1.wav", text="你好", romanization="nǐ hǎo"),
        ]
        contour_ds = ContourDataset(samples, k=30)

        assert contour_ds.k == 30

    def test_empty_dataset(self) -> None:
        """Test with empty sample list."""
        contour_ds = ContourDataset([])

        assert len(contour_ds) == 0
        assert contour_ds.samples == []
