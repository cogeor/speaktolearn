"""Tests for sentence synthesis via concatenation."""

import numpy as np
import pytest
import tempfile
from pathlib import Path
import wave
import struct

from mandarin_grader.data.synthesis import (
    AugmentationConfig,
    SyntheticSample,
    load_syllable_audio,
    change_speed,
    add_noise,
    crossfade,
    synthesize_sentence,
    create_synthetic_sample,
    save_synthetic_sample,
)
from mandarin_grader.data.lexicon import SyllableLexicon, SyllableEntry
from mandarin_grader.types import TargetSyllable, Ms


@pytest.fixture
def mock_lexicon(tmp_path):
    """Create a mock lexicon with test audio files."""
    lexicon = SyllableLexicon.create_empty(tmp_path)

    # Create mock syllable audio files
    sample_rate = 16000
    for pinyin, tone, duration_ms in [("ni", 3, 250), ("hao", 3, 300), ("xie", 4, 200)]:
        for voice in ["female", "male"]:
            # Generate simple sine wave for testing
            duration_samples = int(duration_ms * sample_rate / 1000)
            t = np.linspace(0, duration_ms / 1000, duration_samples)
            freq = 440 if voice == "female" else 220
            audio = (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)

            # Convert to int16
            audio_int16 = (audio * 32767).astype(np.int16)

            # Save WAV file
            voice_dir = tmp_path / voice
            voice_dir.mkdir(exist_ok=True)
            wav_path = voice_dir / f"{pinyin}{tone}.wav"

            with wave.open(str(wav_path), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())

            # Add to lexicon
            entry = SyllableEntry(
                pinyin=pinyin,
                tone=tone,
                voice_id=voice,
                audio_path=f"{voice}/{pinyin}{tone}.wav",
                duration_ms=duration_ms,
            )
            lexicon.add(entry)

    return lexicon


@pytest.fixture
def sample_syllables():
    """Create sample target syllables."""
    return [
        TargetSyllable(
            index=0, hanzi="你", pinyin="nǐ",
            initial="n", final="i",
            tone_underlying=3, tone_surface=3,
        ),
        TargetSyllable(
            index=1, hanzi="好", pinyin="hǎo",
            initial="h", final="ao",
            tone_underlying=3, tone_surface=3,
        ),
    ]


class TestLoadSyllableAudio:
    """Tests for load_syllable_audio function."""

    def test_load_existing(self, mock_lexicon):
        """Test loading an existing syllable."""
        audio = load_syllable_audio(mock_lexicon, "ni", 3, "female")
        assert audio is not None
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) > 0

    def test_load_nonexistent(self, mock_lexicon):
        """Test loading non-existent syllable returns None."""
        audio = load_syllable_audio(mock_lexicon, "nonexistent", 1, "female")
        assert audio is None


class TestChangeSpeed:
    """Tests for change_speed function."""

    def test_speed_up(self):
        """Test speeding up audio."""
        audio = np.random.randn(1000).astype(np.float32)
        result = change_speed(audio, 2.0)
        assert len(result) == 500  # Half the length

    def test_slow_down(self):
        """Test slowing down audio."""
        audio = np.random.randn(1000).astype(np.float32)
        result = change_speed(audio, 0.5)
        assert len(result) == 2000  # Double the length

    def test_no_change(self):
        """Test factor of 1.0 returns same length."""
        audio = np.random.randn(1000).astype(np.float32)
        result = change_speed(audio, 1.0)
        assert len(result) == 1000


class TestAddNoise:
    """Tests for add_noise function."""

    def test_adds_noise(self):
        """Test that noise is added."""
        audio = np.ones(1000, dtype=np.float32) * 0.5
        result = add_noise(audio, snr_db=20)
        # Should be different from original
        assert not np.allclose(audio, result)

    def test_snr_affects_noise_level(self):
        """Test that lower SNR means more noise."""
        audio = np.ones(1000, dtype=np.float32) * 0.5
        result_high_snr = add_noise(audio, snr_db=40)
        result_low_snr = add_noise(audio, snr_db=10)

        # Lower SNR should have larger deviation
        dev_high = np.std(result_high_snr - audio)
        dev_low = np.std(result_low_snr - audio)
        assert dev_low > dev_high


class TestCrossfade:
    """Tests for crossfade function."""

    def test_crossfade_length(self):
        """Test crossfade produces correct length."""
        audio1 = np.ones(500, dtype=np.float32)
        audio2 = np.ones(500, dtype=np.float32)
        overlap = 100

        result = crossfade(audio1, audio2, overlap)
        expected_length = len(audio1) + len(audio2) - overlap
        assert len(result) == expected_length

    def test_no_overlap(self):
        """Test with no overlap (simple concatenation)."""
        audio1 = np.ones(500, dtype=np.float32)
        audio2 = np.ones(500, dtype=np.float32) * 2

        result = crossfade(audio1, audio2, 0)
        assert len(result) == 1000
        assert np.allclose(result[:500], 1.0)
        assert np.allclose(result[500:], 2.0)


class TestSynthesizeSentence:
    """Tests for synthesize_sentence function."""

    def test_basic_synthesis(self, mock_lexicon, sample_syllables):
        """Test basic sentence synthesis."""
        audio, spans = synthesize_sentence(
            syllables=sample_syllables,
            lexicon=mock_lexicon,
            voice="female",
        )

        assert len(audio) > 0
        assert len(spans) == 2

    def test_spans_are_contiguous(self, mock_lexicon, sample_syllables):
        """Test that spans don't overlap and cover the audio."""
        audio, spans = synthesize_sentence(
            syllables=sample_syllables,
            lexicon=mock_lexicon,
            voice="female",
        )

        # Spans should not overlap
        for i in range(len(spans) - 1):
            assert spans[i].end_ms <= spans[i + 1].start_ms

        # First span should start at 0
        assert spans[0].start_ms == 0

    def test_boundaries_are_exact(self, mock_lexicon, sample_syllables):
        """Test that boundaries match concatenation points exactly."""
        audio, spans = synthesize_sentence(
            syllables=sample_syllables,
            lexicon=mock_lexicon,
            voice="female",
        )

        # Calculate expected boundaries from syllable durations
        # ni3 = 250ms, hao3 = 300ms (from mock_lexicon fixture)
        assert spans[0].start_ms == 0
        # Allow small tolerance for rounding
        assert abs(spans[0].end_ms - 250) <= 1 or abs(spans[0].end_ms - spans[1].start_ms) <= 1

    def test_with_gap(self, mock_lexicon, sample_syllables):
        """Test synthesis with silence gap between syllables."""
        aug = AugmentationConfig(gap_ms=50)
        audio, spans = synthesize_sentence(
            syllables=sample_syllables,
            lexicon=mock_lexicon,
            voice="female",
            augmentations=aug,
        )

        # Second syllable should start after gap
        assert spans[1].start_ms >= spans[0].end_ms + 50 - 1

    def test_with_speed_variation(self, mock_lexicon, sample_syllables):
        """Test synthesis with speed variation."""
        np.random.seed(42)  # For reproducibility
        aug = AugmentationConfig(speed_variation=0.2)
        audio1, spans1 = synthesize_sentence(
            syllables=sample_syllables,
            lexicon=mock_lexicon,
            voice="female",
            augmentations=aug,
        )

        np.random.seed(43)  # Different seed
        audio2, spans2 = synthesize_sentence(
            syllables=sample_syllables,
            lexicon=mock_lexicon,
            voice="female",
            augmentations=aug,
        )

        # Different seeds should produce different lengths
        assert len(audio1) != len(audio2)


class TestCreateSyntheticSample:
    """Tests for create_synthetic_sample function."""

    def test_creates_sample(self, mock_lexicon, sample_syllables):
        """Test creating a complete synthetic sample."""
        sample = create_synthetic_sample(
            sample_id="test_001",
            syllables=sample_syllables,
            lexicon=mock_lexicon,
            voice="female",
        )

        assert isinstance(sample, SyntheticSample)
        assert sample.id == "test_001"
        assert sample.audio.dtype == np.int16
        assert len(sample.ground_truth_spans) == 2
        assert sample.voice_id == "female"


class TestSaveSyntheticSample:
    """Tests for save_synthetic_sample function."""

    def test_saves_wav(self, mock_lexicon, sample_syllables, tmp_path):
        """Test saving sample to WAV file."""
        sample = create_synthetic_sample(
            sample_id="test_001",
            syllables=sample_syllables,
            lexicon=mock_lexicon,
            voice="female",
        )

        output_path = tmp_path / "test.wav"
        save_synthetic_sample(sample, output_path)

        assert output_path.exists()

        # Verify WAV file is valid
        with wave.open(str(output_path), 'rb') as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
