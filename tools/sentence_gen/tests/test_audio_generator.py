"""Unit tests for AudioGenerator with mocked OpenAI TTS API."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from sentence_gen.generators.audio_generator import AudioGenerator
from sentence_gen.config import Config, OpenAIConfig, TTSConfig
from sentence_gen.models.dataset import Dataset
from sentence_gen.models.text_sequence import TextSequence
from sentence_gen.models.voice import VoiceRef


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return Config(
        openai=OpenAIConfig(api_key="test-key", model="gpt-4"),
        tts=TTSConfig(
            provider="openai",
            openai_voice_female="nova",
            openai_voice_male="onyx",
        ),
    )


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    dataset = Dataset(dataset_id="test_v1", language="zh-CN")
    dataset.add_item(TextSequence.create(index=1, text="你好"))
    return dataset


def test_audio_generator_init(mock_config):
    """Verify AudioGenerator initializes with config."""
    with patch("sentence_gen.generators.audio_generator.OpenAI") as mock_openai_class:
        generator = AudioGenerator(mock_config)

        mock_openai_class.assert_called_once_with(api_key="test-key")
        assert generator.config == mock_config


def test_generate_audio_creates_file(mock_config, sample_dataset, tmp_path):
    """Verify audio generation creates file and returns path."""
    with patch("sentence_gen.generators.audio_generator.OpenAI") as mock_openai_class:
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Mock TTS response
        mock_response = Mock()
        mock_response.iter_bytes.return_value = [b"fake audio data"]
        mock_client.audio.speech.create.return_value = mock_response

        generator = AudioGenerator(mock_config)

        item = sample_dataset.items[0]
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "female").mkdir()

        result = generator._generate_audio(
            item=item,
            voice_name="female",
            voice_id="nova",
            output_dir=output_dir,
        )

        # Verify file was created
        assert result is not None
        assert result.exists()
        assert result.name == "ts_000001.opus"

        # Verify API was called correctly
        mock_client.audio.speech.create.assert_called_once_with(
            model="tts-1",
            voice="nova",
            input="你好",
            response_format="opus",
        )


def test_generate_audio_skips_existing(mock_config, sample_dataset, tmp_path):
    """Verify audio generation skips existing files."""
    with patch("sentence_gen.generators.audio_generator.OpenAI") as mock_openai_class:
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        generator = AudioGenerator(mock_config)

        item = sample_dataset.items[0]
        output_dir = tmp_path / "output"
        (output_dir / "female").mkdir(parents=True)

        # Create existing file
        existing_file = output_dir / "female" / f"{item.id}.opus"
        existing_file.write_bytes(b"existing audio")

        result = generator._generate_audio(
            item=item,
            voice_name="female",
            voice_id="nova",
            output_dir=output_dir,
        )

        # Should return existing file without calling API
        assert result == existing_file
        mock_client.audio.speech.create.assert_not_called()


def test_add_audio_ref_creates_voice_ref(mock_config, sample_dataset, tmp_path):
    """Verify VoiceRef is correctly created and added to item."""
    with patch("sentence_gen.generators.audio_generator.OpenAI"):
        generator = AudioGenerator(mock_config)

    item = sample_dataset.items[0]
    output_dir = tmp_path / "output"
    audio_path = output_dir / "female" / "ts_000001.opus"

    # Add audio reference
    generator._add_audio_ref(item, "female", audio_path, output_dir)

    # Verify example_audio was created
    assert item.example_audio is not None
    assert len(item.example_audio.voices) == 1

    voice_ref = item.example_audio.voices[0]
    assert voice_ref.id == "f1"
    assert voice_ref.label == {"en": "Female"}
    assert voice_ref.uri == "assets://examples/female/ts_000001.opus"


def test_add_audio_ref_replaces_existing_voice(mock_config, sample_dataset, tmp_path):
    """Verify adding same voice replaces existing reference."""
    with patch("sentence_gen.generators.audio_generator.OpenAI"):
        generator = AudioGenerator(mock_config)

    item = sample_dataset.items[0]
    output_dir = tmp_path / "output"
    audio_path = output_dir / "female" / "ts_000001.opus"

    # Add audio reference twice
    generator._add_audio_ref(item, "female", audio_path, output_dir)
    generator._add_audio_ref(item, "female", audio_path, output_dir)

    # Should still have only one voice reference
    assert len(item.example_audio.voices) == 1


def test_add_audio_ref_multiple_voices(mock_config, sample_dataset, tmp_path):
    """Verify multiple voice references can be added."""
    with patch("sentence_gen.generators.audio_generator.OpenAI"):
        generator = AudioGenerator(mock_config)

    item = sample_dataset.items[0]
    output_dir = tmp_path / "output"

    # Add both female and male voices
    generator._add_audio_ref(
        item, "female", output_dir / "female" / "ts_000001.opus", output_dir
    )
    generator._add_audio_ref(
        item, "male", output_dir / "male" / "ts_000001.opus", output_dir
    )

    # Should have two voice references
    assert len(item.example_audio.voices) == 2
    voice_ids = {v.id for v in item.example_audio.voices}
    assert voice_ids == {"f1", "m1"}


def test_generate_all_creates_directories(mock_config, sample_dataset, tmp_path):
    """Verify generate_all creates output directories."""
    with patch("sentence_gen.generators.audio_generator.OpenAI") as mock_openai_class:
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Mock TTS response
        mock_response = Mock()
        mock_response.iter_bytes.return_value = [b"fake audio data"]
        mock_client.audio.speech.create.return_value = mock_response

        generator = AudioGenerator(mock_config)
        output_dir = tmp_path / "examples"

        generator.generate_all(
            dataset=sample_dataset,
            output_dir=output_dir,
            voices=["female", "male"],
        )

        # Verify directories were created
        assert (output_dir / "female").exists()
        assert (output_dir / "male").exists()
