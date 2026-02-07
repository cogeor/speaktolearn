"""Unit tests for TextGenerator with mocked OpenAI client."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from sentence_gen.generators.text_generator import TextGenerator
from sentence_gen.config import Config, OpenAIConfig, TTSConfig


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return Config(
        openai=OpenAIConfig(api_key="test-key", model="gpt-4"),
        tts=TTSConfig(),
    )


def test_generate_creates_dataset(mock_config):
    """Verify generate() creates a dataset with correct structure."""
    # Create generator with mocked OpenAI client
    with patch("sentence_gen.generators.text_generator.OpenAI") as mock_openai_class:
        # Set up mock client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Create generator (this will use mocked OpenAI)
        generator = TextGenerator(mock_config)

        # Set up mock response
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content='{"sentences": [{"text": "你好", "romanization": "ni hao", "translation": "Hello", "tags": ["greeting"]}]}'
                )
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        # Mock prompt loading
        with patch.object(generator, "_load_prompt", return_value="Test prompt"):
            dataset = generator.generate(
                language="zh-CN",
                count=1,
                tags=["greeting"],
                difficulty=1,
            )

    # Verify dataset structure
    assert dataset.dataset_id == "zh_cn_v1"
    assert dataset.language == "zh-CN"
    assert len(dataset.items) == 1

    # Verify item content
    item = dataset.items[0]
    assert item.text == "你好"
    assert item.id == "ts_000001"
    assert item.gloss.get("en") == "Hello"


def test_generate_batches_large_counts(mock_config):
    """Verify generate() batches requests for large counts."""
    with patch("sentence_gen.generators.text_generator.OpenAI") as mock_openai_class:
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        generator = TextGenerator(mock_config)

        # Set up mock response that returns one item per call
        def create_mock_response(*args, **kwargs):
            mock_response = Mock()
            mock_response.choices = [
                Mock(
                    message=Mock(
                        content='{"sentences": [{"text": "测试", "romanization": "ce shi", "translation": "Test"}]}'
                    )
                )
            ]
            return mock_response

        mock_client.chat.completions.create.side_effect = create_mock_response

        with patch.object(generator, "_load_prompt", return_value="Test prompt"):
            dataset = generator.generate(
                language="zh-CN",
                count=25,  # Should trigger batching (batch_size=20)
                tags=[],
            )

    # Verify API was called multiple times for batching
    assert mock_client.chat.completions.create.call_count == 2  # 20 + 5


def test_build_system_prompt(mock_config):
    """Verify system prompt is correctly built."""
    with patch("sentence_gen.generators.text_generator.OpenAI"):
        generator = TextGenerator(mock_config)

    prompt = generator._build_system_prompt(
        language="zh-CN",
        tags=["hsk1", "daily"],
        difficulty=2,
    )

    assert "zh-CN" in prompt
    assert "2/5" in prompt
    assert "hsk1" in prompt
    assert "daily" in prompt
