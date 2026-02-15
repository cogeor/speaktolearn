"""Tests for SyllablePredictorV5."""

import pytest
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from mandarin_grader.model.syllable_predictor_v5 import (
    SyllablePredictorV5,
    SyllablePredictorConfigV5,
    PredictorOutput,
)


@pytest.fixture
def config():
    """Create test config with smaller dimensions for speed."""
    return SyllablePredictorConfigV5(
        d_model=64,
        n_heads=4,
        n_layers=2,
        dim_feedforward=128,
        max_audio_frames=100,  # Smaller for test
        max_positions=10,
    )


@pytest.fixture
def model(config):
    """Create model for testing."""
    return SyllablePredictorV5(config)


class TestSyllablePredictorV5:
    """Tests for V5 model."""

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_model_instantiation(self, config):
        """Test model creates without errors."""
        model = SyllablePredictorV5(config)
        assert model is not None
        assert model.config == config

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_forward_pass_shapes(self, model, config):
        """Test forward pass produces correct output shapes."""
        batch_size = 4
        n_mels = config.n_mels
        time_frames = 100  # Within max_audio_frames

        mel = torch.randn(batch_size, n_mels, time_frames)
        position = torch.tensor([[0], [1], [2], [3]])

        syllable_logits, tone_logits = model(mel, position)

        assert syllable_logits.shape == (batch_size, config.n_syllables)
        assert tone_logits.shape == (batch_size, config.n_tones)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_forward_with_mask(self, model, config):
        """Test forward pass with audio mask."""
        batch_size = 2
        mel = torch.randn(batch_size, config.n_mels, 80)
        position = torch.tensor([[0], [1]])
        audio_mask = torch.zeros(batch_size, 80, dtype=torch.bool)
        audio_mask[:, 60:] = True  # Mask last 20 frames

        syllable_logits, tone_logits = model(mel, position, audio_mask)

        assert syllable_logits.shape == (batch_size, config.n_syllables)
        assert tone_logits.shape == (batch_size, config.n_tones)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_predict_method(self, model, config):
        """Test predict convenience method."""
        mel = torch.randn(config.n_mels, 50)  # Single sample, no batch dim
        position = torch.tensor([0])

        output = model.predict(mel, position)

        assert isinstance(output, PredictorOutput)
        assert output.syllable_pred is not None
        assert output.tone_pred is not None
        assert 0 <= output.syllable_pred < config.n_syllables
        assert 0 <= output.tone_pred < config.n_tones
        assert output.syllable_prob is not None
        assert output.tone_prob is not None

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_numpy_input(self, model, config):
        """Test model accepts numpy arrays."""
        mel = np.random.randn(2, config.n_mels, 60).astype(np.float32)
        position = np.array([[0], [1]])

        syllable_logits, tone_logits = model(mel, position)

        assert syllable_logits.shape == (2, config.n_syllables)
        assert tone_logits.shape == (2, config.n_tones)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_parameter_count(self, model):
        """Test parameter counting."""
        total, trainable = model.count_parameters()
        assert total > 0
        assert trainable == total  # All params should be trainable

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_full_size_model_params(self):
        """Test full-size model stays under target."""
        config = SyllablePredictorConfigV5()  # Default config
        model = SyllablePredictorV5(config)
        total, _ = model.count_parameters()

        # Target: <10M params
        assert total < 10_000_000, f"Model has {total:,} params, exceeds 10M target"
        print(f"V5 model params: {total:,} ({total * 4 / 1024 / 1024:.2f} MB)")

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_position_embedding_range(self, model, config):
        """Test position embedding handles valid range."""
        mel = torch.randn(1, config.n_mels, 50)

        # Test various positions
        for pos in [0, 1, config.max_positions - 1]:
            position = torch.tensor([[pos]])
            syllable_logits, tone_logits = model(mel, position)
            assert syllable_logits.shape == (1, config.n_syllables)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_different_audio_lengths(self, model, config):
        """Test model handles various audio lengths."""
        batch_size = 2
        position = torch.tensor([[0], [1]])

        # Test short audio
        mel_short = torch.randn(batch_size, config.n_mels, 20)
        out_short = model(mel_short, position)
        assert out_short[0].shape == (batch_size, config.n_syllables)

        # Test max audio
        mel_max = torch.randn(batch_size, config.n_mels, config.max_audio_frames)
        out_max = model(mel_max, position)
        assert out_max[0].shape == (batch_size, config.n_syllables)

        # Test over max (should truncate)
        mel_over = torch.randn(batch_size, config.n_mels, config.max_audio_frames + 50)
        out_over = model(mel_over, position)
        assert out_over[0].shape == (batch_size, config.n_syllables)


class TestConfigV5:
    """Tests for V5 config."""

    def test_default_config(self):
        """Test default config values."""
        config = SyllablePredictorConfigV5()
        assert config.max_audio_frames == 1000
        assert config.max_positions == 30
        assert config.n_mels == 80

    def test_custom_config(self):
        """Test custom config."""
        config = SyllablePredictorConfigV5(
            max_audio_frames=500,
            max_positions=20,
        )
        assert config.max_audio_frames == 500
        assert config.max_positions == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
