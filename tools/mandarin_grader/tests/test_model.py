"""Tests for the syllable-tone neural network model."""

import numpy as np
import pytest

from mandarin_grader.model import (
    ModelConfig,
    ModelOutput,
    SyllableToneModel,
    extract_mel_spectrogram,
    TORCH_AVAILABLE,
)


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ModelConfig()

        assert config.n_mels == 80
        assert config.sample_rate == 16000
        assert config.hop_length == 160
        assert config.n_tones == 5
        assert config.cnn_channels == [32, 64, 128]

    def test_custom_config(self):
        """Test custom configuration."""
        config = ModelConfig(n_mels=40, lstm_hidden=64)

        assert config.n_mels == 40
        assert config.lstm_hidden == 64


class TestExtractMelSpectrogram:
    """Tests for mel spectrogram extraction."""

    def test_basic_extraction(self):
        """Test basic mel spectrogram extraction."""
        config = ModelConfig()
        # Create 1 second of audio at 16kHz
        audio = np.random.randn(16000).astype(np.float32) * 0.5

        mel = extract_mel_spectrogram(audio, config)

        assert mel.shape[0] == config.n_mels
        assert mel.shape[1] > 0
        assert mel.dtype == np.float32

    def test_short_audio(self):
        """Test with very short audio."""
        config = ModelConfig()
        audio = np.random.randn(100).astype(np.float32) * 0.5

        mel = extract_mel_spectrogram(audio, config)

        assert mel.shape[0] == config.n_mels
        assert mel.shape[1] >= 1

    def test_expected_time_frames(self):
        """Test expected number of time frames."""
        config = ModelConfig()
        duration_sec = 0.5  # 500ms
        audio = np.random.randn(int(config.sample_rate * duration_sec)).astype(np.float32)

        mel = extract_mel_spectrogram(audio, config)

        # Expected frames: (samples - win_length) / hop_length + 1
        expected_frames = (len(audio) - config.win_length) // config.hop_length + 1
        assert mel.shape[1] == expected_frames


class TestSyllableToneModel:
    """Tests for SyllableToneModel."""

    def test_model_creation(self):
        """Test model instantiation."""
        config = ModelConfig()
        model = SyllableToneModel(config)

        assert model.config == config

    def test_forward_shape(self):
        """Test forward pass output shapes."""
        config = ModelConfig()
        model = SyllableToneModel(config)

        # Create batch of mel spectrograms
        batch_size = 2
        n_frames = 100
        mel = np.random.randn(batch_size, config.n_mels, n_frames).astype(np.float32)

        boundary_logits, tone_logits = model.forward(mel)

        # Check output shapes
        assert boundary_logits.shape[0] == batch_size
        assert boundary_logits.shape[2] == 2  # boundary/no-boundary

        assert tone_logits.shape[0] == batch_size
        assert tone_logits.shape[2] == config.n_tones

    def test_predict(self):
        """Test predict method returns ModelOutput."""
        config = ModelConfig()
        model = SyllableToneModel(config)

        batch_size = 1
        n_frames = 50
        mel = np.random.randn(batch_size, config.n_mels, n_frames).astype(np.float32)

        output = model.predict(mel)

        assert isinstance(output, ModelOutput)
        assert output.boundary_logits is not None
        assert output.tone_logits is not None

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_torch_model_parameters(self):
        """Test PyTorch model has trainable parameters."""
        config = ModelConfig()
        model = SyllableToneModel(config)

        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())

        # Should have some parameters
        assert param_count > 0


class TestModelOutput:
    """Tests for ModelOutput dataclass."""

    def test_output_creation(self):
        """Test creating ModelOutput."""
        boundary_logits = np.random.randn(1, 50, 2).astype(np.float32)
        tone_logits = np.random.randn(1, 50, 5).astype(np.float32)

        output = ModelOutput(
            boundary_logits=boundary_logits,
            tone_logits=tone_logits,
        )

        assert output.boundary_logits.shape == (1, 50, 2)
        assert output.tone_logits.shape == (1, 50, 5)
        assert output.syllable_starts is None
        assert output.tone_predictions is None

    def test_output_with_decoded(self):
        """Test ModelOutput with decoded predictions."""
        output = ModelOutput(
            boundary_logits=np.random.randn(1, 50, 2).astype(np.float32),
            tone_logits=np.random.randn(1, 50, 5).astype(np.float32),
            syllable_starts=[[0, 10, 25, 40]],
            tone_predictions=[[3, 2, 4, 1]],
        )

        assert output.syllable_starts[0] == [0, 10, 25, 40]
        assert output.tone_predictions[0] == [3, 2, 4, 1]


class TestEndToEnd:
    """End-to-end tests with audio to prediction."""

    def test_audio_to_prediction(self):
        """Test full pipeline from audio to predictions."""
        config = ModelConfig()
        model = SyllableToneModel(config)

        # Create synthetic audio (1 second)
        audio = np.random.randn(16000).astype(np.float32) * 0.3

        # Extract mel spectrogram
        mel = extract_mel_spectrogram(audio, config)

        # Add batch dimension
        mel_batch = mel[np.newaxis, :, :]

        # Get predictions
        output = model.predict(mel_batch)

        assert output.boundary_logits.shape[0] == 1
        assert output.boundary_logits.shape[1] > 0

    def test_multiple_samples(self):
        """Test processing multiple samples in batch."""
        config = ModelConfig()
        model = SyllableToneModel(config)

        # Create batch of audio
        batch_size = 3
        audio_length = 8000  # 0.5 seconds

        mels = []
        for _ in range(batch_size):
            audio = np.random.randn(audio_length).astype(np.float32) * 0.3
            mel = extract_mel_spectrogram(audio, config)
            mels.append(mel)

        # Pad to same length
        max_frames = max(m.shape[1] for m in mels)
        mel_batch = np.zeros((batch_size, config.n_mels, max_frames), dtype=np.float32)
        for i, mel in enumerate(mels):
            mel_batch[i, :, :mel.shape[1]] = mel

        # Get predictions
        output = model.predict(mel_batch)

        assert output.boundary_logits.shape[0] == batch_size
