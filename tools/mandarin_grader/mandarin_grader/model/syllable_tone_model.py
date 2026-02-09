"""End-to-end deep learning model for syllable segmentation and tone classification.

This module defines a neural network architecture that takes mel spectrograms
as input and outputs:
1. Syllable boundary predictions (via CTC or frame-level classification)
2. Tone classification per syllable (tones 0-4)

The model is designed to be portable to mobile devices.

Architecture: Small CNN-BiLSTM with CTC output

NOTE: This module requires PyTorch. If PyTorch is not available,
a numpy-based stub implementation is provided for testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple
import numpy as np
from numpy.typing import NDArray


# Model configuration
@dataclass
class ModelConfig:
    """Configuration for the syllable-tone model."""

    # Input features
    n_mels: int = 80  # Number of mel frequency bins
    sample_rate: int = 16000
    hop_length: int = 160  # 10ms at 16kHz
    win_length: int = 400  # 25ms at 16kHz

    # Architecture
    cnn_channels: list[int] = None  # Will be set in __post_init__
    lstm_hidden: int = 128
    lstm_layers: int = 2
    dropout: float = 0.1

    # Output
    n_tones: int = 5  # Tones 0-4
    blank_token: int = 0  # CTC blank token

    # Training
    max_syllables: int = 20  # Max syllables per sentence

    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [32, 64, 128]


@dataclass
class ModelOutput:
    """Output from the syllable-tone model."""

    # Frame-level predictions
    boundary_logits: NDArray[np.float32]  # [batch, time, 2] - syllable boundary prob
    tone_logits: NDArray[np.float32]  # [batch, time, n_tones]

    # Decoded outputs (after post-processing)
    syllable_starts: list[list[int]] | None = None  # Frame indices of syllable starts
    tone_predictions: list[list[int]] | None = None  # Predicted tones per syllable


# Try to import PyTorch, fall back to numpy stub
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class ConvBlock(nn.Module):
        """Convolutional block with BatchNorm and ReLU."""

        def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.pool = nn.MaxPool2d(kernel_size=(2, 1))  # Pool in frequency only

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.pool(x)
            return x

    class SyllableToneModel(nn.Module):
        """End-to-end model for syllable segmentation and tone classification.

        Architecture:
        1. CNN encoder for local feature extraction from mel spectrogram
        2. BiLSTM for sequential modeling
        3. Two output heads:
           - Boundary head: predicts syllable boundaries (frame-level)
           - Tone head: predicts tone for each frame (aggregated per syllable)
        """

        def __init__(self, config: ModelConfig | None = None):
            super().__init__()

            if config is None:
                config = ModelConfig()
            self.config = config

            # CNN encoder
            self.conv_layers = nn.ModuleList()
            in_channels = 1
            for out_channels in config.cnn_channels:
                self.conv_layers.append(ConvBlock(in_channels, out_channels))
                in_channels = out_channels

            # Calculate CNN output size
            # After 3 pooling layers in frequency: 80 -> 40 -> 20 -> 10
            cnn_freq_out = config.n_mels // (2 ** len(config.cnn_channels))
            cnn_out_dim = config.cnn_channels[-1] * cnn_freq_out

            # BiLSTM
            self.lstm = nn.LSTM(
                input_size=cnn_out_dim,
                hidden_size=config.lstm_hidden,
                num_layers=config.lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=config.dropout if config.lstm_layers > 1 else 0,
            )

            lstm_out_dim = config.lstm_hidden * 2  # Bidirectional

            # Output heads
            self.boundary_head = nn.Linear(lstm_out_dim, 2)  # boundary/no-boundary
            self.tone_head = nn.Linear(lstm_out_dim, config.n_tones)

            # Dropout
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, mel: torch.Tensor | np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
            """Forward pass.

            Args:
                mel: Mel spectrogram [batch, n_mels, time] (tensor or numpy array)

            Returns:
                Tuple of (boundary_logits, tone_logits)
                - boundary_logits: [batch, time, 2]
                - tone_logits: [batch, time, n_tones]
            """
            # Convert numpy to tensor if needed
            if isinstance(mel, np.ndarray):
                mel = torch.from_numpy(mel).float()

            batch_size, n_mels, time = mel.shape

            # Add channel dimension: [batch, 1, n_mels, time]
            x = mel.unsqueeze(1)

            # CNN encoder
            for conv in self.conv_layers:
                x = conv(x)

            # Reshape for LSTM: [batch, time, features]
            # x is [batch, channels, freq, time]
            x = x.permute(0, 3, 1, 2)  # [batch, time, channels, freq]
            x = x.reshape(batch_size, -1, x.shape[2] * x.shape[3])

            # Adjust time dimension if needed (due to any conv/pool stride in time)
            actual_time = x.shape[1]

            # BiLSTM
            x, _ = self.lstm(x)
            x = self.dropout(x)

            # Output heads
            boundary_logits = self.boundary_head(x)  # [batch, time, 2]
            tone_logits = self.tone_head(x)  # [batch, time, n_tones]

            return boundary_logits, tone_logits

        def predict(self, mel: torch.Tensor | np.ndarray) -> ModelOutput:
            """Make predictions with post-processing.

            Args:
                mel: Mel spectrogram [batch, n_mels, time] (tensor or numpy array)

            Returns:
                ModelOutput with logits and decoded predictions
            """
            # Convert numpy to tensor if needed
            if isinstance(mel, np.ndarray):
                mel = torch.from_numpy(mel).float()

            batch_size = mel.shape[0]

            with torch.no_grad():
                boundary_logits, tone_logits = self.forward(mel)

            # Decode syllable boundaries
            boundary_probs = torch.softmax(boundary_logits, dim=-1)
            boundary_preds = boundary_probs[:, :, 1] > 0.5  # Threshold at 0.5

            # Find syllable start frames
            syllable_starts = []
            for b in range(batch_size):
                starts = torch.where(boundary_preds[b])[0].tolist()
                if not starts or starts[0] != 0:
                    starts = [0] + starts
                syllable_starts.append(starts)

            # Decode tones per syllable
            tone_probs = torch.softmax(tone_logits, dim=-1)
            tone_predictions = []

            for b in range(batch_size):
                starts = syllable_starts[b]
                tones = []

                for i in range(len(starts)):
                    start = starts[i]
                    end = starts[i + 1] if i + 1 < len(starts) else tone_probs.shape[1]

                    # Average tone probabilities over syllable frames
                    if end > start:
                        syllable_tone_probs = tone_probs[b, start:end].mean(dim=0)
                        predicted_tone = syllable_tone_probs.argmax().item()
                    else:
                        predicted_tone = 0

                    tones.append(predicted_tone)

                tone_predictions.append(tones)

            return ModelOutput(
                boundary_logits=boundary_logits.numpy(),
                tone_logits=tone_logits.numpy(),
                syllable_starts=syllable_starts,
                tone_predictions=tone_predictions,
            )

else:
    # Numpy stub implementation for testing without PyTorch
    class SyllableToneModel:
        """Stub implementation when PyTorch is not available."""

        def __init__(self, config: ModelConfig | None = None):
            if config is None:
                config = ModelConfig()
            self.config = config
            self._parameters = {}  # Stub parameters

        def forward(self, mel: NDArray[np.float32]) -> Tuple[NDArray, NDArray]:
            """Stub forward pass returning random outputs."""
            batch_size = mel.shape[0]
            time = mel.shape[2]

            # Random predictions for testing
            boundary_logits = np.random.randn(batch_size, time, 2).astype(np.float32)
            tone_logits = np.random.randn(batch_size, time, self.config.n_tones).astype(np.float32)

            return boundary_logits, tone_logits

        def predict(self, mel: NDArray[np.float32]) -> ModelOutput:
            """Stub prediction."""
            boundary_logits, tone_logits = self.forward(mel)

            return ModelOutput(
                boundary_logits=boundary_logits,
                tone_logits=tone_logits,
                syllable_starts=None,
                tone_predictions=None,
            )

        def parameters(self):
            """Return empty parameters for compatibility."""
            return iter([])

        def state_dict(self):
            """Return empty state dict."""
            return {}

        def load_state_dict(self, state_dict):
            """Stub load."""
            pass


def extract_mel_spectrogram(
    audio: NDArray[np.float32],
    config: ModelConfig,
) -> NDArray[np.float32]:
    """Extract mel spectrogram from audio.

    Args:
        audio: Audio samples (float32, normalized to [-1, 1])
        config: Model configuration with mel parameters

    Returns:
        Mel spectrogram [n_mels, time]
    """
    # Simple mel spectrogram using numpy only
    # For production, would use librosa or torchaudio

    n_fft = config.win_length
    hop_length = config.hop_length
    n_mels = config.n_mels
    sr = config.sample_rate

    # Compute STFT
    n_frames = 1 + (len(audio) - n_fft) // hop_length
    if n_frames < 1:
        # Pad if too short
        audio = np.pad(audio, (0, n_fft - len(audio) + hop_length))
        n_frames = 1

    # Window function
    window = np.hanning(n_fft)

    # Compute spectrogram
    spec = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start:start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        windowed = frame * window
        fft = np.fft.rfft(windowed)
        spec[:, i] = np.abs(fft) ** 2

    # Mel filterbank
    mel_basis = _create_mel_filterbank(sr, n_fft, n_mels)
    mel_spec = np.dot(mel_basis, spec)

    # Log compression
    mel_spec = np.log(mel_spec + 1e-9)

    return mel_spec.astype(np.float32)


def _create_mel_filterbank(sr: int, n_fft: int, n_mels: int) -> NDArray[np.float32]:
    """Create mel filterbank matrix."""
    fmin = 0
    fmax = sr // 2

    # Mel scale conversion
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    # Create mel points
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # Convert to FFT bins
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    # Create filterbank
    n_freqs = n_fft // 2 + 1
    filterbank = np.zeros((n_mels, n_freqs), dtype=np.float32)

    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        # Rising edge
        for j in range(left, center):
            if center != left:
                filterbank[i, j] = (j - left) / (center - left)

        # Falling edge
        for j in range(center, right):
            if right != center:
                filterbank[i, j] = (right - j) / (right - center)

    return filterbank
