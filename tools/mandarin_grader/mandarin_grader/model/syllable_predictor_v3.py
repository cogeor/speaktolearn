"""Autoregressive Syllable+Tone Predictor V3.

This model predicts the next syllable and its tone given:
1. Audio features (mel spectrogram, 1s chunk = ~100 frames)
2. Previous pinyin context (up to but not including the current syllable)

Architecture:
    Input: concat([audio_embed, pinyin_embed]) -> Transformer -> dual heads
    - Audio: mel [bs, n_mels, time] -> linear proj -> [bs, time, d_model]
    - Pinyin: token IDs -> embedding -> [bs, seq_len, d_model]
    - Combined: [bs, N_audio + N_pinyin, d_model]
    - Output: syllable head (530 classes) + tone head (5 classes)

Model size target: <5M params
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray


# Syllable vocabulary - loaded from metadata or hardcoded
SYLLABLE_VOCAB_PATH = Path(__file__).parent.parent.parent / "data" / "syllables_v2" / "metadata.json"


def load_syllable_vocab() -> list[str]:
    """Load syllable vocabulary from metadata.json."""
    if SYLLABLE_VOCAB_PATH.exists():
        with open(SYLLABLE_VOCAB_PATH, encoding="utf-8") as f:
            data = json.load(f)
            return data.get("syllables", [])
    # Fallback to common syllables if file not found
    return []


@dataclass
class SyllablePredictorConfig:
    """Configuration for the syllable predictor model."""

    # Audio input
    n_mels: int = 80
    sample_rate: int = 16000
    hop_length: int = 160  # 10ms at 16kHz
    win_length: int = 400  # 25ms at 16kHz
    max_audio_frames: int = 100  # ~1 second at 10ms per frame

    # Transformer architecture
    d_model: int = 192  # Model dimension (kept small for <5M params)
    n_heads: int = 6  # Attention heads
    n_layers: int = 4  # Transformer encoder layers
    dim_feedforward: int = 384  # FFN hidden dim
    dropout: float = 0.1

    # Vocabulary sizes
    n_syllables: int = 530  # Number of base syllables (loaded dynamically)
    n_tones: int = 5  # Tones 0-4
    max_pinyin_len: int = 50  # Max syllables in context (some sentences have 20+ syllables)

    # Special tokens
    pad_token: int = 0
    bos_token: int = 1  # Beginning of sequence

    def __post_init__(self):
        # Try to load actual vocab size (including special tokens: PAD, BOS)
        vocab = load_syllable_vocab()
        if vocab:
            # Total vocab = syllables + 2 special tokens (PAD=0, BOS=1)
            self.n_syllables = len(vocab) + 2


@dataclass
class PredictorOutput:
    """Output from syllable predictor."""
    syllable_logits: NDArray[np.float32]  # [batch, n_syllables]
    tone_logits: NDArray[np.float32]  # [batch, n_tones]
    syllable_pred: int | None = None
    tone_pred: int | None = None


class SyllableVocab:
    """Syllable vocabulary with encoding/decoding."""

    def __init__(self, syllables: list[str] | None = None):
        if syllables is None:
            syllables = load_syllable_vocab()

        # Special tokens: 0=PAD, 1=BOS, 2+=syllables
        self.pad_token = 0
        self.bos_token = 1
        self.special_tokens = 2

        self.syllables = syllables
        self.syl_to_idx = {s: i + self.special_tokens for i, s in enumerate(syllables)}
        self.idx_to_syl = {i + self.special_tokens: s for i, s in enumerate(syllables)}

    def __len__(self) -> int:
        return len(self.syllables) + self.special_tokens

    def encode(self, syllable: str) -> int:
        """Encode a syllable to token ID."""
        # Normalize: lowercase, handle ü variants
        syllable = syllable.lower().replace("v", "ü")
        return self.syl_to_idx.get(syllable, self.pad_token)

    def decode(self, idx: int) -> str:
        """Decode token ID to syllable."""
        if idx == self.pad_token:
            return "<PAD>"
        if idx == self.bos_token:
            return "<BOS>"
        return self.idx_to_syl.get(idx, "<UNK>")

    def encode_sequence(self, syllables: list[str], add_bos: bool = True) -> list[int]:
        """Encode a sequence of syllables."""
        tokens = []
        if add_bos:
            tokens.append(self.bos_token)
        for syl in syllables:
            tokens.append(self.encode(syl))
        return tokens


# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding."""

        def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # [1, max_len, d_model]

            self.register_buffer("pe", pe)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Add positional encoding to input [batch, seq_len, d_model]."""
            x = x + self.pe[:, :x.size(1)]
            return self.dropout(x)


    class SyllablePredictorV3(nn.Module):
        """Autoregressive syllable+tone predictor.

        Takes mel spectrogram + pinyin context, predicts next syllable and tone.

        Architecture:
        1. Audio projection: mel frames -> d_model
        2. Pinyin embedding: token IDs -> d_model
        3. Concatenate audio + pinyin embeddings
        4. Transformer encoder
        5. Dual output heads (syllable + tone)
        """

        def __init__(self, config: SyllablePredictorConfig | None = None):
            super().__init__()

            if config is None:
                config = SyllablePredictorConfig()
            self.config = config

            # Load vocabulary
            self.vocab = SyllableVocab()
            vocab_size = len(self.vocab)

            # Audio input projection: [n_mels] -> [d_model]
            self.audio_proj = nn.Linear(config.n_mels, config.d_model)

            # Pinyin embedding: [vocab_size] -> [d_model]
            self.pinyin_embed = nn.Embedding(vocab_size, config.d_model, padding_idx=0)

            # Separate positional encodings for audio and pinyin
            self.audio_pos = PositionalEncoding(
                config.d_model,
                max_len=config.max_audio_frames,
                dropout=config.dropout
            )
            self.pinyin_pos = PositionalEncoding(
                config.d_model,
                max_len=config.max_pinyin_len + 1,  # +1 for BOS
                dropout=config.dropout
            )

            # Modality embeddings (to distinguish audio vs pinyin)
            self.audio_type_embed = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
            self.pinyin_type_embed = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)

            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                batch_first=True,
                norm_first=True,  # Pre-norm for stability
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=config.n_layers,
            )

            # Output projection (from transformer output to prediction)
            self.output_norm = nn.LayerNorm(config.d_model)

            # Dual heads
            self.syllable_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model, config.n_syllables),
            )

            self.tone_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model // 2, config.n_tones),
            )

            # Initialize weights
            self._init_weights()

        def _init_weights(self):
            """Initialize model weights."""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.padding_idx is not None:
                        module.weight.data[module.padding_idx].zero_()

        def forward(
            self,
            mel: torch.Tensor | np.ndarray,
            pinyin_ids: torch.Tensor | np.ndarray,
            audio_mask: torch.Tensor | None = None,
            pinyin_mask: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Forward pass.

            Args:
                mel: Mel spectrogram [batch, n_mels, time]
                pinyin_ids: Pinyin token IDs [batch, seq_len]
                audio_mask: Mask for padded audio frames [batch, time]
                pinyin_mask: Mask for padded pinyin tokens [batch, seq_len]

            Returns:
                Tuple of (syllable_logits, tone_logits)
                - syllable_logits: [batch, n_syllables]
                - tone_logits: [batch, n_tones]
            """
            if isinstance(mel, np.ndarray):
                mel = torch.from_numpy(mel).float()
            if isinstance(pinyin_ids, np.ndarray):
                pinyin_ids = torch.from_numpy(pinyin_ids).long()

            device = next(self.parameters()).device
            mel = mel.to(device)
            pinyin_ids = pinyin_ids.to(device)

            batch_size = mel.shape[0]

            # Transpose mel: [batch, n_mels, time] -> [batch, time, n_mels]
            mel = mel.transpose(1, 2)
            audio_len = mel.shape[1]

            # Pad/truncate audio to max_audio_frames
            original_audio_len = audio_len
            if audio_len < self.config.max_audio_frames:
                pad_len = self.config.max_audio_frames - audio_len
                mel = F.pad(mel, (0, 0, 0, pad_len))
            elif audio_len > self.config.max_audio_frames:
                mel = mel[:, :self.config.max_audio_frames, :]
                original_audio_len = self.config.max_audio_frames

            # Create or adjust audio mask to match max_audio_frames
            if audio_mask is not None:
                audio_mask = audio_mask.to(device)
                # Truncate/pad mask to match max_audio_frames
                mask_len = audio_mask.shape[1]
                if mask_len < self.config.max_audio_frames:
                    # Pad mask with True (masked positions)
                    pad_mask = torch.ones(batch_size, self.config.max_audio_frames - mask_len, dtype=torch.bool, device=device)
                    audio_mask = torch.cat([audio_mask, pad_mask], dim=1)
                elif mask_len > self.config.max_audio_frames:
                    audio_mask = audio_mask[:, :self.config.max_audio_frames]
            else:
                # Create mask: True for padded positions
                audio_mask = torch.zeros(batch_size, self.config.max_audio_frames, dtype=torch.bool, device=device)
                audio_mask[:, original_audio_len:] = True

            if pinyin_mask is not None:
                pinyin_mask = pinyin_mask.to(device)

            # Audio embedding: project mel to d_model + positional encoding + type embedding
            audio_embed = self.audio_proj(mel)  # [batch, time, d_model]
            audio_embed = self.audio_pos(audio_embed)
            audio_embed = audio_embed + self.audio_type_embed

            # Pinyin embedding: lookup + positional encoding + type embedding
            pinyin_embed = self.pinyin_embed(pinyin_ids)  # [batch, seq_len, d_model]
            pinyin_embed = self.pinyin_pos(pinyin_embed)
            pinyin_embed = pinyin_embed + self.pinyin_type_embed

            # Concatenate: [batch, audio_len + pinyin_len, d_model]
            combined = torch.cat([audio_embed, pinyin_embed], dim=1)

            # Create combined attention mask
            # audio_mask is always created above, pinyin_mask may be None
            pinyin_mask_full = pinyin_mask if pinyin_mask is not None else torch.zeros(
                batch_size, pinyin_ids.shape[1], dtype=torch.bool, device=device
            )
            combined_mask = torch.cat([audio_mask, pinyin_mask_full], dim=1)

            # Transformer
            encoded = self.transformer(combined, src_key_padding_mask=combined_mask)

            # Pool: use mean over non-padded positions
            # Invert mask (True = padded/masked, False = valid)
            keep_mask = ~combined_mask
            # Expand for broadcasting: [batch, seq_len, 1]
            keep_mask_expanded = keep_mask.unsqueeze(-1).float()
            # Masked mean
            pooled = (encoded * keep_mask_expanded).sum(dim=1) / keep_mask_expanded.sum(dim=1).clamp(min=1)

            pooled = self.output_norm(pooled)

            # Dual heads
            syllable_logits = self.syllable_head(pooled)
            tone_logits = self.tone_head(pooled)

            return syllable_logits, tone_logits

        def predict(
            self,
            mel: torch.Tensor | np.ndarray,
            pinyin_ids: torch.Tensor | np.ndarray,
        ) -> PredictorOutput:
            """Make predictions.

            Args:
                mel: Mel spectrogram [batch, n_mels, time] or [n_mels, time]
                pinyin_ids: Pinyin token IDs [batch, seq_len] or [seq_len]

            Returns:
                PredictorOutput with logits and predictions
            """
            # Add batch dimension if needed
            if isinstance(mel, np.ndarray):
                mel = torch.from_numpy(mel).float()
            if isinstance(pinyin_ids, np.ndarray):
                pinyin_ids = torch.from_numpy(pinyin_ids).long()

            if mel.dim() == 2:
                mel = mel.unsqueeze(0)
            if pinyin_ids.dim() == 1:
                pinyin_ids = pinyin_ids.unsqueeze(0)

            with torch.no_grad():
                syllable_logits, tone_logits = self.forward(mel, pinyin_ids)

            syllable_pred = syllable_logits[0].argmax().item()
            tone_pred = tone_logits[0].argmax().item()

            return PredictorOutput(
                syllable_logits=syllable_logits.cpu().numpy(),
                tone_logits=tone_logits.cpu().numpy(),
                syllable_pred=syllable_pred,
                tone_pred=tone_pred,
            )

        def count_parameters(self) -> tuple[int, int]:
            """Count model parameters.

            Returns:
                Tuple of (total_params, trainable_params)
            """
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            return total, trainable


else:
    # Stub when PyTorch not available
    class SyllablePredictorV3:
        def __init__(self, config: SyllablePredictorConfig | None = None):
            self.config = config or SyllablePredictorConfig()
            self.vocab = SyllableVocab()

        def forward(self, mel, pinyin_ids, audio_mask=None, pinyin_mask=None):
            batch = mel.shape[0]
            return (
                np.random.randn(batch, self.config.n_syllables),
                np.random.randn(batch, self.config.n_tones),
            )

        def predict(self, mel, pinyin_ids):
            syl_logits, tone_logits = self.forward(
                mel if mel.ndim == 3 else mel[np.newaxis],
                pinyin_ids if pinyin_ids.ndim == 2 else pinyin_ids[np.newaxis],
            )
            return PredictorOutput(
                syllable_logits=syl_logits,
                tone_logits=tone_logits,
                syllable_pred=0,
                tone_pred=0,
            )

        def parameters(self):
            return iter([])

        def state_dict(self):
            return {}

        def load_state_dict(self, d):
            pass

        def count_parameters(self):
            return 0, 0


def extract_mel_spectrogram(
    audio: NDArray[np.float32],
    config: SyllablePredictorConfig,
) -> NDArray[np.float32]:
    """Extract mel spectrogram from audio.

    Args:
        audio: Audio samples (float32, normalized to [-1, 1])
        config: Config with mel parameters

    Returns:
        Mel spectrogram [n_mels, time]
    """
    # Normalize audio amplitude to [-1, 1] for consistent mel features
    max_abs = np.max(np.abs(audio))
    if max_abs > 1e-6:
        audio = audio / max_abs

    n_fft = config.win_length
    hop_length = config.hop_length
    n_mels = config.n_mels
    sr = config.sample_rate

    # Compute STFT
    n_frames = 1 + (len(audio) - n_fft) // hop_length
    if n_frames < 1:
        audio = np.pad(audio, (0, n_fft - len(audio) + hop_length))
        n_frames = 1

    window = np.hanning(n_fft)

    spec = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start:start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        windowed = frame * window
        fft = np.fft.rfft(windowed)
        spec[:, i] = np.abs(fft) ** 2

    mel_basis = _create_mel_filterbank(sr, n_fft, n_mels)
    mel_spec = np.dot(mel_basis, spec)

    mel_spec = np.log(mel_spec + 1e-9)

    return mel_spec.astype(np.float32)


def _create_mel_filterbank(sr: int, n_fft: int, n_mels: int) -> NDArray[np.float32]:
    """Create mel filterbank matrix."""
    fmin = 0
    fmax = sr // 2

    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    n_freqs = n_fft // 2 + 1
    filterbank = np.zeros((n_mels, n_freqs), dtype=np.float32)

    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        for j in range(left, center):
            if center != left:
                filterbank[i, j] = (j - left) / (center - left)

        for j in range(center, right):
            if right != center:
                filterbank[i, j] = (right - j) / (right - center)

    return filterbank
