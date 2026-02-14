"""Autoregressive Syllable+Tone Predictor V4.

This model predicts the next syllable and its tone given:
1. Audio features (mel spectrogram, 1s chunk = ~100 frames)
2. Previous pinyin context (up to but not including the current syllable)

Architecture improvements over V3:
1. CNN front-end (4x downsampling) replacing linear mel projection
2. Attention pooling (PMA) replacing mean pooling
3. Rotary Position Embeddings (RoPE) replacing sinusoidal PE

Architecture:
    Input: concat([audio_embed, pinyin_embed]) -> Transformer -> dual heads
    - Audio: mel [bs, n_mels, time] -> CNN -> [bs, time//4, d_model]
    - Pinyin: token IDs -> embedding -> [bs, seq_len, d_model]
    - Combined: [bs, N_audio//4 + N_pinyin, d_model]
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
class SyllablePredictorConfigV4:
    """Configuration for the syllable predictor model V4."""

    # Audio input
    n_mels: int = 80
    sample_rate: int = 16000
    hop_length: int = 160  # 10ms at 16kHz
    win_length: int = 400  # 25ms at 16kHz
    max_audio_frames: int = 100  # ~1 second at 10ms per frame

    # CNN front-end
    cnn_kernel_size: int = 3  # Kernel size for CNN layers

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
        # Normalize: lowercase, handle u variants
        syllable = syllable.lower().replace("v", "u")
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

    class RotaryPositionalEmbedding(nn.Module):
        """Rotary Position Embedding (RoPE).

        RoPE encodes position information by rotating the query and key vectors
        in the attention mechanism. This allows the model to learn relative
        positions naturally through the dot product of rotated vectors.
        """

        def __init__(self, dim: int, max_len: int = 1024):
            super().__init__()
            inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer("inv_freq", inv_freq)
            self.max_len = max_len
            self._init_cache(max_len)

        def _init_cache(self, seq_len: int):
            """Initialize cos/sin cache for given sequence length."""
            t = torch.arange(seq_len, device=self.inv_freq.device)
            freqs = torch.einsum("i,j->ij", t.float(), self.inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            self.register_buffer("cos_cached", emb.cos()[None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, :, :], persistent=False)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Return cos and sin for the sequence length of x.

            Args:
                x: Input tensor [batch, seq_len, dim]

            Returns:
                Tuple of (cos, sin) each [1, seq_len, dim]
            """
            seq_len = x.shape[1]
            if seq_len > self.cos_cached.shape[1]:
                self._init_cache(seq_len)
            return self.cos_cached[:, :seq_len], self.sin_cached[:, :seq_len]


    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)


    def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embeddings to input tensor."""
        return (x * cos) + (rotate_half(x) * sin)


    class RoPETransformerEncoderLayer(nn.Module):
        """Transformer encoder layer with Rotary Position Embeddings.

        This is a custom encoder layer that applies RoPE to the query and key
        vectors before computing attention, replacing absolute positional encoding.
        """

        def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int,
            dropout: float = 0.1,
            norm_first: bool = True,
        ):
            super().__init__()
            self.d_model = d_model
            self.nhead = nhead
            self.head_dim = d_model // nhead
            self.norm_first = norm_first

            # Self-attention components
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)

            # Feedforward
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.linear2 = nn.Linear(dim_feedforward, d_model)

            # Norms and dropout
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)

            self.activation = nn.GELU()

        def forward(
            self,
            src: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            src_key_padding_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Forward pass with RoPE.

            Args:
                src: Input tensor [batch, seq_len, d_model]
                cos: Cosine component from RoPE [1, seq_len, d_model]
                sin: Sine component from RoPE [1, seq_len, d_model]
                src_key_padding_mask: Mask for padding [batch, seq_len], True = masked

            Returns:
                Output tensor [batch, seq_len, d_model]
            """
            if self.norm_first:
                src = src + self._sa_block(self.norm1(src), cos, sin, src_key_padding_mask)
                src = src + self._ff_block(self.norm2(src))
            else:
                src = self.norm1(src + self._sa_block(src, cos, sin, src_key_padding_mask))
                src = self.norm2(src + self._ff_block(src))
            return src

        def _sa_block(
            self,
            x: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            key_padding_mask: torch.Tensor | None,
        ) -> torch.Tensor:
            """Self-attention block with RoPE."""
            batch_size, seq_len, _ = x.shape

            # Project Q, K, V
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

            # Reshape for multi-head attention
            q = q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

            # Apply RoPE to Q and K
            # Reshape cos/sin for head dimension: [1, seq_len, d_model] -> [1, 1, seq_len, head_dim]
            cos_head = cos.view(1, seq_len, self.nhead, self.head_dim).transpose(1, 2)
            sin_head = sin.view(1, seq_len, self.nhead, self.head_dim).transpose(1, 2)

            q = apply_rotary_pos_emb(q, cos_head, sin_head)
            k = apply_rotary_pos_emb(k, cos_head, sin_head)

            # Scaled dot-product attention
            scale = math.sqrt(self.head_dim)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

            # Apply padding mask
            if key_padding_mask is not None:
                # key_padding_mask: [batch, seq_len] -> [batch, 1, 1, seq_len]
                attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
                attn_weights = attn_weights.masked_fill(attn_mask, float("-inf"))

            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)

            attn_output = torch.matmul(attn_weights, v)

            # Reshape back
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            return self.dropout1(self.out_proj(attn_output))

        def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
            """Feedforward block."""
            return self.dropout2(self.linear2(self.activation(self.linear1(x))))


    class SyllablePredictorV4(nn.Module):
        """Autoregressive syllable+tone predictor V4.

        Takes mel spectrogram + pinyin context, predicts next syllable and tone.

        Architecture improvements over V3:
        1. CNN front-end: mel frames -> 4x downsampling -> d_model
        2. Attention pooling (PMA): learnable query -> cross-attention -> pooled
        3. RoPE: Rotary positional embeddings in transformer

        Architecture:
        1. Audio CNN: mel frames -> d_model (4x sequence reduction)
        2. Pinyin embedding: token IDs -> d_model
        3. Concatenate audio + pinyin embeddings
        4. RoPE Transformer encoder
        5. Attention pooling (PMA)
        6. Dual output heads (syllable + tone)
        """

        def __init__(self, config: SyllablePredictorConfigV4 | None = None):
            super().__init__()

            if config is None:
                config = SyllablePredictorConfigV4()
            self.config = config

            # Load vocabulary
            self.vocab = SyllableVocab()
            vocab_size = len(self.vocab)

            # CNN front-end: [n_mels, time] -> [d_model, time//4]
            # Two conv layers with stride=2 each for 4x downsampling
            self.audio_cnn = nn.Sequential(
                # Layer 1: n_mels -> d_model//2, stride=2
                nn.Conv1d(
                    config.n_mels,
                    config.d_model // 2,
                    kernel_size=config.cnn_kernel_size,
                    stride=2,
                    padding=config.cnn_kernel_size // 2,
                ),
                nn.BatchNorm1d(config.d_model // 2),
                nn.GELU(),
                # Layer 2: d_model//2 -> d_model, stride=2
                nn.Conv1d(
                    config.d_model // 2,
                    config.d_model,
                    kernel_size=config.cnn_kernel_size,
                    stride=2,
                    padding=config.cnn_kernel_size // 2,
                ),
                nn.BatchNorm1d(config.d_model),
                nn.GELU(),
            )

            # Pinyin embedding: [vocab_size] -> [d_model]
            self.pinyin_embed = nn.Embedding(vocab_size, config.d_model, padding_idx=0)

            # Modality embeddings (to distinguish audio vs pinyin)
            self.audio_type_embed = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
            self.pinyin_type_embed = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)

            # RoPE for positional encoding
            max_seq_len = config.max_audio_frames // 4 + config.max_pinyin_len + 1
            self.rope = RotaryPositionalEmbedding(dim=config.d_model, max_len=max_seq_len)

            # Transformer encoder with RoPE
            self.transformer_layers = nn.ModuleList([
                RoPETransformerEncoderLayer(
                    d_model=config.d_model,
                    nhead=config.n_heads,
                    dim_feedforward=config.dim_feedforward,
                    dropout=config.dropout,
                    norm_first=True,
                )
                for _ in range(config.n_layers)
            ])

            # Attention pooling (PMA - Pooling by Multihead Attention)
            self.pool_query = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
            self.pool_attention = nn.MultiheadAttention(
                embed_dim=config.d_model,
                num_heads=config.n_heads,
                dropout=config.dropout,
                batch_first=True,
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
                elif isinstance(module, nn.Conv1d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

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
            original_audio_len = mel.shape[2]  # [batch, n_mels, time]

            # CNN front-end: [batch, n_mels, time] -> [batch, d_model, time//4]
            audio_embed = self.audio_cnn(mel)  # [batch, d_model, time//4]

            # Transpose to [batch, time//4, d_model]
            audio_embed = audio_embed.transpose(1, 2)
            downsampled_audio_len = audio_embed.shape[1]

            # Calculate max downsampled frames
            max_downsampled_frames = self.config.max_audio_frames // 4

            # Pad/truncate audio to max_downsampled_frames
            actual_downsampled_len = downsampled_audio_len
            if downsampled_audio_len < max_downsampled_frames:
                pad_len = max_downsampled_frames - downsampled_audio_len
                audio_embed = F.pad(audio_embed, (0, 0, 0, pad_len))
            elif downsampled_audio_len > max_downsampled_frames:
                audio_embed = audio_embed[:, :max_downsampled_frames, :]
                actual_downsampled_len = max_downsampled_frames

            # Create or adjust audio mask for downsampled sequence
            if audio_mask is not None:
                audio_mask = audio_mask.to(device)
                # Downsample the mask: ceil(original_len / 4)
                # For each group of 4 frames, if any is valid, the downsampled frame is valid
                orig_mask_len = audio_mask.shape[1]
                # Compute downsampled length
                ds_len = (orig_mask_len + 3) // 4
                # Create downsampled mask
                downsampled_mask = torch.ones(batch_size, ds_len, dtype=torch.bool, device=device)
                for i in range(ds_len):
                    start_idx = i * 4
                    end_idx = min(start_idx + 4, orig_mask_len)
                    # If any frame in the group is valid (False), the downsampled frame is valid
                    downsampled_mask[:, i] = audio_mask[:, start_idx:end_idx].all(dim=1)
                audio_mask = downsampled_mask

                # Adjust to max_downsampled_frames
                if audio_mask.shape[1] < max_downsampled_frames:
                    pad_mask = torch.ones(batch_size, max_downsampled_frames - audio_mask.shape[1],
                                          dtype=torch.bool, device=device)
                    audio_mask = torch.cat([audio_mask, pad_mask], dim=1)
                elif audio_mask.shape[1] > max_downsampled_frames:
                    audio_mask = audio_mask[:, :max_downsampled_frames]
            else:
                # Create mask: True for padded positions
                # Calculate downsampled length from original
                ds_len = (original_audio_len + 3) // 4
                ds_len = min(ds_len, max_downsampled_frames)
                audio_mask = torch.zeros(batch_size, max_downsampled_frames, dtype=torch.bool, device=device)
                audio_mask[:, ds_len:] = True

            if pinyin_mask is not None:
                pinyin_mask = pinyin_mask.to(device)

            # Add modality embedding to audio
            audio_embed = audio_embed + self.audio_type_embed

            # Pinyin embedding: lookup + type embedding
            pinyin_embed = self.pinyin_embed(pinyin_ids)  # [batch, seq_len, d_model]
            pinyin_embed = pinyin_embed + self.pinyin_type_embed

            # Concatenate: [batch, audio_len + pinyin_len, d_model]
            combined = torch.cat([audio_embed, pinyin_embed], dim=1)

            # Create combined attention mask
            pinyin_mask_full = pinyin_mask if pinyin_mask is not None else torch.zeros(
                batch_size, pinyin_ids.shape[1], dtype=torch.bool, device=device
            )
            combined_mask = torch.cat([audio_mask, pinyin_mask_full], dim=1)

            # Get RoPE cos/sin for the combined sequence
            cos, sin = self.rope(combined)

            # Transformer with RoPE
            encoded = combined
            for layer in self.transformer_layers:
                encoded = layer(encoded, cos, sin, src_key_padding_mask=combined_mask)

            # Attention pooling (PMA)
            query = self.pool_query.expand(batch_size, -1, -1)  # [batch, 1, d_model]
            pooled, _ = self.pool_attention(
                query=query,
                key=encoded,
                value=encoded,
                key_padding_mask=combined_mask,
            )
            pooled = pooled.squeeze(1)  # [batch, d_model]

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
    class SyllablePredictorV4:
        def __init__(self, config: SyllablePredictorConfigV4 | None = None):
            self.config = config or SyllablePredictorConfigV4()
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
    config: SyllablePredictorConfigV4,
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
