"""Full-Sentence Syllable+Tone Predictor V5.

This model predicts a syllable and its tone given:
1. Full sentence audio (mel spectrogram, up to 10s = ~1000 frames)
2. Position index indicating which syllable to predict

Key differences from V4:
- Full sentence audio instead of 1s chunks
- Single position token instead of pinyin sequence
- Non-autoregressive: position selects which syllable to predict
- Longformer-style local+global attention for efficient processing

Architecture:
    Input: mel [bs, n_mels, time] + position [bs, 1]
    - Audio: mel -> CNN (8x downsampling) -> [bs, time//8, d_model]
    - Position: embedding -> [bs, 1, d_model] -> broadcast to audio
    - Combined: audio_embed + position_embed
    - Transformer with RoPE and Longformer attention (local window + global)
    - Attention pooling (PMA)
    - Output: syllable head (532 classes) + tone head (5 classes)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray


# Syllable vocabulary - loaded from metadata
SYLLABLE_VOCAB_PATH = Path(__file__).parent.parent.parent / "data" / "syllables_v2" / "metadata.json"


def load_syllable_vocab() -> list[str]:
    """Load syllable vocabulary from metadata.json."""
    if SYLLABLE_VOCAB_PATH.exists():
        with open(SYLLABLE_VOCAB_PATH, encoding="utf-8") as f:
            data = json.load(f)
            return data.get("syllables", [])
    return []


@dataclass
class SyllablePredictorConfigV5:
    """Configuration for the syllable predictor model V5."""

    # Audio input - 10s max at 16kHz with hop=160 gives ~1000 frames
    n_mels: int = 80
    sample_rate: int = 16000
    hop_length: int = 160  # 10ms at 16kHz
    win_length: int = 400  # 25ms at 16kHz
    max_audio_frames: int = 1000  # ~10 seconds at 10ms per frame

    # CNN front-end
    cnn_kernel_size: int = 3

    # Transformer architecture
    d_model: int = 192
    n_heads: int = 6
    n_layers: int = 4
    dim_feedforward: int = 384
    dropout: float = 0.1

    # Longformer-style attention
    attention_window: int = 32  # Local attention window size (each side)
    use_global_attention: bool = True  # Position token has global attention

    # Vocabulary sizes
    n_syllables: int = 530  # Number of base syllables
    n_tones: int = 5  # Tones 0-4
    max_positions: int = 60  # Max syllables per sentence (10s at ~5 syl/s)

    # Special tokens
    pad_token: int = 0

    def __post_init__(self):
        vocab = load_syllable_vocab()
        if vocab:
            # Total vocab = syllables + 2 special tokens (PAD=0, BOS=1)
            self.n_syllables = len(vocab) + 2


@dataclass
class PredictorOutput:
    """Output from syllable predictor."""
    syllable_logits: NDArray[np.float32]
    tone_logits: NDArray[np.float32]
    syllable_pred: int | None = None
    tone_pred: int | None = None
    syllable_prob: float | None = None
    tone_prob: float | None = None


# Reuse SyllableVocab from V4
from .syllable_predictor_v4 import SyllableVocab


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
        """Rotary Position Embedding (RoPE)."""

        def __init__(self, dim: int, max_len: int = 2048):
            super().__init__()
            inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer("inv_freq", inv_freq)
            self.max_len = max_len
            self._init_cache(max_len)

        def _init_cache(self, seq_len: int):
            t = torch.arange(seq_len, device=self.inv_freq.device)
            freqs = torch.einsum("i,j->ij", t.float(), self.inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            self.register_buffer("cos_cached", emb.cos()[None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, :, :], persistent=False)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            seq_len = x.shape[1]
            if seq_len > self.cos_cached.shape[1]:
                self._init_cache(seq_len)
            return self.cos_cached[:, :seq_len], self.sin_cached[:, :seq_len]


    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)


    def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return (x * cos) + (rotate_half(x) * sin)


    def create_longformer_attention_mask(
        seq_len: int,
        window_size: int,
        global_indices: list[int] | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Create Longformer-style attention mask.

        Args:
            seq_len: Sequence length
            window_size: Size of sliding window (total window = 2*window_size + 1)
            global_indices: Positions with global attention (can attend everywhere)
            device: Device to create tensor on

        Returns:
            Attention mask of shape [seq_len, seq_len] where True = masked (cannot attend)
        """
        # Start with all positions masked
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)

        # Create local attention windows
        for i in range(seq_len):
            # Each position can attend to window_size positions on each side
            start = max(0, i - window_size)
            end = min(seq_len, i + window_size + 1)
            mask[i, start:end] = False

        # Global attention: specified positions can attend to all and be attended by all
        if global_indices:
            for idx in global_indices:
                if 0 <= idx < seq_len:
                    mask[idx, :] = False  # Global token attends to all
                    mask[:, idx] = False  # All tokens attend to global token

        return mask


    class RoPETransformerEncoderLayer(nn.Module):
        """Transformer encoder layer with Rotary Position Embeddings and Longformer attention."""

        def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int,
            dropout: float = 0.1,
            norm_first: bool = True,
            attention_window: int | None = None,
        ):
            super().__init__()
            self.d_model = d_model
            self.nhead = nhead
            self.head_dim = d_model // nhead
            self.norm_first = norm_first
            self.attention_window = attention_window

            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)

            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.linear2 = nn.Linear(dim_feedforward, d_model)

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
            attn_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if self.norm_first:
                src = src + self._sa_block(self.norm1(src), cos, sin, src_key_padding_mask, attn_mask)
                src = src + self._ff_block(self.norm2(src))
            else:
                src = self.norm1(src + self._sa_block(src, cos, sin, src_key_padding_mask, attn_mask))
                src = self.norm2(src + self._ff_block(src))
            return src

        def _sa_block(
            self,
            x: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            key_padding_mask: torch.Tensor | None,
            attn_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            batch_size, seq_len, _ = x.shape

            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

            q = q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

            cos_head = cos.view(1, seq_len, self.nhead, self.head_dim).transpose(1, 2)
            sin_head = sin.view(1, seq_len, self.nhead, self.head_dim).transpose(1, 2)

            q = apply_rotary_pos_emb(q, cos_head, sin_head)
            k = apply_rotary_pos_emb(k, cos_head, sin_head)

            scale = math.sqrt(self.head_dim)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

            # Apply Longformer attention mask (local + global)
            if attn_mask is not None:
                # attn_mask shape: [seq_len, seq_len], True = masked
                # Expand for batch and heads: [1, 1, seq_len, seq_len]
                attn_mask_expanded = attn_mask.unsqueeze(0).unsqueeze(0)
                attn_weights = attn_weights.masked_fill(attn_mask_expanded, float("-inf"))

            # Apply padding mask
            if key_padding_mask is not None:
                attn_mask_pad = key_padding_mask.unsqueeze(1).unsqueeze(2)
                attn_weights = attn_weights.masked_fill(attn_mask_pad, float("-inf"))

            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)

            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            return self.dropout1(self.out_proj(attn_output))

        def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
            return self.dropout2(self.linear2(self.activation(self.linear1(x))))


    class SyllablePredictorV5(nn.Module):
        """Full-sentence syllable+tone predictor V5.

        Takes full sentence mel spectrogram + position index,
        predicts syllable and tone for that position.

        Architecture:
        1. Audio CNN: mel frames -> d_model (8x sequence reduction)
        2. Position embedding: position_idx -> d_model (broadcast to audio)
        3. Add position embedding to audio frames
        4. RoPE Transformer encoder with Longformer-style attention:
           - Local sliding window attention (default window=32)
           - Global attention on position 0 for aggregating sentence context
        5. Attention pooling (PMA)
        6. Dual output heads (syllable + tone)
        """

        def __init__(self, config: SyllablePredictorConfigV5 | None = None):
            super().__init__()

            if config is None:
                config = SyllablePredictorConfigV5()
            self.config = config

            self.vocab = SyllableVocab()
            vocab_size = len(self.vocab)

            # CNN front-end: [n_mels, time] -> [d_model, time//8]
            self.audio_cnn = nn.Sequential(
                nn.Conv1d(
                    config.n_mels,
                    config.d_model // 2,
                    kernel_size=config.cnn_kernel_size,
                    stride=2,
                    padding=config.cnn_kernel_size // 2,
                ),
                nn.BatchNorm1d(config.d_model // 2),
                nn.GELU(),
                nn.Conv1d(
                    config.d_model // 2,
                    config.d_model,
                    kernel_size=config.cnn_kernel_size,
                    stride=2,
                    padding=config.cnn_kernel_size // 2,
                ),
                nn.BatchNorm1d(config.d_model),
                nn.GELU(),
                nn.Conv1d(
                    config.d_model,
                    config.d_model,
                    kernel_size=config.cnn_kernel_size,
                    stride=2,
                    padding=config.cnn_kernel_size // 2,
                ),
                nn.BatchNorm1d(config.d_model),
                nn.GELU(),
            )

            # Position embedding: position_idx -> d_model
            # This tells the model which syllable position to predict
            self.position_embed = nn.Embedding(config.max_positions, config.d_model)

            # Audio type embedding (to distinguish audio features)
            self.audio_type_embed = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)

            # RoPE for transformer positional encoding
            max_seq_len = config.max_audio_frames // 8 + 1  # +1 for safety
            self.rope = RotaryPositionalEmbedding(dim=config.d_model, max_len=max_seq_len)

            # Transformer encoder with RoPE and Longformer attention
            self.transformer_layers = nn.ModuleList([
                RoPETransformerEncoderLayer(
                    d_model=config.d_model,
                    nhead=config.n_heads,
                    dim_feedforward=config.dim_feedforward,
                    dropout=config.dropout,
                    norm_first=True,
                    attention_window=config.attention_window,
                )
                for _ in range(config.n_layers)
            ])

            # Attention pooling (PMA)
            self.pool_query = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
            self.pool_attention = nn.MultiheadAttention(
                embed_dim=config.d_model,
                num_heads=config.n_heads,
                dropout=config.dropout,
                batch_first=True,
            )

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

            self._init_weights()

        def _init_weights(self):
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, std=0.02)
                elif isinstance(module, nn.Conv1d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def forward(
            self,
            mel: torch.Tensor | np.ndarray,
            position: torch.Tensor | np.ndarray,
            audio_mask: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Forward pass.

            Args:
                mel: Mel spectrogram [batch, n_mels, time]
                position: Position index [batch, 1] or [batch] - which syllable to predict
                audio_mask: Mask for padded audio frames [batch, time], True = masked

            Returns:
                Tuple of (syllable_logits, tone_logits)
                - syllable_logits: [batch, n_syllables]
                - tone_logits: [batch, n_tones]
            """
            if isinstance(mel, np.ndarray):
                mel = torch.from_numpy(mel).float()
            if isinstance(position, np.ndarray):
                position = torch.from_numpy(position).long()

            device = next(self.parameters()).device
            mel = mel.to(device)
            position = position.to(device)

            # Ensure position is [batch, 1]
            if position.dim() == 1:
                position = position.unsqueeze(1)

            batch_size = mel.shape[0]
            original_audio_len = mel.shape[2]

            # CNN front-end: [batch, n_mels, time] -> [batch, d_model, time//8]
            audio_embed = self.audio_cnn(mel)
            audio_embed = audio_embed.transpose(1, 2)  # [batch, time//8, d_model]
            downsampled_audio_len = audio_embed.shape[1]

            max_downsampled_frames = self.config.max_audio_frames // 8

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
                orig_mask_len = audio_mask.shape[1]
                ds_len = (orig_mask_len + 7) // 8
                downsampled_mask = torch.ones(batch_size, ds_len, dtype=torch.bool, device=device)
                for i in range(ds_len):
                    start_idx = i * 8
                    end_idx = min(start_idx + 8, orig_mask_len)
                    downsampled_mask[:, i] = audio_mask[:, start_idx:end_idx].all(dim=1)
                audio_mask = downsampled_mask

                if audio_mask.shape[1] < max_downsampled_frames:
                    pad_mask = torch.ones(batch_size, max_downsampled_frames - audio_mask.shape[1],
                                          dtype=torch.bool, device=device)
                    audio_mask = torch.cat([audio_mask, pad_mask], dim=1)
                elif audio_mask.shape[1] > max_downsampled_frames:
                    audio_mask = audio_mask[:, :max_downsampled_frames]
            else:
                ds_len = (original_audio_len + 7) // 8
                ds_len = min(ds_len, max_downsampled_frames)
                audio_mask = torch.zeros(batch_size, max_downsampled_frames, dtype=torch.bool, device=device)
                audio_mask[:, ds_len:] = True

            # Add audio type embedding
            audio_embed = audio_embed + self.audio_type_embed

            # Position embedding: [batch, 1] -> [batch, 1, d_model]
            pos_embed = self.position_embed(position)  # [batch, 1, d_model]

            # Broadcast position embedding to all audio frames
            # This tells every transformer layer which syllable we're looking for
            audio_embed = audio_embed + pos_embed  # Broadcasting [batch, T, d_model] + [batch, 1, d_model]

            # Get RoPE cos/sin
            cos, sin = self.rope(audio_embed)

            # Create Longformer attention mask for local + global attention
            # Position 0 is treated as global (can attend to all positions)
            # This is compatible with the position embedding which broadcasts to all frames
            seq_len = audio_embed.shape[1]
            if self.config.use_global_attention:
                global_indices = [0]  # First position has global attention
            else:
                global_indices = None

            attn_mask = create_longformer_attention_mask(
                seq_len=seq_len,
                window_size=self.config.attention_window,
                global_indices=global_indices,
                device=audio_embed.device,
            )

            # Transformer with RoPE and Longformer attention
            encoded = audio_embed
            for layer in self.transformer_layers:
                encoded = layer(encoded, cos, sin, src_key_padding_mask=audio_mask, attn_mask=attn_mask)

            # Attention pooling (PMA)
            query = self.pool_query.expand(batch_size, -1, -1)
            pooled, _ = self.pool_attention(
                query=query,
                key=encoded,
                value=encoded,
                key_padding_mask=audio_mask,
            )
            pooled = pooled.squeeze(1)

            pooled = self.output_norm(pooled)

            # Dual heads
            syllable_logits = self.syllable_head(pooled)
            tone_logits = self.tone_head(pooled)

            return syllable_logits, tone_logits

        def predict(
            self,
            mel: torch.Tensor | np.ndarray,
            position: torch.Tensor | np.ndarray,
        ) -> PredictorOutput:
            """Make predictions.

            Args:
                mel: Mel spectrogram [batch, n_mels, time] or [n_mels, time]
                position: Position index [batch, 1] or scalar

            Returns:
                PredictorOutput with logits and predictions
            """
            if isinstance(mel, np.ndarray):
                mel = torch.from_numpy(mel).float()
            if isinstance(position, np.ndarray):
                position = torch.from_numpy(position).long()

            if mel.dim() == 2:
                mel = mel.unsqueeze(0)
            if position.dim() == 0:
                position = position.unsqueeze(0).unsqueeze(0)
            elif position.dim() == 1:
                position = position.unsqueeze(1)

            with torch.no_grad():
                syllable_logits, tone_logits = self.forward(mel, position)

            syllable_pred = syllable_logits[0].argmax().item()
            tone_pred = tone_logits[0].argmax().item()

            syllable_probs = torch.softmax(syllable_logits[0], dim=-1)
            tone_probs = torch.softmax(tone_logits[0], dim=-1)
            syllable_prob = syllable_probs[syllable_pred].item()
            tone_prob = tone_probs[tone_pred].item()

            return PredictorOutput(
                syllable_logits=syllable_logits.cpu().numpy(),
                tone_logits=tone_logits.cpu().numpy(),
                syllable_pred=syllable_pred,
                tone_pred=tone_pred,
                syllable_prob=syllable_prob,
                tone_prob=tone_prob,
            )

        def count_parameters(self) -> tuple[int, int]:
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            return total, trainable


else:
    # Stub when PyTorch not available
    class SyllablePredictorV5:
        def __init__(self, config: SyllablePredictorConfigV5 | None = None):
            self.config = config or SyllablePredictorConfigV5()
            self.vocab = SyllableVocab()

        def forward(self, mel, position, audio_mask=None):
            batch = mel.shape[0]
            return (
                np.random.randn(batch, self.config.n_syllables),
                np.random.randn(batch, self.config.n_tones),
            )

        def predict(self, mel, position):
            syl_logits, tone_logits = self.forward(
                mel if mel.ndim == 3 else mel[np.newaxis],
                position if position.ndim >= 1 else np.array([[position]]),
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
