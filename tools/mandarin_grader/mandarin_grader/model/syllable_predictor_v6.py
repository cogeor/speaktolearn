"""Full-Sentence Syllable+Tone Predictor V6 - Sliding Window Attention Transformer.

This model uses SDPA with sliding window attention mask for efficient attention
on full 10-second audio sequences.

Architecture:
    Input: mel [bs, n_mels, time] + position [bs, 1]
    - Audio: mel -> CNN (4x downsampling) -> [bs, time//4, d_model]
    - Position: token [BOS, pos] concatenated with audio (like V4)
    - Transformer with RoPE and sliding window + global attention on position token
    - Attention pooling (PMA)
    - Output: syllable head (532 classes) + tone head (5 classes)

Key differences from V4:
- Uses sliding window attention (window=32) + global attention on position tokens
- Trains on full 10s sentences instead of 1s chunks
- Works on all platforms (no Triton dependency)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray


SYLLABLE_VOCAB_PATH = Path(__file__).parent.parent.parent / "data" / "syllables_v2" / "metadata.json"


def load_syllable_vocab() -> list[str]:
    """Load syllable vocabulary from metadata.json."""
    if SYLLABLE_VOCAB_PATH.exists():
        with open(SYLLABLE_VOCAB_PATH, encoding="utf-8") as f:
            data = json.load(f)
            return data.get("syllables", [])
    return []


@dataclass
class SyllablePredictorConfigV6:
    """Configuration for sliding window attention transformer model V6."""

    # Audio input - 10s max at 16kHz with hop=160 gives ~1000 frames
    n_mels: int = 80
    sample_rate: int = 16000
    hop_length: int = 160  # 10ms at 16kHz
    win_length: int = 400  # 25ms at 16kHz
    max_audio_frames: int = 1000  # ~10 seconds at 10ms per frame

    # CNN front-end (4x downsampling like V4)
    cnn_kernel_size: int = 3

    # Transformer architecture
    d_model: int = 192
    n_heads: int = 6
    n_layers: int = 4
    dim_feedforward: int = 384
    dropout: float = 0.1

    # Sliding window attention
    attention_window: int = 32  # Local attention window size (each side)
    use_global_attention: bool = True  # Position tokens have global attention

    # Vocabulary sizes
    n_syllables: int = 530
    n_tones: int = 5
    max_positions: int = 60  # Max syllables per sentence

    # Special tokens (same as V4)
    pad_token: int = 0
    bos_token: int = 1

    def __post_init__(self):
        vocab = load_syllable_vocab()
        if vocab:
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


from .syllable_predictor_v4 import SyllableVocab


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Check for FlexAttention availability (PyTorch 2.5+)
FLEX_ATTENTION_AVAILABLE = False
if TORCH_AVAILABLE:
    try:
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask
        FLEX_ATTENTION_AVAILABLE = True
    except ImportError:
        pass


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


    def create_sliding_window_mask(window_size: int, use_global: bool = True):
        """Create a mask_mod function for FlexAttention.

        Args:
            window_size: Number of tokens to attend on each side
            use_global: If True, position 0 has global attention

        Returns:
            mask_mod function compatible with create_block_mask
        """
        def mask_mod(b, h, q_idx, kv_idx):
            # Local sliding window: |q - kv| <= window_size
            local_mask = torch.abs(q_idx - kv_idx) <= window_size

            if use_global:
                # Global attention: position 0 attends everywhere and is attended by all
                global_mask = (q_idx == 0) | (kv_idx == 0)
                return local_mask | global_mask
            return local_mask

        return mask_mod


    class SlidingWindowAttentionLayer(nn.Module):
        """Transformer layer using SDPA with sliding window attention mask.

        Uses sliding window + global attention on last N tokens (position tokens).
        This is efficient on all platforms (no Triton dependency).
        """

        def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int,
            dropout: float = 0.1,
            attention_window: int = 32,
            n_global_tokens: int = 2,  # Number of tokens at END with global attention
        ):
            super().__init__()
            self.d_model = d_model
            self.nhead = nhead
            self.head_dim = d_model // nhead
            self.attention_window = attention_window
            self.n_global_tokens = n_global_tokens
            self.dropout_p = dropout

            # Projections
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)

            # FFN
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.linear2 = nn.Linear(dim_feedforward, d_model)

            # Norms and dropout
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)

            self.activation = nn.GELU()

            # Cache for attention masks (keyed by seq_len)
            self._attn_mask_cache = {}

        def _get_attn_mask(self, seq_len: int, n_global: int, device: torch.device):
            """Get or create cached sliding window attention mask.

            Global attention tokens are the LAST n_global tokens in the sequence.
            This matches V4 where [audio | BOS | pos_token] has position tokens at end.
            """
            cache_key = (seq_len, n_global, str(device))
            if cache_key not in self._attn_mask_cache:
                idx = torch.arange(seq_len, device=device)

                # Sliding window: |q - kv| <= window_size
                local_mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() <= self.attention_window

                # Global attention: last n_global tokens attend everywhere and are attended by all
                if n_global > 0:
                    global_start = seq_len - n_global
                    is_global_q = idx.unsqueeze(0) >= global_start
                    is_global_kv = idx.unsqueeze(1) >= global_start
                    global_mask = is_global_q | is_global_kv
                    mask = local_mask | global_mask
                else:
                    mask = local_mask

                # Convert to additive mask: 0 for allowed, -inf for masked
                attn_mask = torch.zeros(seq_len, seq_len, device=device)
                attn_mask.masked_fill_(~mask, float('-inf'))
                self._attn_mask_cache[cache_key] = attn_mask
            return self._attn_mask_cache[cache_key]

        def forward(
            self,
            src: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            src_key_padding_mask: torch.Tensor | None = None,
            n_global_tokens: int | None = None,
        ) -> torch.Tensor:
            """Forward pass with SDPA sliding window attention.

            Args:
                src: Input tensor [batch, seq_len, d_model]
                cos, sin: RoPE embeddings [1, seq_len, d_model]
                src_key_padding_mask: Padding mask [batch, seq_len], True = masked
                n_global_tokens: Number of tokens at end with global attention
            """
            if n_global_tokens is None:
                n_global_tokens = self.n_global_tokens
            # Pre-norm
            x = self.norm1(src)
            x = src + self._sa_block(x, cos, sin, src_key_padding_mask, n_global_tokens)
            x = x + self._ff_block(self.norm2(x))
            return x

        def _sa_block(
            self,
            x: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            key_padding_mask: torch.Tensor | None,
            n_global_tokens: int,
        ) -> torch.Tensor:
            batch_size, seq_len, _ = x.shape

            # Project to Q, K, V
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

            # Reshape for multi-head attention: [batch, seq, heads, head_dim]
            q = q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

            # Apply RoPE (cos/sin have shape [1, seq_len, d_model], reshape to [1, nhead, seq_len, head_dim])
            cos_head = cos.view(1, seq_len, self.nhead, self.head_dim).transpose(1, 2)
            sin_head = sin.view(1, seq_len, self.nhead, self.head_dim).transpose(1, 2)
            q = apply_rotary_pos_emb(q, cos_head, sin_head)
            k = apply_rotary_pos_emb(k, cos_head, sin_head)

            # Get sliding window attention mask with global tokens at end
            attn_mask = self._get_attn_mask(seq_len, n_global_tokens, x.device)

            # Combine with padding mask if provided
            if key_padding_mask is not None:
                # key_padding_mask: [batch, seq_len] -> [batch, 1, 1, seq_len]
                padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
                # Add to attention mask (masked positions get -inf)
                attn_mask = attn_mask.unsqueeze(0) + padding_mask.float().masked_fill(padding_mask, float('-inf'))

            # SDPA with combined mask
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
            )

            # Reshape back
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            return self.dropout1(self.out_proj(attn_output))

        def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
            return self.dropout2(self.linear2(self.activation(self.linear1(x))))

    # Alias for backwards compatibility
    FlexAttentionLayer = SlidingWindowAttentionLayer


    class SyllablePredictorV6(nn.Module):
        """Sliding window attention transformer for syllable+tone prediction.

        Architecture (matches V4 with sliding window attention):
        1. Audio CNN: mel frames -> d_model (4x sequence reduction, like V4)
        2. Position as TOKEN: [audio | BOS | pos_token] concatenated (like V4)
        3. Transformer with RoPE and sliding window + global attention on position tokens
        4. Attention pooling (PMA) with padding mask
        5. Dual output heads (syllable + tone)
        """

        def __init__(self, config: SyllablePredictorConfigV6 | None = None):
            super().__init__()

            if config is None:
                config = SyllablePredictorConfigV6()
            self.config = config

            self.vocab = SyllableVocab()
            vocab_size = len(self.vocab)

            # CNN front-end: [n_mels, time] -> [d_model, time//4] (4x like V4)
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
            )

            # Position token embedding (reuse vocab embedding like V4)
            # Tokens: 0=PAD, 1=BOS, 2+=position indices
            self.position_embed = nn.Embedding(vocab_size, config.d_model, padding_idx=0)

            # Modality embeddings (like V4)
            self.audio_type_embed = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
            self.position_type_embed = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)

            # RoPE for transformer (dim=d_model like V4)
            max_seq_len = config.max_audio_frames // 4 + config.max_positions + 2
            self.rope = RotaryPositionalEmbedding(dim=config.d_model, max_len=max_seq_len)

            # Sliding window transformer layers
            self.transformer_layers = nn.ModuleList([
                SlidingWindowAttentionLayer(
                    d_model=config.d_model,
                    nhead=config.n_heads,
                    dim_feedforward=config.dim_feedforward,
                    dropout=config.dropout,
                    attention_window=config.attention_window,
                    n_global_tokens=2,  # BOS + position token have global attention
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
                    if module.padding_idx is not None:
                        module.weight.data[module.padding_idx].zero_()
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
                position: Position index [batch, 1] or [batch]
                audio_mask: Padding mask [batch, time], True = padded

            Returns:
                Tuple of (syllable_logits, tone_logits)
            """
            if isinstance(mel, np.ndarray):
                mel = torch.from_numpy(mel).float()
            if isinstance(position, np.ndarray):
                position = torch.from_numpy(position).long()

            device = next(self.parameters()).device
            mel = mel.to(device)
            position = position.to(device)

            if position.dim() == 1:
                position = position.unsqueeze(1)

            batch_size = mel.shape[0]
            original_audio_len = mel.shape[2]

            # CNN front-end: [batch, n_mels, time] -> [batch, d_model, time//4]
            audio_embed = self.audio_cnn(mel)
            audio_embed = audio_embed.transpose(1, 2)  # [batch, time//4, d_model]
            downsampled_len = audio_embed.shape[1]

            # Add audio type embedding
            audio_embed = audio_embed + self.audio_type_embed

            # Create position tokens: [BOS, position_index]
            # Position token = 2 + syllable_idx (0=PAD, 1=BOS, 2+=positions)
            bos_token = torch.full((batch_size, 1), self.config.bos_token, dtype=torch.long, device=device)
            pos_token = position + 2  # offset by special tokens
            position_ids = torch.cat([bos_token, pos_token], dim=1)  # [batch, 2]

            # Embed position tokens
            pos_embed = self.position_embed(position_ids)  # [batch, 2, d_model]
            pos_embed = pos_embed + self.position_type_embed

            # Concatenate: [audio | BOS | pos_token] like V4
            combined = torch.cat([audio_embed, pos_embed], dim=1)  # [batch, audio_len + 2, d_model]

            # Create combined padding mask
            if audio_mask is not None:
                audio_mask = audio_mask.to(device)
                # Downsample audio mask (4x)
                ds_len = (audio_mask.shape[1] + 3) // 4
                downsampled_mask = torch.zeros(batch_size, ds_len, dtype=torch.bool, device=device)
                for i in range(ds_len):
                    start_idx = i * 4
                    end_idx = min(start_idx + 4, audio_mask.shape[1])
                    downsampled_mask[:, i] = audio_mask[:, start_idx:end_idx].all(dim=1)

                # Truncate/pad to match actual downsampled length
                if downsampled_mask.shape[1] > downsampled_len:
                    downsampled_mask = downsampled_mask[:, :downsampled_len]
                elif downsampled_mask.shape[1] < downsampled_len:
                    pad = torch.ones(batch_size, downsampled_len - downsampled_mask.shape[1],
                                     dtype=torch.bool, device=device)
                    downsampled_mask = torch.cat([downsampled_mask, pad], dim=1)
            else:
                # No padding in audio
                downsampled_mask = torch.zeros(batch_size, downsampled_len, dtype=torch.bool, device=device)

            # Position tokens are never masked
            pos_mask = torch.zeros(batch_size, 2, dtype=torch.bool, device=device)
            combined_mask = torch.cat([downsampled_mask, pos_mask], dim=1)

            # Get RoPE
            cos, sin = self.rope(combined)

            # Transformer with sliding window + global attention on last 2 tokens
            encoded = combined
            for layer in self.transformer_layers:
                encoded = layer(encoded, cos, sin, src_key_padding_mask=combined_mask, n_global_tokens=2)

            # Attention pooling (PMA) with mask
            query = self.pool_query.expand(batch_size, -1, -1)
            pooled, _ = self.pool_attention(
                query=query,
                key=encoded,
                value=encoded,
                key_padding_mask=combined_mask,
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
            audio_mask: torch.Tensor | np.ndarray | None = None,
        ) -> PredictorOutput:
            """Make predictions."""
            if isinstance(mel, np.ndarray):
                mel = torch.from_numpy(mel).float()
            if isinstance(position, np.ndarray):
                position = torch.from_numpy(position).long()
            if isinstance(audio_mask, np.ndarray):
                audio_mask = torch.from_numpy(audio_mask).bool()

            if mel.dim() == 2:
                mel = mel.unsqueeze(0)
            if position.dim() == 0:
                position = position.unsqueeze(0).unsqueeze(0)
            elif position.dim() == 1:
                position = position.unsqueeze(1)
            if audio_mask is not None and audio_mask.dim() == 1:
                audio_mask = audio_mask.unsqueeze(0)

            with torch.no_grad():
                syllable_logits, tone_logits = self.forward(mel, position, audio_mask)

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
    class SyllablePredictorV6:
        def __init__(self, config=None):
            self.config = config or SyllablePredictorConfigV6()
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
                position if position.ndim >= 1 else np.array([position]),
            )
            return PredictorOutput(
                syllable_logits=syl_logits,
                tone_logits=tone_logits,
                syllable_pred=0,
                tone_pred=0,
            )

        def parameters(self):
            return iter([])

        def count_parameters(self):
            return 0, 0
