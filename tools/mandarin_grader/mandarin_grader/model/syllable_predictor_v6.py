"""Full-Sentence Syllable+Tone Predictor V6 - FlexAttention Transformer.

This model uses PyTorch FlexAttention for efficient O(n×w) sliding window attention,
providing true sparse attention computation instead of masked full attention.

Architecture:
    Input: mel [bs, n_mels, time] + position [bs, 1]
    - Audio: mel -> CNN (8x downsampling) -> [bs, time//8, d_model]
    - Position: embedding -> [bs, 1, d_model] -> broadcast to audio
    - FlexAttention Transformer with RoPE and sliding window + global attention
    - Attention pooling (PMA)
    - Output: syllable head (532 classes) + tone head (5 classes)

Key differences from V5:
- Uses FlexAttention instead of scaled_dot_product_attention
- True O(n×w) complexity via block-sparse masks
- Requires torch.compile for optimal performance
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
    """Configuration for FlexAttention transformer model V6."""

    # Audio input - 10s max at 16kHz with hop=160 gives ~1000 frames
    n_mels: int = 80
    sample_rate: int = 16000
    hop_length: int = 160  # 10ms at 16kHz
    win_length: int = 400  # 25ms at 16kHz
    max_audio_frames: int = 1000  # ~10 seconds at 10ms per frame

    # CNN front-end
    cnn_kernel_size: int = 3

    # Transformer architecture (same as V5)
    d_model: int = 192
    n_heads: int = 6
    n_layers: int = 4
    dim_feedforward: int = 384
    dropout: float = 0.1

    # FlexAttention sliding window
    attention_window: int = 32  # Local attention window size (each side)
    use_global_attention: bool = True  # Position 0 has global attention

    # Vocabulary sizes
    n_syllables: int = 530
    n_tones: int = 5
    max_positions: int = 60  # Max syllables per sentence

    pad_token: int = 0

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


    class FlexAttentionLayer(nn.Module):
        """Transformer layer using FlexAttention for efficient sparse attention.

        Uses sliding window + optional global attention via FlexAttention,
        which compiles into a fused kernel with true O(n×w) complexity.
        """

        def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int,
            dropout: float = 0.1,
            attention_window: int = 32,
            use_global_attention: bool = True,
        ):
            super().__init__()
            self.d_model = d_model
            self.nhead = nhead
            self.head_dim = d_model // nhead
            self.attention_window = attention_window
            self.use_global_attention = use_global_attention
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

            # Cache for block masks (keyed by seq_len)
            self._block_mask_cache = {}

        def _get_block_mask(self, seq_len: int, device: torch.device):
            """Get or create cached block mask for given sequence length."""
            cache_key = (seq_len, str(device))
            if cache_key not in self._block_mask_cache:
                mask_mod = create_sliding_window_mask(
                    self.attention_window,
                    self.use_global_attention
                )
                # B=None, H=None means the mask is the same for all batches/heads
                block_mask = create_block_mask(
                    mask_mod,
                    B=None,
                    H=None,
                    Q_LEN=seq_len,
                    KV_LEN=seq_len,
                    device=device,
                )
                self._block_mask_cache[cache_key] = block_mask
            return self._block_mask_cache[cache_key]

        def forward(
            self,
            src: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
        ) -> torch.Tensor:
            """Forward pass with FlexAttention.

            Args:
                src: Input tensor [batch, seq_len, d_model]
                cos, sin: RoPE embeddings [1, seq_len, head_dim]
            """
            # Pre-norm
            x = self.norm1(src)
            x = src + self._sa_block(x, cos, sin)
            x = x + self._ff_block(self.norm2(x))
            return x

        def _sa_block(
            self,
            x: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
        ) -> torch.Tensor:
            batch_size, seq_len, _ = x.shape

            # Project to Q, K, V
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

            # Reshape for multi-head attention: [batch, seq, heads, head_dim]
            q = q.view(batch_size, seq_len, self.nhead, self.head_dim)
            k = k.view(batch_size, seq_len, self.nhead, self.head_dim)
            v = v.view(batch_size, seq_len, self.nhead, self.head_dim)

            # Apply RoPE
            cos_head = cos.view(1, seq_len, 1, self.head_dim)
            sin_head = sin.view(1, seq_len, 1, self.head_dim)
            q = apply_rotary_pos_emb(q, cos_head, sin_head)
            k = apply_rotary_pos_emb(k, cos_head, sin_head)

            # Transpose for FlexAttention: [batch, heads, seq, head_dim]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Get block mask for this sequence length
            block_mask = self._get_block_mask(seq_len, x.device)

            # FlexAttention with sliding window + global attention
            if FLEX_ATTENTION_AVAILABLE:
                attn_output = flex_attention(q, k, v, block_mask=block_mask)
            else:
                # Fallback to scaled_dot_product_attention (full attention)
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.dropout_p if self.training else 0.0,
                )

            # Reshape back
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            return self.dropout1(self.out_proj(attn_output))

        def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
            return self.dropout2(self.linear2(self.activation(self.linear1(x))))


    class SyllablePredictorV6(nn.Module):
        """FlexAttention transformer for syllable+tone prediction.

        Architecture:
        1. Audio CNN: mel frames -> d_model (8x sequence reduction)
        2. Position embedding: broadcast to all audio frames
        3. FlexAttention Transformer with sliding window + global attention
        4. Attention pooling (PMA)
        5. Dual output heads (syllable + tone)
        """

        def __init__(self, config: SyllablePredictorConfigV6 | None = None):
            super().__init__()

            if config is None:
                config = SyllablePredictorConfigV6()
            self.config = config

            self.vocab = SyllableVocab()

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

            # Position embedding
            self.position_embed = nn.Embedding(config.max_positions, config.d_model)

            # Audio type embedding
            self.audio_type_embed = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)

            # RoPE for transformer (uses head_dim, not d_model)
            max_seq_len = config.max_audio_frames // 8 + 1
            head_dim = config.d_model // config.n_heads
            self.rope = RotaryPositionalEmbedding(dim=head_dim, max_len=max_seq_len)

            # FlexAttention transformer layers
            self.transformer_layers = nn.ModuleList([
                FlexAttentionLayer(
                    d_model=config.d_model,
                    nhead=config.n_heads,
                    dim_feedforward=config.dim_feedforward,
                    dropout=config.dropout,
                    attention_window=config.attention_window,
                    use_global_attention=config.use_global_attention,
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
                position: Position index [batch, 1] or [batch]
                audio_mask: Optional mask (currently unused with FlexAttention)

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

            # CNN front-end: [batch, n_mels, time] -> [batch, d_model, time//8]
            audio_embed = self.audio_cnn(mel)
            audio_embed = audio_embed.transpose(1, 2)  # [batch, time//8, d_model]

            # Add audio type embedding
            audio_embed = audio_embed + self.audio_type_embed

            # Position embedding: broadcast to all frames
            pos_embed = self.position_embed(position)  # [batch, 1, d_model]
            audio_embed = audio_embed + pos_embed

            # Get RoPE
            cos, sin = self.rope(audio_embed)

            # FlexAttention transformer
            encoded = audio_embed
            for layer in self.transformer_layers:
                encoded = layer(encoded, cos, sin)

            # Attention pooling (PMA)
            query = self.pool_query.expand(batch_size, -1, -1)
            pooled, _ = self.pool_attention(
                query=query,
                key=encoded,
                value=encoded,
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
            """Make predictions."""
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
