"""Full-Sentence Syllable+Tone Predictor V7 - CTC-based Architecture.

This model uses CTC (Connectionist Temporal Classification) for sequence prediction,
outputting per-frame probabilities for syllables and tones.

Architecture:
    Input: mel [bs, n_mels, time]
    - Audio: mel -> CNN (4x downsampling) -> [bs, time//4, d_model]
    - Transformer with RoPE and sliding window attention
    - Per-frame output heads (no position token needed)
    - Output: syllable_logits [bs, time//4, n_syl+1], tone_logits [bs, time//4, n_tone+1]

Key differences from V6:
- No position input - predicts all frames in single pass
- CTC blank token at index 0 for both syllable and tone outputs
- Per-frame predictions instead of single prediction per position
- Variable length inputs without padding to fixed size
- Single forward pass scores all syllables (vs N passes in V6)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

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
class SyllablePredictorConfigV7:
    """Configuration for CTC-based transformer model V7."""

    # Audio input
    n_mels: int = 80
    sample_rate: int = 16000
    hop_length: int = 160  # 10ms at 16kHz
    win_length: int = 400  # 25ms at 16kHz
    max_audio_frames: int = 1500  # ~15 seconds max (flexible)

    # CNN front-end downsampling
    # 32x = ~3 fps output (1000 mel frames -> 31 output frames for 10s audio)
    # 16x = ~6 fps, 8x = ~12 fps, 4x = ~25 fps
    cnn_downsample: int = 32  # Total downsampling factor
    cnn_kernel_size: int = 3

    # Transformer architecture
    d_model: int = 192
    n_heads: int = 6
    n_layers: int = 4
    dim_feedforward: int = 384
    dropout: float = 0.1

    # Sliding window attention
    attention_window: int = 32

    # Vocabulary sizes (CTC blank at index 0)
    n_syllables: int = 530  # Will be updated from vocab
    n_tones: int = 5  # Tones 1-4 + neutral
    blank_index: int = 0  # CTC blank token

    def __post_init__(self):
        vocab = load_syllable_vocab()
        if vocab:
            self.n_syllables = len(vocab)


@dataclass
class PredictorOutputV7:
    """Output from V7 syllable predictor (CTC-based)."""

    # Raw logits [batch, time, vocab_size]
    syllable_logits: NDArray[np.float32]
    tone_logits: NDArray[np.float32]

    # Decoded sequences (after CTC decoding)
    syllable_ids: Optional[List[List[int]]] = None  # [batch] of lists
    tone_ids: Optional[List[List[int]]] = None

    # Frame-level probabilities (after softmax)
    syllable_probs: Optional[NDArray[np.float32]] = None
    tone_probs: Optional[NDArray[np.float32]] = None


from .syllable_predictor_v4 import SyllableVocab


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class RotaryPositionalEmbedding(nn.Module):
        """Rotary Position Embedding (RoPE) - reused from V6."""

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


    class SlidingWindowAttentionLayer(nn.Module):
        """Transformer layer using SDPA with sliding window attention mask.

        Simplified from V6 - no global attention tokens needed for CTC.
        """

        def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int,
            dropout: float = 0.1,
            attention_window: int = 32,
        ):
            super().__init__()
            self.d_model = d_model
            self.nhead = nhead
            self.head_dim = d_model // nhead
            self.attention_window = attention_window
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
            self._attn_mask_cache = {}

        def _get_attn_mask(self, seq_len: int, device: torch.device):
            """Get sliding window attention mask (no global tokens for CTC)."""
            cache_key = (seq_len, str(device))
            if cache_key not in self._attn_mask_cache:
                idx = torch.arange(seq_len, device=device)
                # Sliding window: |q - kv| <= window_size
                local_mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() <= self.attention_window
                # Convert to additive mask
                attn_mask = torch.zeros(seq_len, seq_len, device=device)
                attn_mask.masked_fill_(~local_mask, float('-inf'))
                self._attn_mask_cache[cache_key] = attn_mask
            return self._attn_mask_cache[cache_key]

        def forward(
            self,
            src: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            src_key_padding_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            # Pre-norm
            x = self.norm1(src)
            x = src + self._sa_block(x, cos, sin, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
            return x

        def _sa_block(
            self,
            x: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            key_padding_mask: torch.Tensor | None,
        ) -> torch.Tensor:
            batch_size, seq_len, _ = x.shape

            q = self.q_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

            # Apply RoPE
            cos_head = cos.view(1, seq_len, self.nhead, self.head_dim).transpose(1, 2)
            sin_head = sin.view(1, seq_len, self.nhead, self.head_dim).transpose(1, 2)
            q = apply_rotary_pos_emb(q, cos_head, sin_head)
            k = apply_rotary_pos_emb(k, cos_head, sin_head)

            # Sliding window mask
            attn_mask = self._get_attn_mask(seq_len, x.device)

            # Combine with padding mask
            if key_padding_mask is not None:
                padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
                attn_mask = attn_mask.unsqueeze(0) + padding_mask.float().masked_fill(padding_mask, float('-inf'))

            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
            )

            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            return self.dropout1(self.out_proj(attn_output))

        def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
            return self.dropout2(self.linear2(self.activation(self.linear1(x))))


    class CTCDecoder:
        """CTC decoder for converting frame-level logits to sequences."""

        def __init__(self, blank_index: int = 0):
            self.blank_index = blank_index

        def greedy_decode(self, logits: torch.Tensor) -> List[List[int]]:
            """Greedy CTC decoding: argmax + collapse + remove blanks.

            Args:
                logits: [batch, time, vocab_size]

            Returns:
                List of decoded sequences (one per batch)
            """
            # Get most likely token at each frame
            predictions = logits.argmax(dim=-1)  # [batch, time]

            decoded = []
            for seq in predictions:
                # Collapse repeated tokens
                collapsed = []
                prev = None
                for token in seq.tolist():
                    if token != prev:
                        collapsed.append(token)
                        prev = token

                # Remove blank tokens
                result = [t for t in collapsed if t != self.blank_index]
                decoded.append(result)

            return decoded

        def decode_with_probs(
            self,
            logits: torch.Tensor,
        ) -> Tuple[List[List[int]], List[List[float]]]:
            """Greedy decode with per-token probabilities.

            Returns:
                Tuple of (decoded_sequences, per_token_probabilities)
            """
            probs = F.softmax(logits, dim=-1)
            predictions = logits.argmax(dim=-1)

            decoded_seqs = []
            decoded_probs = []

            for batch_idx in range(predictions.shape[0]):
                seq = predictions[batch_idx]
                frame_probs = probs[batch_idx]

                collapsed_ids = []
                collapsed_probs = []
                prev = None

                for t, token in enumerate(seq.tolist()):
                    if token != prev:
                        if token != self.blank_index:
                            collapsed_ids.append(token)
                            collapsed_probs.append(frame_probs[t, token].item())
                        prev = token

                decoded_seqs.append(collapsed_ids)
                decoded_probs.append(collapsed_probs)

            return decoded_seqs, decoded_probs

        def score_against_targets(
            self,
            logits: torch.Tensor,
            target_ids: List[int],
        ) -> List[float]:
            """Score pronunciation by finding max probability for each target syllable.

            This is the core grading logic for pronunciation assessment:
            1. Apply softmax to get per-frame probabilities
            2. For each target syllable, find frames where that syllable has highest prob
            3. Take the max probability across those frames as the score

            Args:
                logits: [time, vocab_size] frame-level logits (single sample)
                target_ids: List of target syllable/tone IDs to match

            Returns:
                List of per-target scores (0.0 to 1.0)
            """
            probs = F.softmax(logits, dim=-1)  # [time, vocab]

            scores = []
            for target_id in target_ids:
                # Get probability of target at each frame
                target_probs = probs[:, target_id]  # [time]

                # Find max probability for this target
                max_prob = target_probs.max().item()
                scores.append(max_prob)

            return scores

        def score_with_alignment(
            self,
            logits: torch.Tensor,
            target_ids: List[int],
        ) -> Tuple[List[float], List[int]]:
            """Score with frame alignment - assigns frames to target syllables.

            More sophisticated grading that considers temporal order:
            1. Find best frame for each target syllable in sequence
            2. Ensure frames are monotonically increasing (respects time)
            3. Return scores and aligned frame indices

            Args:
                logits: [time, vocab_size] frame-level logits
                target_ids: List of target syllable/tone IDs

            Returns:
                Tuple of (per_target_scores, aligned_frame_indices)
            """
            probs = F.softmax(logits, dim=-1)  # [time, vocab]
            n_frames = probs.shape[0]
            n_targets = len(target_ids)

            if n_targets == 0:
                return [], []

            scores = []
            aligned_frames = []
            min_frame = 0

            for i, target_id in enumerate(target_ids):
                # Search from min_frame to end (or proportional window)
                # Allow some flexibility but maintain order
                max_search_frame = min(n_frames, min_frame + (n_frames - min_frame) // max(1, n_targets - i))
                max_search_frame = max(max_search_frame, min_frame + 1)

                target_probs = probs[min_frame:max_search_frame, target_id]

                if len(target_probs) > 0:
                    best_rel_frame = target_probs.argmax().item()
                    best_frame = min_frame + best_rel_frame
                    score = target_probs[best_rel_frame].item()
                else:
                    best_frame = min_frame
                    score = 0.0

                scores.append(score)
                aligned_frames.append(best_frame)
                min_frame = best_frame + 1  # Next target must come after

            return scores, aligned_frames


    class SyllablePredictorV7(nn.Module):
        """CTC-based transformer for syllable+tone prediction.

        Architecture:
        1. Audio CNN: mel frames -> d_model (4x sequence reduction)
        2. Transformer with RoPE and sliding window attention
        3. Per-frame output heads for syllable and tone

        No position tokens needed - CTC learns alignment from targets.
        """

        def __init__(self, config: SyllablePredictorConfigV7 | None = None):
            super().__init__()

            if config is None:
                config = SyllablePredictorConfigV7()
            self.config = config

            self.vocab = SyllableVocab()

            # CNN front-end: [n_mels, time] -> [d_model, time//downsample]
            # Build CNN with enough stride-2 layers to achieve target downsampling
            # downsample=32 needs 5 layers (2^5=32), downsample=16 needs 4, etc.
            self.audio_cnn = self._build_cnn(config)
            self.cnn_downsample = config.cnn_downsample

            # RoPE for transformer
            max_seq_len = config.max_audio_frames // config.cnn_downsample + 10
            self.rope = RotaryPositionalEmbedding(dim=config.d_model, max_len=max_seq_len)

            # Sliding window transformer layers
            self.transformer_layers = nn.ModuleList([
                SlidingWindowAttentionLayer(
                    d_model=config.d_model,
                    nhead=config.n_heads,
                    dim_feedforward=config.dim_feedforward,
                    dropout=config.dropout,
                    attention_window=config.attention_window,
                )
                for _ in range(config.n_layers)
            ])

            self.output_norm = nn.LayerNorm(config.d_model)

            # CTC output heads (vocab_size + 1 for blank token at index 0)
            # Syllable head: blank + n_syllables
            self.syllable_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model, config.n_syllables + 1),
            )

            # Tone head: blank + n_tones
            self.tone_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model // 2, config.n_tones + 1),
            )

            self._init_weights()

            # CTC decoder
            self.ctc_decoder = CTCDecoder(blank_index=config.blank_index)

        def _build_cnn(self, config: SyllablePredictorConfigV7) -> nn.Sequential:
            """Build CNN frontend with configurable downsampling.

            Downsampling is achieved via stride-2 convolutions.
            32x = 5 layers, 16x = 4 layers, 8x = 3 layers, 4x = 2 layers
            """
            import math
            n_layers = int(math.log2(config.cnn_downsample))

            layers = []
            in_channels = config.n_mels

            for i in range(n_layers):
                # Gradually increase channels, cap at d_model
                if i == 0:
                    out_channels = config.d_model // 4
                elif i == 1:
                    out_channels = config.d_model // 2
                else:
                    out_channels = config.d_model

                layers.extend([
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=config.cnn_kernel_size,
                        stride=2,
                        padding=config.cnn_kernel_size // 2,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                ])
                in_channels = out_channels

            # Final projection to d_model if not already there
            if in_channels != config.d_model:
                layers.extend([
                    nn.Conv1d(in_channels, config.d_model, kernel_size=1),
                    nn.BatchNorm1d(config.d_model),
                    nn.GELU(),
                ])

            return nn.Sequential(*layers)

        def _init_weights(self):
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Conv1d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def forward(
            self,
            mel: torch.Tensor | np.ndarray,
            audio_mask: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Forward pass.

            Args:
                mel: Mel spectrogram [batch, n_mels, time]
                audio_mask: Padding mask [batch, time], True = padded

            Returns:
                Tuple of:
                - syllable_logits: [batch, time//4, n_syllables+1]
                - tone_logits: [batch, time//4, n_tones+1]
            """
            if isinstance(mel, np.ndarray):
                mel = torch.from_numpy(mel).float()

            device = next(self.parameters()).device
            mel = mel.to(device)

            batch_size = mel.shape[0]
            original_len = mel.shape[2]

            # CNN: [batch, n_mels, time] -> [batch, d_model, time//4]
            audio_embed = self.audio_cnn(mel)
            audio_embed = audio_embed.transpose(1, 2)  # [batch, time//4, d_model]
            downsampled_len = audio_embed.shape[1]

            # Create padding mask for downsampled sequence
            ds = self.cnn_downsample
            if audio_mask is not None:
                audio_mask = audio_mask.to(device)
                # Downsample mask
                ds_len = (audio_mask.shape[1] + ds - 1) // ds
                downsampled_mask = torch.zeros(batch_size, ds_len, dtype=torch.bool, device=device)
                for i in range(min(ds_len, downsampled_len)):
                    start_idx = i * ds
                    end_idx = min(start_idx + ds, audio_mask.shape[1])
                    downsampled_mask[:, i] = audio_mask[:, start_idx:end_idx].all(dim=1)

                if downsampled_mask.shape[1] > downsampled_len:
                    downsampled_mask = downsampled_mask[:, :downsampled_len]
                elif downsampled_mask.shape[1] < downsampled_len:
                    pad = torch.ones(batch_size, downsampled_len - downsampled_mask.shape[1],
                                     dtype=torch.bool, device=device)
                    downsampled_mask = torch.cat([downsampled_mask, pad], dim=1)
            else:
                downsampled_mask = None

            # Get RoPE
            cos, sin = self.rope(audio_embed)

            # Transformer
            encoded = audio_embed
            for layer in self.transformer_layers:
                encoded = layer(encoded, cos, sin, src_key_padding_mask=downsampled_mask)

            encoded = self.output_norm(encoded)

            # Per-frame output heads
            syllable_logits = self.syllable_head(encoded)  # [batch, time//4, n_syl+1]
            tone_logits = self.tone_head(encoded)  # [batch, time//4, n_tone+1]

            return syllable_logits, tone_logits

        def predict(
            self,
            mel: torch.Tensor | np.ndarray,
            audio_mask: torch.Tensor | np.ndarray | None = None,
        ) -> PredictorOutputV7:
            """Make predictions with CTC decoding."""
            if isinstance(mel, np.ndarray):
                mel = torch.from_numpy(mel).float()
            if isinstance(audio_mask, np.ndarray):
                audio_mask = torch.from_numpy(audio_mask).bool()

            if mel.dim() == 2:
                mel = mel.unsqueeze(0)
            if audio_mask is not None and audio_mask.dim() == 1:
                audio_mask = audio_mask.unsqueeze(0)

            with torch.no_grad():
                syllable_logits, tone_logits = self.forward(mel, audio_mask)

            # CTC decode
            syllable_ids, syllable_probs_list = self.ctc_decoder.decode_with_probs(syllable_logits)
            tone_ids, tone_probs_list = self.ctc_decoder.decode_with_probs(tone_logits)

            return PredictorOutputV7(
                syllable_logits=syllable_logits.cpu().numpy(),
                tone_logits=tone_logits.cpu().numpy(),
                syllable_ids=syllable_ids,
                tone_ids=tone_ids,
                syllable_probs=F.softmax(syllable_logits, dim=-1).cpu().numpy(),
                tone_probs=F.softmax(tone_logits, dim=-1).cpu().numpy(),
            )

        def get_input_lengths(self, mel_lengths: torch.Tensor) -> torch.Tensor:
            """Get output lengths after CNN downsampling (for CTC loss).

            Args:
                mel_lengths: [batch] tensor of input mel lengths

            Returns:
                [batch] tensor of output lengths (after downsampling)
            """
            # CNN does cnn_downsample (e.g., 32x) downsampling with padding
            ds = self.cnn_downsample
            return (mel_lengths + ds - 1) // ds

        def count_parameters(self) -> tuple[int, int]:
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            return total, trainable


else:
    # Stub when PyTorch not available
    class CTCDecoder:
        def __init__(self, blank_index: int = 0):
            self.blank_index = blank_index

        def greedy_decode(self, logits):
            return [[]]

    class SyllablePredictorV7:
        def __init__(self, config=None):
            self.config = config or SyllablePredictorConfigV7()
            self.vocab = SyllableVocab()
            self.ctc_decoder = CTCDecoder()

        def forward(self, mel, audio_mask=None):
            batch = mel.shape[0]
            ds = self.config.cnn_downsample
            time = mel.shape[2] // ds
            return (
                np.random.randn(batch, time, self.config.n_syllables + 1),
                np.random.randn(batch, time, self.config.n_tones + 1),
            )

        def predict(self, mel, audio_mask=None):
            syl_logits, tone_logits = self.forward(
                mel if mel.ndim == 3 else mel[np.newaxis],
                audio_mask,
            )
            return PredictorOutputV7(
                syllable_logits=syl_logits,
                tone_logits=tone_logits,
                syllable_ids=[[]],
                tone_ids=[[]],
            )

        def get_input_lengths(self, mel_lengths):
            ds = self.config.cnn_downsample
            return (mel_lengths + ds - 1) // ds

        def parameters(self):
            return iter([])

        def count_parameters(self):
            return 0, 0


if __name__ == "__main__":
    # Quick test
    print("Testing SyllablePredictorV7...")

    if TORCH_AVAILABLE:
        import torch

        config = SyllablePredictorConfigV7()
        print(f"Config: n_syllables={config.n_syllables}, n_tones={config.n_tones}")

        model = SyllablePredictorV7(config)
        total_params, trainable_params = model.count_parameters()
        print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
        print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")

        # Test forward pass with variable lengths
        for time_frames in [200, 500, 1000]:
            mel = torch.randn(2, 80, time_frames)
            syl_logits, tone_logits = model(mel)

            expected_time = (time_frames + 3) // 4
            print(f"Input: [2, 80, {time_frames}] -> "
                  f"Syllable: {list(syl_logits.shape)}, Tone: {list(tone_logits.shape)}")

            assert syl_logits.shape[0] == 2
            assert syl_logits.shape[2] == config.n_syllables + 1
            assert tone_logits.shape[0] == 2
            assert tone_logits.shape[2] == config.n_tones + 1

        # Test CTC decoding
        print("\nTesting CTC decoding...")
        mel = torch.randn(1, 80, 400)
        output = model.predict(mel)
        print(f"Decoded syllables: {len(output.syllable_ids[0])} tokens")
        print(f"Decoded tones: {len(output.tone_ids[0])} tokens")

        print("\n✓ All tests passed!")
    else:
        print("PyTorch not available, skipping tests")
