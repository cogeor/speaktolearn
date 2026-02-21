"""Full-Sentence Syllable+Tone Predictor V7 - CTC-based BiLSTM Architecture.

This model uses CTC (Connectionist Temporal Classification) for sequence prediction,
outputting per-frame probabilities for syllables and tones.

Architecture:
    Input: mel [bs, n_mels, time]
    - Audio: mel -> CNN (4x downsampling) -> [bs, time//4, d_model]
    - Bidirectional LSTM encoder (2 layers)
    - Per-frame output heads (simple Linear)
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
    """Configuration for CTC-based BiLSTM model V7."""

    # Audio input
    n_mels: int = 80
    sample_rate: int = 16000
    hop_length: int = 160  # 10ms at 16kHz
    win_length: int = 400  # 25ms at 16kHz
    max_audio_frames: int = 1500  # ~15 seconds max (flexible)

    # CNN front-end downsampling (4x = 2 layers with stride 2)
    cnn_downsample: int = 4
    cnn_kernel_size: int = 3

    # BiLSTM architecture
    d_model: int = 192
    lstm_layers: int = 2
    lstm_hidden: int = 96  # d_model // 2, bidirectional doubles this
    dropout: float = 0.1

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

    class CTCDecoder:
        """CTC decoder for converting frame-level logits to sequences."""

        def __init__(self, blank_index: int = 0):
            self.blank_index = blank_index

        def greedy_decode(self, logits: torch.Tensor) -> List[List[int]]:
            """Greedy CTC decoding: argmax + collapse + remove blanks."""
            predictions = logits.argmax(dim=-1)

            decoded = []
            for seq in predictions:
                collapsed = []
                prev = None
                for token in seq.tolist():
                    if token != prev:
                        collapsed.append(token)
                        prev = token
                result = [t for t in collapsed if t != self.blank_index]
                decoded.append(result)

            return decoded

        def decode_with_probs(
            self,
            logits: torch.Tensor,
        ) -> Tuple[List[List[int]], List[List[float]]]:
            """Greedy decode with per-token probabilities."""
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
            """Score pronunciation by finding max probability for each target syllable."""
            probs = F.softmax(logits, dim=-1)

            scores = []
            for target_id in target_ids:
                target_probs = probs[:, target_id]
                max_prob = target_probs.max().item()
                scores.append(max_prob)

            return scores

        def score_with_alignment(
            self,
            logits: torch.Tensor,
            target_ids: List[int],
        ) -> Tuple[List[float], List[int]]:
            """Score with frame alignment - assigns frames to target syllables."""
            probs = F.softmax(logits, dim=-1)
            n_frames = probs.shape[0]
            n_targets = len(target_ids)

            if n_targets == 0:
                return [], []

            scores = []
            aligned_frames = []
            min_frame = 0

            for i, target_id in enumerate(target_ids):
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
                min_frame = best_frame + 1

            return scores, aligned_frames


    class SyllablePredictorV7(nn.Module):
        """CTC-based BiLSTM for syllable+tone prediction.

        Architecture:
        1. Audio CNN: mel frames -> d_model (4x sequence reduction)
        2. Bidirectional LSTM encoder (2 layers)
        3. Per-frame Linear output heads for syllable and tone
        """

        def __init__(self, config: SyllablePredictorConfigV7 | None = None):
            super().__init__()

            if config is None:
                config = SyllablePredictorConfigV7()
            self.config = config

            self.vocab = SyllableVocab()

            # CNN front-end: [n_mels, time] -> [d_model, time//4]
            # 2 layers with stride 2 each = 4x downsampling
            self.audio_cnn = nn.Sequential(
                nn.Conv1d(config.n_mels, config.d_model // 2, kernel_size=config.cnn_kernel_size, stride=2, padding=config.cnn_kernel_size // 2),
                nn.BatchNorm1d(config.d_model // 2),
                nn.GELU(),
                nn.Conv1d(config.d_model // 2, config.d_model, kernel_size=config.cnn_kernel_size, stride=2, padding=config.cnn_kernel_size // 2),
                nn.BatchNorm1d(config.d_model),
                nn.GELU(),
            )
            self.cnn_downsample = config.cnn_downsample

            # Bidirectional LSTM encoder
            self.lstm = nn.LSTM(
                input_size=config.d_model,
                hidden_size=config.lstm_hidden,
                num_layers=config.lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=config.dropout if config.lstm_layers > 1 else 0,
            )

            # Simple Linear output heads (vocab_size + 1 for blank token at index 0)
            self.syllable_head = nn.Linear(config.d_model, config.n_syllables + 1)
            self.tone_head = nn.Linear(config.d_model, config.n_tones + 1)

            self._init_weights()

            # CTC decoder
            self.ctc_decoder = CTCDecoder(blank_index=config.blank_index)

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
                audio_mask: Padding mask [batch, time], True = padded (unused in BiLSTM)

            Returns:
                Tuple of:
                - syllable_logits: [batch, time//4, n_syllables+1]
                - tone_logits: [batch, time//4, n_tones+1]
            """
            if isinstance(mel, np.ndarray):
                mel = torch.from_numpy(mel).float()

            device = next(self.parameters()).device
            mel = mel.to(device)

            # CNN: [batch, n_mels, time] -> [batch, d_model, time//4]
            x = self.audio_cnn(mel)
            x = x.transpose(1, 2)  # [batch, time//4, d_model]

            # BiLSTM: [batch, time//4, d_model] -> [batch, time//4, d_model]
            x, _ = self.lstm(x)

            # Per-frame output heads
            syllable_logits = self.syllable_head(x)  # [batch, time//4, n_syl+1]
            tone_logits = self.tone_head(x)  # [batch, time//4, n_tone+1]

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
            """Get output lengths after CNN downsampling (for CTC loss)."""
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
    print("Testing SyllablePredictorV7 (BiLSTM)...")

    if TORCH_AVAILABLE:
        import torch

        config = SyllablePredictorConfigV7()
        print(f"Config: n_syllables={config.n_syllables}, n_tones={config.n_tones}")
        print(f"Config: lstm_layers={config.lstm_layers}, lstm_hidden={config.lstm_hidden}")

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

        print("\nAll tests passed!")
    else:
        print("PyTorch not available, skipping tests")
