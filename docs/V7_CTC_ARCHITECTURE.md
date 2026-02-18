# V7: CTC-Based Syllable Grading Architecture

**Version:** 1.0
**Date:** 2026-02-18
**Status:** Draft

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Model Design](#3-model-design)
4. [Training Pipeline](#4-training-pipeline)
5. [Grading Pipeline](#5-grading-pipeline)
6. [Flutter Integration](#6-flutter-integration)
7. [Testing & Benchmarking](#7-testing--benchmarking)
8. [Migration Plan](#8-migration-plan)

---

## 1. Executive Summary

### 1.1 Problem with V6 (Position-Query)

V6 uses a position-query approach:
```
Query: "What syllable is at position N?"
Model: Attends to audio, returns P(syllable | audio, position=N)
```

**Critical Issue:** This assumes position N maps to a predictable audio region. Analysis shows:
- Position 0: 83% accuracy
- Position 7: 40% accuracy
- Systematic **syllable shift errors** in longer sentences

Root cause: Model learns AISHELL3 speaker timing. TTS/real recordings have different timing.

### 1.2 CTC Solution

CTC (Connectionist Temporal Classification) removes position-timing dependency:

```
Input:  [audio frames 0...T]
Output: [syllable at each frame, or blank]
         ↓ collapse
Result: Detected syllable sequence
```

**Key Advantage:** No assumption about where syllables appear. Model learns to find them.

### 1.3 Grading with CTC

CTC doesn't directly give per-syllable scores, but we can compute them:

1. **Forward-backward algorithm** gives alignment posteriors
2. **Constrained decoding** with expected sequence gives per-position probabilities
3. **Frame-level aggregation** gives per-syllable confidence

### 1.4 Success Criteria

| Metric | V6 Baseline | V7 Target |
|--------|-------------|-----------|
| Overall TTS accuracy | 71.5% | >85% |
| Position 5-7 accuracy | 40-60% | >80% |
| Pulled recording accuracy | ~20% | >60% |
| Inference latency (10s audio) | ~200ms | <300ms |
| Model size (ONNX INT8) | 80MB | <100MB |

---

## 2. Architecture Overview

### 2.1 High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING                                  │
├─────────────────────────────────────────────────────────────────┤
│  Audio ──► Mel ──► Encoder ──► Frame Logits ──► CTC Loss        │
│                                     ↓                            │
│                              Syllable Sequence (target)          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        INFERENCE (Grading)                       │
├─────────────────────────────────────────────────────────────────┤
│  Audio ──► Mel ──► Encoder ──► Frame Logits                     │
│                                     ↓                            │
│                    Forward-Backward (with expected sequence)     │
│                                     ↓                            │
│                         Alignment Posteriors                     │
│                                     ↓                            │
│                       Per-Syllable Scores                        │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Components

| Component | Purpose | Implementation |
|-----------|---------|----------------|
| Mel Extractor | Audio → Spectrogram | Same as V6 (80 mel, hop=160) |
| Encoder | Mel → Frame Features | CNN + Transformer (from V6) |
| CTC Head | Frame Features → Logits | Linear(d_model, vocab_size+1) |
| CTC Loss | Training objective | `torch.nn.CTCLoss` |
| Forward-Backward | Alignment posteriors | Custom implementation |
| Score Aggregator | Posteriors → Grades | Per-syllable confidence |

---

## 3. Model Design

### 3.1 Encoder (Reuse from V6)

```python
class CTCEncoder(nn.Module):
    """
    Audio encoder for CTC model.
    Reuses V6 architecture: CNN frontend + Transformer layers.
    """
    def __init__(self, config: CTCConfig):
        super().__init__()

        # CNN frontend: mel [B, 80, T] -> features [B, T//4, d_model]
        self.cnn = nn.Sequential(
            nn.Conv1d(80, config.d_model // 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(config.d_model // 2, config.d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Transformer layers with RoPE
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.n_layers)
        ])

        # RoPE embeddings
        self.rope = RotaryPositionalEmbedding(config.d_model // config.n_heads)

    def forward(self, mel: Tensor, audio_mask: Tensor) -> Tensor:
        # mel: [B, 80, T]
        x = self.cnn(mel)  # [B, d_model, T//4]
        x = x.transpose(1, 2)  # [B, T//4, d_model]

        # Downsample mask
        mask = audio_mask[:, ::4]

        # Apply transformer layers
        cos, sin = self.rope(x)
        for layer in self.layers:
            x = layer(x, cos, sin, mask)

        return x  # [B, T//4, d_model]
```

### 3.2 CTC Head

```python
class CTCHead(nn.Module):
    """
    CTC output head: frame features -> syllable logits.

    Output vocabulary:
    - Index 0: blank token
    - Index 1-530: syllable tokens (toneless)
    - Total: 531 classes
    """
    def __init__(self, d_model: int, vocab_size: int = 530):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size + 1)  # +1 for blank

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, d_model]
        return self.proj(x)  # [B, T, vocab_size+1]
```

### 3.3 Full Model

```python
class SyllablePredictorV7(nn.Module):
    """
    CTC-based syllable predictor V7.

    Differences from V6:
    - No position tokens
    - No attention pooling
    - CTC loss instead of cross-entropy
    - Frame-level output instead of single prediction
    """
    def __init__(self, config: V7Config):
        super().__init__()
        self.encoder = CTCEncoder(config)
        self.ctc_head = CTCHead(config.d_model, config.vocab_size)
        self.blank_idx = 0

    def forward(self, mel: Tensor, audio_mask: Tensor) -> Tensor:
        """
        Forward pass for training.

        Args:
            mel: [B, 80, T] mel spectrogram
            audio_mask: [B, T] padding mask (True = padded)

        Returns:
            logits: [B, T//4, vocab_size+1] frame-level logits
        """
        features = self.encoder(mel, audio_mask)
        logits = self.ctc_head(features)
        return logits
```

### 3.4 Configuration

```python
@dataclass
class V7Config:
    # Audio
    n_mels: int = 80
    sample_rate: int = 16000
    hop_length: int = 160
    max_audio_frames: int = 1000  # 10 seconds

    # Encoder
    d_model: int = 480
    n_heads: int = 6
    n_layers: int = 8
    dim_feedforward: int = 1440
    dropout: float = 0.1

    # CTC
    vocab_size: int = 530  # Toneless syllables
    blank_idx: int = 0

    # Derived
    @property
    def encoder_frames(self) -> int:
        return self.max_audio_frames // 4  # After CNN downsampling
```

---

## 4. Training Pipeline

### 4.1 Dataset

Reuse AISHELL3 dataset with modifications:

```python
class CTCDataset(Dataset):
    """
    Dataset for CTC training.

    Each sample provides:
    - mel: [80, T] mel spectrogram
    - target_ids: [S] syllable indices (no blanks)
    - input_length: T//4 (encoder output length)
    - target_length: S (number of syllables)
    """
    def __init__(
        self,
        data_dir: Path,
        vocab: SyllableVocab,
        max_duration_s: float = 10.0,
        augment: bool = True,
        noise_snr_db: tuple[float, float] = (10.0, 30.0),
    ):
        self.data = self._load_data(data_dir)
        self.vocab = vocab
        self.max_frames = int(max_duration_s * 100)  # 10ms per frame
        self.augment = augment
        self.noise_snr_db = noise_snr_db

    def __getitem__(self, idx) -> dict:
        item = self.data[idx]

        # Load and process audio
        audio = self._load_audio(item['audio_path'])

        # Augmentations (same as V6)
        if self.augment:
            audio = self._apply_augmentations(audio)

        # Compute mel
        mel = self._compute_mel(audio)

        # Encode syllables (toneless)
        syllables = item['pinyin'].split()
        target_ids = [self.vocab.encode(s) for s in syllables]

        return {
            'mel': mel,
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'input_length': mel.shape[1] // 4,
            'target_length': len(target_ids),
        }
```

### 4.2 Collate Function

```python
def ctc_collate_fn(batch: list[dict]) -> dict:
    """
    Collate batch for CTC training.

    Pads mel to max length, concatenates targets.
    """
    # Find max mel length
    max_mel_len = max(item['mel'].shape[1] for item in batch)

    # Pad mels
    mels = []
    audio_masks = []
    for item in batch:
        mel = item['mel']
        pad_len = max_mel_len - mel.shape[1]
        if pad_len > 0:
            mel = F.pad(mel, (0, pad_len))
            mask = torch.cat([
                torch.zeros(mel.shape[1] - pad_len, dtype=torch.bool),
                torch.ones(pad_len, dtype=torch.bool)
            ])
        else:
            mask = torch.zeros(mel.shape[1], dtype=torch.bool)
        mels.append(mel)
        audio_masks.append(mask)

    # Stack mels and masks
    mels = torch.stack(mels)
    audio_masks = torch.stack(audio_masks)

    # Concatenate targets (CTC loss expects flat targets)
    targets = torch.cat([item['target_ids'] for item in batch])
    input_lengths = torch.tensor([item['input_length'] for item in batch])
    target_lengths = torch.tensor([item['target_length'] for item in batch])

    return {
        'mel': mels,
        'audio_mask': audio_masks,
        'targets': targets,
        'input_lengths': input_lengths,
        'target_lengths': target_lengths,
    }
```

### 4.3 Training Loop

```python
def train_ctc(
    model: SyllablePredictorV7,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
):
    """
    Train CTC model.
    """
    optimizer = AdamW(model.parameters(), lr=config.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.total_steps,
    )
    ctc_loss = nn.CTCLoss(blank=model.blank_idx, reduction='mean')

    for epoch in range(config.epochs):
        model.train()
        for batch in train_loader:
            # Forward pass
            logits = model(batch['mel'], batch['audio_mask'])
            log_probs = F.log_softmax(logits, dim=-1)

            # Transpose for CTC: [T, B, C]
            log_probs = log_probs.transpose(0, 1)

            # CTC loss
            loss = ctc_loss(
                log_probs,
                batch['targets'],
                batch['input_lengths'],
                batch['target_lengths'],
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()

        # Validation
        val_metrics = evaluate_ctc(model, val_loader)
        print(f"Epoch {epoch}: loss={loss:.4f}, WER={val_metrics['wer']:.2%}")
```

### 4.4 Augmentations

Same as V6 augmented training:

| Augmentation | Range | Purpose |
|--------------|-------|---------|
| Speed | ±10% | Tempo variation |
| Noise SNR | 10-30 dB | Noise robustness |
| Random padding | Start/end | Position invariance |
| Volume | ±12 dB | Level normalization |

---

## 5. Grading Pipeline

### 5.1 Forward-Backward Algorithm

The key to CTC grading is computing **alignment posteriors**: P(syllable k aligned to frame t | audio, expected_sequence).

```python
def ctc_forward_backward(
    log_probs: Tensor,  # [T, V] log probabilities
    target_ids: list[int],  # [S] expected syllable indices
    blank_idx: int = 0,
) -> tuple[Tensor, Tensor, float]:
    """
    Compute CTC forward-backward for alignment posteriors.

    Returns:
        alpha: [T, 2S+1] forward probabilities
        beta: [T, 2S+1] backward probabilities
        total_prob: log P(target | audio)
    """
    T = log_probs.shape[0]
    S = len(target_ids)

    # Expand target with blanks: [blank, s0, blank, s1, blank, ...]
    # Length = 2*S + 1
    expanded = [blank_idx]
    for s in target_ids:
        expanded.extend([s, blank_idx])
    L = len(expanded)

    # Forward pass
    alpha = torch.full((T, L), float('-inf'))
    alpha[0, 0] = log_probs[0, blank_idx]
    alpha[0, 1] = log_probs[0, expanded[1]]

    for t in range(1, T):
        for s in range(L):
            # Stay or move from previous
            alpha[t, s] = alpha[t-1, s]
            if s > 0:
                alpha[t, s] = torch.logaddexp(alpha[t, s], alpha[t-1, s-1])
            # Skip blank (only if not blank and same as 2 positions back)
            if s > 1 and expanded[s] != blank_idx and expanded[s] != expanded[s-2]:
                alpha[t, s] = torch.logaddexp(alpha[t, s], alpha[t-1, s-2])
            # Add emission
            alpha[t, s] += log_probs[t, expanded[s]]

    # Backward pass (similar)
    beta = torch.full((T, L), float('-inf'))
    beta[-1, -1] = 0
    beta[-1, -2] = 0

    for t in range(T-2, -1, -1):
        for s in range(L):
            # ... (symmetric to forward)
            pass

    # Total probability
    total_prob = torch.logaddexp(alpha[-1, -1], alpha[-1, -2])

    return alpha, beta, total_prob
```

### 5.2 Per-Syllable Scores

```python
def compute_syllable_scores(
    log_probs: Tensor,  # [T, V]
    alpha: Tensor,  # [T, L]
    beta: Tensor,  # [T, L]
    total_prob: float,
    target_ids: list[int],
) -> list[float]:
    """
    Compute per-syllable confidence scores from CTC posteriors.

    For each syllable k, score = sum over aligned frames of:
        P(syllable k at frame t | audio, target)
    """
    T = log_probs.shape[0]
    S = len(target_ids)

    scores = []
    for k in range(S):
        # Syllable k is at position 2k+1 in expanded sequence
        s = 2 * k + 1

        # Sum posterior over all frames
        syllable_prob = 0.0
        for t in range(T):
            # Posterior = alpha * beta / total
            posterior = torch.exp(alpha[t, s] + beta[t, s] - total_prob)
            syllable_prob += posterior.item()

        # Normalize by expected alignment length
        expected_frames = T / S
        score = min(1.0, syllable_prob / expected_frames)
        scores.append(score)

    return scores
```

### 5.3 Alternative: Constrained Greedy Decoding

Simpler approach that works well in practice:

```python
def grade_with_constrained_decode(
    log_probs: Tensor,  # [T, V]
    target_ids: list[int],
    blank_idx: int = 0,
) -> list[float]:
    """
    Grade using constrained greedy decoding.

    For each target syllable, find frames where it has highest probability
    and compute average confidence.
    """
    T, V = log_probs.shape
    probs = torch.softmax(log_probs, dim=-1)

    scores = []
    frame_ptr = 0

    for k, syl_id in enumerate(target_ids):
        # Find frames where this syllable is most likely
        syllable_frames = []

        while frame_ptr < T:
            # Is this frame best explained by target syllable or blank?
            syl_prob = probs[frame_ptr, syl_id].item()
            blank_prob = probs[frame_ptr, blank_idx].item()

            # Check if next syllable is more likely
            next_syl_id = target_ids[k+1] if k+1 < len(target_ids) else None
            next_prob = probs[frame_ptr, next_syl_id].item() if next_syl_id else 0

            if syl_prob > next_prob or next_syl_id is None:
                syllable_frames.append((frame_ptr, syl_prob))
                frame_ptr += 1

                # Stop if blank dominates and we have enough frames
                if blank_prob > syl_prob and len(syllable_frames) > 3:
                    break
            else:
                break

        # Compute score from collected frames
        if syllable_frames:
            avg_prob = sum(p for _, p in syllable_frames) / len(syllable_frames)
            scores.append(avg_prob)
        else:
            scores.append(0.0)  # Syllable not found

    return scores
```

### 5.4 Full Grading Function

```python
class CTCGrader:
    """
    Grade pronunciations using CTC model.
    """
    def __init__(self, model: SyllablePredictorV7, vocab: SyllableVocab):
        self.model = model
        self.vocab = vocab
        self.model.eval()

    def grade(
        self,
        audio: np.ndarray,
        expected_pinyin: list[str],
    ) -> GradeResult:
        """
        Grade audio against expected syllables.

        Args:
            audio: Audio samples (16kHz)
            expected_pinyin: Expected syllables e.g. ['ni', 'hao']

        Returns:
            GradeResult with per-syllable scores
        """
        # Compute mel
        mel = self._compute_mel(audio)
        mel_tensor = torch.from_numpy(mel).unsqueeze(0)

        # Get frame log-probs
        with torch.no_grad():
            logits = self.model(mel_tensor, audio_mask=None)
            log_probs = F.log_softmax(logits[0], dim=-1)

        # Encode expected syllables
        target_ids = [self.vocab.encode(s) for s in expected_pinyin]

        # Method 1: Full forward-backward
        alpha, beta, total_prob = ctc_forward_backward(
            log_probs, target_ids, blank_idx=0
        )
        scores = compute_syllable_scores(
            log_probs, alpha, beta, total_prob, target_ids
        )

        # Method 2: Constrained decode (simpler, faster)
        # scores = grade_with_constrained_decode(log_probs, target_ids)

        return GradeResult(
            overall=sum(scores) / len(scores) if scores else 0.0,
            syllable_scores=scores,
            syllables=expected_pinyin,
        )
```

---

## 6. Flutter Integration

### 6.1 ONNX Export

```python
def export_v7_to_onnx(model: SyllablePredictorV7, output_path: str):
    """
    Export V7 CTC model to ONNX format.
    """
    model.eval()

    # Dummy inputs
    batch_size = 1
    max_frames = 1000
    mel = torch.randn(batch_size, 80, max_frames)
    audio_mask = torch.zeros(batch_size, max_frames, dtype=torch.bool)

    # Export
    torch.onnx.export(
        model,
        (mel, audio_mask),
        output_path,
        input_names=['mel', 'audio_mask'],
        output_names=['log_probs'],
        dynamic_axes={
            'mel': {2: 'time'},
            'audio_mask': {1: 'time'},
            'log_probs': {1: 'time'},
        },
        opset_version=17,
    )

    # Quantize
    from onnxruntime.quantization import quantize_dynamic
    quantize_dynamic(
        output_path,
        output_path.replace('.onnx', '_int8.onnx'),
        weight_type=QuantType.QInt8,
    )
```

### 6.2 Dart Interface

```dart
/// ONNX-based ML scorer V7 with CTC grading.
class OnnxMlScorerV7 implements MlScorer {
  OrtSession? _session;
  SyllableVocab? _vocab;
  final MelExtractor _melExtractor = MelExtractor();

  static const _modelPath = 'assets/models/model_v7.onnx';
  static const _maxFrames = 1000;

  @override
  Future<Grade> score(TextSequence sequence, Recording recording) async {
    // 1. Load audio and compute mel
    final audioSamples = await _loadAudio(recording.filePath);
    final mel = _melExtractor.extract(audioSamples);

    // 2. Run ONNX inference to get log_probs
    final logProbs = await _runInference(mel);

    // 3. Parse expected syllables
    final syllables = _parsePinyin(sequence.romanization ?? '');
    final targetIds = syllables.map((s) => _vocab!.encode(s)).toList();

    // 4. Compute per-syllable scores using constrained decode
    final scores = _gradeWithConstrainedDecode(logProbs, targetIds);

    // 5. Build Grade
    return Grade(
      overall: (scores.average * 100).round(),
      method: 'onnx_v7_ctc',
      characterScores: _mapToCharacters(scores, sequence.text.length),
      details: {
        'syllableScores': scores,
        'syllables': syllables,
      },
    );
  }

  /// Run ONNX inference to get frame-level log probabilities.
  Future<List<List<double>>> _runInference(List<List<double>> mel) async {
    // Prepare mel tensor [1, 80, T]
    final melFlat = Float32List(80 * _maxFrames);
    // ... fill tensor ...

    final outputs = await _session!.runAsync(
      OrtRunOptions(),
      {'mel': melTensor, 'audio_mask': maskTensor},
      ['log_probs'],
    );

    // Parse output [1, T', V] -> List<List<double>>
    final logProbsRaw = outputs[0]!.value as List;
    return _parseLogProbs(logProbsRaw);
  }

  /// Grade using constrained greedy decoding.
  List<double> _gradeWithConstrainedDecode(
    List<List<double>> logProbs,
    List<int> targetIds,
  ) {
    final scores = <double>[];
    var framePtr = 0;

    for (var k = 0; k < targetIds.length; k++) {
      final sylId = targetIds[k];
      final syllableFrames = <double>[];

      while (framePtr < logProbs.length) {
        final probs = _softmax(logProbs[framePtr]);
        final sylProb = probs[sylId];
        final blankProb = probs[0];

        // Check if next syllable is more likely
        final nextSylId = k + 1 < targetIds.length ? targetIds[k + 1] : null;
        final nextProb = nextSylId != null ? probs[nextSylId] : 0.0;

        if (sylProb > nextProb || nextSylId == null) {
          syllableFrames.add(sylProb);
          framePtr++;

          if (blankProb > sylProb && syllableFrames.length > 3) {
            break;
          }
        } else {
          break;
        }
      }

      // Average probability for this syllable
      final avgProb = syllableFrames.isEmpty
          ? 0.0
          : syllableFrames.reduce((a, b) => a + b) / syllableFrames.length;
      scores.add(avgProb);
    }

    return scores;
  }
}
```

### 6.3 File Structure

```
apps/mobile_flutter/
├── lib/features/ml_inference/
│   ├── data/
│   │   ├── onnx_ml_scorer_v7.dart      # NEW: CTC scorer
│   │   ├── ctc_decoder.dart            # NEW: Decoding utilities
│   │   └── ...
│   └── domain/
│       └── ml_scorer.dart              # Interface (unchanged)
├── assets/models/
│   ├── model_v7.onnx                   # NEW: CTC model
│   └── model_v7_int8.onnx              # NEW: Quantized
```

---

## 7. Testing & Benchmarking

### 7.1 Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| WER | Word Error Rate on decoded sequence | <15% |
| Per-syllable accuracy | Correct syllable / total | >85% |
| Position-independent accuracy | Same across all positions | Variance <5% |
| Pulled recording accuracy | On real recordings | >60% |
| Inference latency | 10s audio on mobile | <300ms |

### 7.2 Test Sets

| Test Set | Size | Purpose |
|----------|------|---------|
| AISHELL3 val | 12,653 | Training validation |
| TTS Female | 60 sentences | Domain transfer |
| TTS Male | 60 sentences | Voice diversity |
| Pulled recordings | ~20 files | Real-world test |

### 7.3 Benchmark Script

```python
def benchmark_v7(model, test_sets):
    """
    Run V7 benchmarks on all test sets.
    """
    results = {}

    for name, dataset in test_sets.items():
        correct = 0
        total = 0
        position_acc = defaultdict(lambda: {'correct': 0, 'total': 0})

        for sample in dataset:
            # Grade sample
            scores = model.grade(sample['audio'], sample['syllables'])

            # Compare to expected
            for i, (score, expected) in enumerate(zip(scores, sample['syllables'])):
                total += 1
                position_acc[i]['total'] += 1

                if score > 0.5:  # Threshold for "correct"
                    correct += 1
                    position_acc[i]['correct'] += 1

        results[name] = {
            'accuracy': correct / total,
            'position_accuracy': {
                k: v['correct'] / v['total']
                for k, v in position_acc.items()
            }
        }

    return results
```

---

## 8. Migration Plan

### 8.1 Phases

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| 1. Model Training | 1 week | Trained V7 model, validation metrics |
| 2. Grading Pipeline | 3 days | Python grading implementation |
| 3. Benchmarking | 2 days | Comparison with V6 on all test sets |
| 4. ONNX Export | 1 day | Quantized ONNX model |
| 5. Flutter Port | 3 days | Dart scorer implementation |
| 6. Integration | 2 days | App integration, testing |

### 8.2 Risk Mitigation

| Risk | Mitigation |
|------|------------|
| CTC training unstable | Use pre-trained encoder weights from V6 |
| Grading accuracy lower than V6 | Fall back to V6, investigate failure modes |
| Inference too slow | Optimize decoding, use simpler algorithm |
| Model too large | More aggressive quantization, pruning |

### 8.3 Rollback Plan

Keep V6 as fallback:

```dart
// In scorer factory
MlScorer createScorer(ScorerVersion version) {
  switch (version) {
    case ScorerVersion.v6:
      return OnnxMlScorerV6();
    case ScorerVersion.v7:
      return OnnxMlScorerV7();
  }
}

// A/B test or gradual rollout
final scorer = config.useV7 ? createScorer(v7) : createScorer(v6);
```

---

## Appendix A: CTC Loss Details

CTC loss handles the alignment problem by summing over all valid alignments:

```
P(target | audio) = Σ P(alignment) for all alignments that collapse to target

Example:
target = [ni, hao]
valid alignments:
  [ni, ni, -, -, hao, hao, -]
  [ni, -, -, hao, -, -, -]
  [-, ni, ni, -, hao, -, -]
  ... many more ...
```

The forward-backward algorithm computes this efficiently in O(T × S) time.

---

## Appendix B: Vocabulary

V7 uses toneless syllables (530 classes):

```
[blank, a, ai, an, ang, ao, ba, bai, ban, bang, ..., zuo]
```

Tones are handled separately via tone classifier head (optional, same as V6).

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-18 | Claude | Initial draft |
