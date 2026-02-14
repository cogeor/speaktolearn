# ML Integration Pipeline

This document details the complete ML integration pipeline from model training to Flutter deployment.

## Overview

```
Training → ONNX Export → Flutter Assets → Recording → Inference → Grading → UI Display
```

| Stage | Status | Location |
|-------|--------|----------|
| 1. Model Training | IMPLEMENTED | `tools/mandarin_grader/scripts/train_v4.py` |
| 2. ONNX Export | IMPLEMENTED | `tools/mandarin_grader/scripts/export_onnx.py` |
| 3. Audio Pipeline | IMPLEMENTED | `apps/mobile_flutter/lib/features/recording/` |
| 4. ML Inference Module | PARTIAL (mock) | `apps/mobile_flutter/lib/features/ml_inference/` |
| 5. Grading Thresholds | IMPLEMENTED | Python + Flutter |
| 6. ColoredText Widget | IMPLEMENTED | `apps/mobile_flutter/lib/features/practice/presentation/widgets/` |
| 7. UI Integration | TODO | Wire ColoredText into home_screen.dart |
| 8. ONNX Runtime Inference | TODO | Replace MockMlScorer with OnnxMlScorer |

---

## 1. Model Training

### Architecture: SyllablePredictorV4

```
Input: mel [batch, 80, time] + pinyin_ids [batch, seq_len]
       ↓
CNN Front-end (4x downsampling)
       ↓
RoPE Transformer (4-6 layers)
       ↓
Attention Pooling (PMA)
       ↓
Dual Heads → syllable_logits [batch, 530] + tone_logits [batch, 5]
```

### Training Command

```bash
cd tools/mandarin_grader
uv run python scripts/train_v4.py \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-4 \
    --checkpoint-dir checkpoints_v4
```

### Key Training Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Sample rate | 16000 Hz | Must match Flutter recording |
| Mel bins | 80 | Standard for speech |
| Hop length | 160 | 10ms frames |
| Win length | 400 | 25ms window |
| d_model | 192-384 | Depends on checkpoint |
| n_layers | 4-6 | Transformer depth |
| n_syllables | 530 | Including PAD, BOS tokens |
| n_tones | 5 | 0-4 |

---

## 2. ONNX Export

### Export Script

```bash
cd tools/mandarin_grader
uv run python scripts/export_onnx.py \
    --checkpoint checkpoints_v4/best_model.pt \
    --output models/syllable_v4.onnx \
    --validate \
    --metadata
```

### Export Process

1. **Load checkpoint** and infer architecture from state_dict shapes
2. **Create model** in eval mode with dropout=0
3. **Export to ONNX** with dynamic axes for batch/sequence dimensions
4. **Validate** PyTorch vs ONNX outputs (15 test samples)
5. **Generate metadata** JSON with preprocessing params

### ONNX Inputs/Outputs

```python
# Inputs
mel: float32[batch, 80, time]        # Log-mel spectrogram
pinyin_ids: int64[batch, seq_len]    # Pinyin context tokens
audio_mask: bool[batch, time]        # True = padded frames
pinyin_mask: bool[batch, seq_len]    # True = padded tokens

# Outputs
syllable_logits: float32[batch, 530] # Apply softmax for probabilities
tone_logits: float32[batch, 5]       # Apply softmax for probabilities
```

### Generated Files

```
models/
├── syllable_v4.onnx           # ONNX model (~2-10MB)
└── syllable_v4_metadata.json  # Preprocessing params
```

### Metadata JSON Structure

```json
{
  "model_info": {
    "name": "SyllablePredictorV4",
    "version": "4.0",
    "precision": "fp32",
    "model_size_mb": 5.2
  },
  "preprocessing": {
    "audio": {
      "sample_rate": 16000,
      "n_mels": 80,
      "hop_length": 160,
      "win_length": 400
    }
  },
  "vocabulary": {
    "n_syllables": 530,
    "n_tones": 5,
    "special_tokens": { "PAD": 0, "BOS": 1 }
  }
}
```

---

## 3. Flutter Audio Pipeline

### Recording Configuration

**File:** `apps/mobile_flutter/lib/features/recording/data/record_plugin_recorder.dart`

```dart
await _recorder.start(
  const RecordConfig(
    encoder: AudioEncoder.wav,      // Uncompressed WAV
    sampleRate: 16000,              // 16kHz (matches training)
    numChannels: 1,                 // Mono
    bitRate: 256000,                // 16-bit PCM
  ),
  path: _currentPath!,
);
```

### Audio Format Requirements

| Property | Value | Notes |
|----------|-------|-------|
| Format | WAV | Uncompressed PCM |
| Sample Rate | 16000 Hz | Must match model training |
| Channels | 1 (mono) | Model expects mono |
| Bit Depth | 16-bit | Standard PCM |

### Validation Script (Python)

```bash
uv run python scripts/validate_flutter_audio.py recording.wav --verbose
```

Validates:
- Sample rate = 16000 Hz
- Channels = 1 (mono)
- Amplitude range [-1, 1]
- Mel-spectrogram shape [80, ~frames]

---

## 4. ML Inference Module

### Current Implementation (Mock)

**File:** `apps/mobile_flutter/lib/features/ml_inference/data/mock_ml_scorer.dart`

```dart
class MockMlScorer implements MlScorer {
  @override
  Future<Grade> score(TextSequence sequence, Recording recording) async {
    // Get character count using proper Unicode handling
    final characters = sequence.text.characters.toList();
    final charCount = characters.length;

    // Generate random scores for each character (0.0-1.0)
    final characterScores = List.generate(
      charCount,
      (_) => _random.nextDouble(),
    );

    // Overall score is average of character scores
    final avgScore = characterScores.reduce((a, b) => a + b) / charCount;

    return Grade(
      overall: (avgScore * 100).round(),
      method: 'mock_ml_v1',
      characterScores: characterScores,
    );
  }
}
```

### TODO: ONNX Runtime Implementation

**Required:** `apps/mobile_flutter/lib/features/ml_inference/data/onnx_ml_scorer.dart`

```dart
class OnnxMlScorer implements MlScorer {
  late final OrtSession _session;
  late final SyllableVocabulary _vocab;
  bool _initialized = false;

  @override
  Future<void> initialize() async {
    // Load ONNX model from assets
    final modelBytes = await rootBundle.load('assets/models/syllable_v4.onnx');
    _session = OrtSession.fromBuffer(modelBytes.buffer.asUint8List());

    // Load vocabulary
    final vocabJson = await rootBundle.loadString('assets/models/syllable_vocab.json');
    _vocab = SyllableVocabulary.fromJson(jsonDecode(vocabJson));

    _initialized = true;
  }

  @override
  Future<Grade> score(TextSequence sequence, Recording recording) async {
    if (!_initialized) await initialize();

    // Run in isolate to avoid blocking UI
    final result = await compute(_scoreInIsolate, _ScoringParams(
      audioPath: recording.filePath,
      pinyinText: sequence.romanization,
      modelBytes: _modelBytes,
      vocab: _vocab,
    ));

    return Grade(
      overall: result.overall,
      method: 'ml_v4',
      characterScores: result.characterScores,
    );
  }
}
```

### Batch Inference (Multiple Syllables per Audio)

The model supports batch inference for scoring multiple syllable positions with a single audio chunk:

```dart
// Score all syllables in "ni hao" with one inference call
final mel = extractMelSpectrogram(audioBytes);  // [1, 80, 100]
final melBatch = mel.repeat(2);  // [2, 80, 100] - same audio repeated

// Different pinyin contexts for each position
final pinyinIds = [
  [BOS],        // Context for first syllable "ni"
  [BOS, ni_id], // Context for second syllable "hao"
];

// Single ONNX inference
final (syllableLogits, toneLogits) = session.run(melBatch, pinyinIds);
// syllableLogits: [2, 530], toneLogits: [2, 5]
```

---

## 5. Grading Thresholds

### Python Definition

**File:** `tools/mandarin_grader/mandarin_grader/model/syllable_predictor_v4.py`

```python
GRADE_THRESHOLDS = {
    'bad': 0.0,
    'almost': 0.2,
    'good': 0.4,
    'easy': 0.6,
}

def probability_to_grade(prob: float) -> str:
    """Map probability to grade.

    prob < 0.2  → 'bad'
    0.2 ≤ prob < 0.4 → 'almost'
    0.4 ≤ prob < 0.6 → 'good'
    prob ≥ 0.6 → 'easy'
    """
    if prob >= GRADE_THRESHOLDS['easy']:
        return 'easy'
    elif prob >= GRADE_THRESHOLDS['good']:
        return 'good'
    elif prob >= GRADE_THRESHOLDS['almost']:
        return 'almost'
    else:
        return 'bad'
```

### Flutter Definition

**File:** `apps/mobile_flutter/lib/features/ml_inference/domain/character_score.dart`

```dart
enum CharacterGrade { bad, almost, good, easy }

extension ProbabilityToGrade on double {
  CharacterGrade toCharacterGrade() {
    if (this >= 0.6) return CharacterGrade.easy;
    if (this >= 0.4) return CharacterGrade.good;
    if (this >= 0.2) return CharacterGrade.almost;
    return CharacterGrade.bad;
  }
}
```

### Grade → Color Mapping

| Grade | Probability | Color | Theme Constant |
|-------|-------------|-------|----------------|
| bad | < 0.2 | Red | `AppTheme.ratingHard` |
| almost | 0.2-0.4 | Yellow | `AppTheme.ratingAlmost` |
| good | 0.4-0.6 | Green | `AppTheme.ratingGood` |
| easy | ≥ 0.6 | Blue | `AppTheme.ratingEasy` |
| (none) | N/A | White | `AppTheme.foreground` |

---

## 6. ColoredText Widget

**File:** `apps/mobile_flutter/lib/features/practice/presentation/widgets/colored_text.dart`

```dart
class ColoredText extends StatelessWidget {
  const ColoredText({
    required this.text,
    this.scores,        // Per-character scores (0.0-1.0)
    this.style,
    this.textAlign = TextAlign.center,
  });

  @override
  Widget build(BuildContext context) {
    // No scores - plain white text
    if (scores == null || scores!.isEmpty) {
      return Text(text, style: style?.copyWith(color: AppTheme.foreground));
    }

    // Build RichText with colored spans
    final characters = text.characters.toList();
    final spans = <TextSpan>[];

    for (int i = 0; i < characters.length; i++) {
      final score = i < scores!.length ? scores![i] : null;
      final color = score != null ? _scoreToColor(score) : AppTheme.foreground;
      spans.add(TextSpan(text: characters[i], style: style?.copyWith(color: color)));
    }

    return RichText(text: TextSpan(children: spans), textAlign: textAlign);
  }

  Color _scoreToColor(double score) {
    if (score >= 0.6) return AppTheme.ratingEasy;   // Blue
    if (score >= 0.4) return AppTheme.ratingGood;   // Green
    if (score >= 0.2) return AppTheme.ratingAlmost; // Yellow
    return AppTheme.ratingHard;                      // Red
  }
}
```

### Usage

```dart
ColoredText(
  text: '你好',
  scores: [0.8, 0.3],  // 你=blue (easy), 好=yellow (almost)
  style: Theme.of(context).textTheme.displayLarge,
)
```

---

## 7. Integration Points

### DI Provider

**File:** `apps/mobile_flutter/lib/app/di.dart`

```dart
/// Provider for ML-based pronunciation scorer.
final pronunciationScorerProvider = Provider<PronunciationScorer>((ref) {
  return MockMlScorer();  // TODO: Replace with OnnxMlScorer
});
```

### Grade Model Extension

**File:** `apps/mobile_flutter/lib/features/scoring/domain/grade.dart`

```dart
@freezed
class Grade with _$Grade {
  const factory Grade({
    required int overall,           // 0-100
    required String method,         // 'ml_v4', 'mock_ml_v1', etc.
    int? accuracy,
    int? completeness,
    String? recognizedText,
    Map<String, dynamic>? details,
    List<double>? characterScores,  // NEW: Per-character scores (0.0-1.0)
  }) = _Grade;
}
```

---

## 8. Remaining Implementation Tasks

### TODO 1: Wire ColoredText into HomeScreen

**File:** `apps/mobile_flutter/lib/features/practice/presentation/home_screen.dart`

Replace line 270-274:

```dart
// Current
Text(
  sequence.text,
  style: Theme.of(context).textTheme.displayLarge,
  textAlign: TextAlign.center,
)

// Should be
ColoredText(
  text: sequence.text,
  scores: currentGrade?.characterScores,  // From scoring result
  style: Theme.of(context).textTheme.displayLarge,
)
```

### TODO 2: Implement OnnxMlScorer

Create `apps/mobile_flutter/lib/features/ml_inference/data/onnx_ml_scorer.dart`:

1. Add `onnxruntime` package to pubspec.yaml
2. Load ONNX model from assets
3. Implement mel-spectrogram extraction in Dart
4. Implement pinyin tokenization
5. Run inference in isolate using `compute()`
6. Map outputs to Grade with characterScores

### TODO 3: Add Flutter Assets

```yaml
# pubspec.yaml
flutter:
  assets:
    - assets/models/syllable_v4.onnx
    - assets/models/syllable_vocab.json
```

### TODO 4: Mel-Spectrogram Extraction in Dart

Implement mel-spectrogram extraction matching Python preprocessing:

```dart
Float32List extractMelSpectrogram(Uint8List wavBytes) {
  // 1. Parse WAV header, extract PCM samples
  // 2. Normalize to [-1, 1]
  // 3. Compute STFT with window=400, hop=160
  // 4. Apply mel filterbank (80 bins)
  // 5. Convert to log scale: log(mel + 1e-9)
  // 6. Return as Float32List [80, num_frames]
}
```

---

## 9. Debug Workflow

### Pull Recordings from Device

```bash
cd tools/mandarin_grader/scripts
python pull_recordings.py --validate
```

### Validate Audio Format

```bash
python validate_flutter_audio.py pulled_recordings/recording.wav --verbose
```

### Test Python Inference

```bash
python -c "
from mandarin_grader.model.syllable_predictor_v4 import SyllablePredictorV4, SyllableVocab
from mandarin_grader.data.audio import load_audio, extract_mel

audio = load_audio('recording.wav')
mel = extract_mel(audio)
vocab = SyllableVocab()

model = SyllablePredictorV4()
# load checkpoint...

pinyin_ids = vocab.encode_sequence(['ni', 'hao'])
output = model.predict(mel, pinyin_ids)

print(f'Syllable: {vocab.decode(output.syllable_pred)} (prob={output.syllable_prob:.3f})')
print(f'Tone: {output.tone_pred} (prob={output.tone_prob:.3f})')
"
```

---

## 10. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FLUTTER APP                                  │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────────┐ │
│  │ HomeScreen  │───▶│ ColoredText  │───▶│ RichText with colored   │ │
│  │             │    │   Widget     │    │ TextSpans per character │ │
│  └─────────────┘    └──────────────┘    └─────────────────────────┘ │
│         │                  ▲                                         │
│         │                  │ characterScores: [0.8, 0.3, ...]       │
│         ▼                  │                                         │
│  ┌─────────────┐    ┌──────────────┐                                │
│  │  Recording  │───▶│  MlScorer    │                                │
│  │  (16kHz WAV)│    │  (Isolate)   │                                │
│  └─────────────┘    └──────────────┘                                │
│                            │                                         │
│                            ▼                                         │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    ONNX RUNTIME                                 │ │
│  │  ┌──────────────┐    ┌──────────────┐    ┌─────────────────┐   │ │
│  │  │ Mel Extract  │───▶│ syllable_v4  │───▶│ Softmax + Grade │   │ │
│  │  │ (Dart)       │    │   .onnx      │    │   Thresholds    │   │ │
│  │  └──────────────┘    └──────────────┘    └─────────────────┘   │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       PYTHON (TRAINING)                              │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────────┐ │
│  │ AISHELL-3   │───▶│ train_v4.py  │───▶│ checkpoints_v4/         │ │
│  │ + TTS Data  │    │              │    │   best_model.pt         │ │
│  └─────────────┘    └──────────────┘    └─────────────────────────┘ │
│                                                     │                │
│                                                     ▼                │
│                                         ┌─────────────────────────┐ │
│                                         │ export_onnx.py          │ │
│                                         │   → syllable_v4.onnx    │ │
│                                         │   → metadata.json       │ │
│                                         └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Summary

| Component | File | Status |
|-----------|------|--------|
| Training Script | `scripts/train_v4.py` | DONE |
| ONNX Export | `scripts/export_onnx.py` | DONE |
| Audio Validation | `scripts/validate_flutter_audio.py` | DONE |
| Recording Pull | `scripts/pull_recordings.py` | DONE |
| Flutter Recording | `record_plugin_recorder.dart` | DONE (16kHz WAV) |
| Grade Model | `grade.dart` | DONE (+characterScores) |
| MlScorer Interface | `ml_scorer.dart` | DONE |
| MockMlScorer | `mock_ml_scorer.dart` | DONE (debugging) |
| ColoredText Widget | `colored_text.dart` | DONE |
| **OnnxMlScorer** | `onnx_ml_scorer.dart` | **TODO** |
| **Mel Extraction (Dart)** | - | **TODO** |
| **UI Integration** | `home_screen.dart` | **TODO** |
| **Flutter Assets** | `pubspec.yaml` | **TODO** |
