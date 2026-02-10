# Minimal Mandarin Scoring Implementation

**Philosophy:** Deterministic first, DL only if benchmarks demand it.

**Language:** Python (port to Dart what works)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Syllable Segmentation](#2-syllable-segmentation)
3. [Tone Classification](#3-tone-classification)
4. [Scoring Pipeline](#4-scoring-pipeline)
5. [Synthetic Data Generation](#5-synthetic-data-generation)
6. [Benchmarks](#6-benchmarks)
7. [Implementation Plan](#7-implementation-plan)

---

## 1. Overview

### 1.1 Problem Statement

Given:
- Learner audio recording (16kHz mono)
- Target sentence with pinyin (e.g., "nǐ hǎo" / 你好)

Produce:
- Per-syllable tone scores (0-100)
- Overall sentence score

### 1.2 Two Core Subproblems

| Subproblem | DL Approach | Deterministic Alternatives |
|------------|-------------|---------------------------|
| **Syllable segmentation** | CTC forced alignment | DTW with TTS reference, energy-based, vowel detection |
| **Tone classification** | Neural classifier | Rule-based contour analysis, template matching |

### 1.3 Decision Criteria

**Use deterministic if:**
- Accuracy within 10% of DL baseline
- Latency < 200ms on target device
- No external model dependencies

**Consider DL if:**
- Deterministic accuracy < 70% on benchmark
- Rule complexity becomes unmaintainable
- Edge cases dominate failure modes

---

## 2. Syllable Segmentation

### 2.1 Option A: DTW with TTS Reference (Recommended)

**Idea:** We have TTS audio for every sentence. Use Dynamic Time Warping to align learner audio to TTS, then transfer known syllable boundaries.

**Advantages:**
- TTS has known timing (can be pre-computed or derived from synthesis)
- DTW is well-understood, fast, no training needed
- Works well when learner is roughly following the reference

**Algorithm:**

```
1. Extract features from both audios:
   - Mel spectrogram (80 bands) or MFCCs (13 coefficients)
   - 10ms frame shift

2. Compute DTW alignment:
   - Cost matrix: cosine distance between frame features
   - Sakoe-Chiba band constraint (e.g., 20% of utterance length)
   - Recover optimal warping path

3. Map reference boundaries to learner:
   - Reference syllable boundaries are known (from TTS metadata or uniform split)
   - Project through warping path to get learner frame indices
   - Convert frames to milliseconds
```

**Complexity:** O(T₁ × T₂) where T = number of frames. With band constraint, O(T × band_width).

**Libraries:** `fastdtw`, `dtw-python`, or pure numpy implementation.

### 2.2 Option B: Energy-Based Segmentation

**Idea:** Syllables are separated by energy dips (brief silences or reduced amplitude).

**Algorithm:**

```
1. Compute frame-level energy:
   - RMS or log-energy per 10ms frame

2. Smooth energy contour:
   - Moving average (50ms window)

3. Find syllable nuclei:
   - Local maxima above threshold
   - Minimum distance between peaks (100ms)

4. Find boundaries:
   - Local minima between adjacent nuclei
   - Or fixed percentage points (e.g., 20% into inter-peak gap)
```

**Limitations:**
- Fails on connected speech without clear pauses
- Sensitive to noise and speaking style
- May over/under-segment

### 2.3 Option C: Pitch-Based Segmentation

**Idea:** Syllable boundaries often coincide with pitch discontinuities or voicing gaps.

**Algorithm:**

```
1. Extract F0 contour (pYIN)
2. Find voicing gaps (unvoiced regions > 30ms)
3. Find pitch jumps (|Δf0| > 3 semitones between frames)
4. Combine with energy dips
```

**Limitations:**
- Mandarin has many connected syllables without pitch jumps
- Requires robust pitch tracking

### 2.4 Recommendation

**Start with Option A (DTW with TTS reference)** because:
1. We already have TTS audio for every sentence
2. Leverages known-good timing from synthesis
3. Most robust to learner variation
4. Can benchmark against manual annotations

**Fallback:** If DTW fails (>20% boundary error), try energy-based as supplement.

---

## 3. Tone Classification

### 3.1 Mandarin Tone Characteristics

| Tone | Name | Pitch Pattern | Key Features |
|------|------|---------------|--------------|
| 1 | High level | ˉ (55) | Flat, high mean |
| 2 | Rising | ˊ (35) | Positive slope, low start |
| 3 | Dipping | ˇ (214) | V-shape, minimum in middle third |
| 4 | Falling | ˋ (51) | Negative slope, high start, low end |
| 0 | Neutral | - | Short, reduced range |

### 3.2 Option A: Rule-Based Classification (Recommended)

**Features to extract from normalized F0 contour:**

```python
@dataclass
class ToneFeatures:
    slope: float          # Linear regression slope
    range_st: float       # Max - min in semitones
    start_level: float    # Mean of first 20%
    end_level: float      # Mean of last 20%
    min_position: float   # Position of minimum (0-1)
    max_position: float   # Position of maximum (0-1)
    duration_ms: int      # Syllable duration
    voicing_ratio: float  # Fraction of voiced frames
```

**Classification rules:**

```python
def classify_tone_rule_based(features: ToneFeatures) -> tuple[int, float]:
    """
    Returns (predicted_tone, confidence).
    """
    # Neutral tone: very short or low voicing
    if features.duration_ms < 80 or features.voicing_ratio < 0.3:
        return (0, 0.7)

    # Tone 1: Flat and high
    if abs(features.slope) < 0.5 and features.range_st < 3.0:
        return (1, 0.8)

    # Tone 2: Rising
    if features.slope > 1.0 and features.end_level > features.start_level + 2.0:
        return (2, 0.8)

    # Tone 3: Dipping (V-shape)
    if 0.3 < features.min_position < 0.7 and features.range_st > 3.0:
        # Check for V-shape: start high, dip, end mid
        if features.start_level > features.min_level and features.end_level > features.min_level:
            return (3, 0.8)

    # Tone 4: Falling
    if features.slope < -1.0 and features.start_level > features.end_level + 2.0:
        return (4, 0.8)

    # Ambiguous - return most likely based on slope
    if features.slope > 0.5:
        return (2, 0.5)
    elif features.slope < -0.5:
        return (4, 0.5)
    else:
        return (1, 0.4)
```

### 3.3 Option B: Template Matching with DTW

**Idea:** Compare learner contour to canonical tone templates using DTW distance.

**Algorithm:**

```python
# Pre-computed templates (K=20 points, normalized)
TONE_TEMPLATES = {
    1: [0.8, 0.8, 0.8, 0.8, 0.8, ...],  # Flat high
    2: [-0.5, -0.3, 0.0, 0.3, 0.6, ...],  # Rising
    3: [0.2, -0.3, -0.8, -0.5, 0.0, ...],  # Dipping
    4: [0.8, 0.5, 0.0, -0.4, -0.8, ...],  # Falling
    0: [0.0, 0.0, 0.0, 0.0, 0.0, ...],  # Neutral (flat mid)
}

def classify_tone_template(contour: np.ndarray) -> tuple[int, float]:
    """
    Returns (predicted_tone, confidence).
    """
    distances = {}
    for tone, template in TONE_TEMPLATES.items():
        dist, _ = fastdtw(contour, template)
        distances[tone] = dist

    # Softmax over negative distances
    best_tone = min(distances, key=distances.get)

    # Confidence based on margin
    sorted_dists = sorted(distances.values())
    margin = sorted_dists[1] - sorted_dists[0]
    confidence = min(1.0, margin / 2.0)

    return (best_tone, confidence)
```

### 3.4 Option C: Simple Statistical Classifier

**Idea:** Train a logistic regression or decision tree on contour features.

```python
# Features: [slope, range, start_level, end_level, min_pos, max_pos]
# Labels: tone (0-4)

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_train, y_train)  # Train on TTS-extracted features
```

**Advantages:**
- Still interpretable
- No deep learning
- Can export coefficients for Dart implementation

**When to use:** If rule-based accuracy < 75%, try this before neural networks.

### 3.5 Recommendation

**Start with Option A (Rule-based)** because:
1. Most interpretable
2. No training data needed
3. Easy to debug and tune
4. Fast to implement

**Evaluate with:** TTS-extracted contours (known tones) as ground truth.

**Upgrade path:**
- If rule-based < 75% accuracy → try template matching (Option B)
- If template matching < 80% accuracy → try logistic regression (Option C)
- If statistical < 85% accuracy → consider neural classifier

---

## 4. Scoring Pipeline

### 4.1 End-to-End Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Learner Audio  │────▶│   Segmentation  │────▶│ Syllable Spans  │
└─────────────────┘     │   (DTW + TTS)   │     └────────┬────────┘
                        └─────────────────┘              │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Target Pinyin  │────▶│  Sandhi Rules   │────▶│ Surface Tones   │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
         ┌───────────────────────────────────────────────┘
         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Extract F0 per │────▶│  Classify Tone  │────▶│ Compare to      │
│  Syllable       │     │  (Rule-based)   │     │ Target Tone     │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │  Syllable Score │
                                                │    (0-100)      │
                                                └─────────────────┘
```

### 4.2 Scoring Formula

```python
def score_syllable(
    predicted_tone: int,
    target_tone: int,
    confidence: float,
    contour_distance: float,  # DTW distance to target template
) -> float:
    """
    Score a single syllable's tone production.

    Returns: Score 0-100
    """
    # Base score: did they get the tone right?
    if predicted_tone == target_tone:
        base_score = 80 + 20 * confidence
    else:
        # Partial credit for similar tones
        confusion_penalty = CONFUSION_MATRIX[target_tone][predicted_tone]
        base_score = 40 * (1 - confusion_penalty)

    # Contour quality bonus/penalty
    contour_quality = max(0, 1 - contour_distance / 4.0)

    # Final score
    score = 0.7 * base_score + 0.3 * (contour_quality * 100)

    return round(score)

# Tone confusion matrix (lower = more similar, less penalty)
CONFUSION_MATRIX = {
    1: {1: 0.0, 2: 0.6, 3: 0.5, 4: 0.4, 0: 0.3},
    2: {1: 0.6, 2: 0.0, 3: 0.7, 4: 0.5, 0: 0.4},
    3: {1: 0.5, 2: 0.7, 3: 0.0, 4: 0.6, 0: 0.5},
    4: {1: 0.4, 2: 0.5, 3: 0.6, 4: 0.0, 0: 0.3},
    0: {1: 0.3, 2: 0.4, 3: 0.5, 4: 0.3, 0: 0.0},
}
```

### 4.3 Sentence Score

```python
def score_sentence(syllable_scores: list[float]) -> float:
    """
    Aggregate syllable scores to sentence score.
    """
    if not syllable_scores:
        return 0.0

    # Weighted average: penalize worst syllables more
    sorted_scores = sorted(syllable_scores)
    n = len(sorted_scores)

    # Bottom 30% weighted 1.5x
    bottom_idx = max(1, int(n * 0.3))
    bottom_weight = 1.5
    normal_weight = 1.0

    weighted_sum = sum(s * bottom_weight for s in sorted_scores[:bottom_idx])
    weighted_sum += sum(s * normal_weight for s in sorted_scores[bottom_idx:])
    total_weight = bottom_idx * bottom_weight + (n - bottom_idx) * normal_weight

    return weighted_sum / total_weight
```

---

## 5. Synthetic Data Generation

### 5.1 Problem with Real TTS Audio

Testing on real TTS audio revealed fundamental issues:
- **No ground-truth boundaries**: TTS doesn't provide syllable timing
- **Energy-based segmentation fails**: Smooth transitions, no clear dips
- **Voicing-based segmentation fails**: Unvoiced consonants create false boundaries
- **Cannot separate alignment vs classification errors**

### 5.2 Solution: Concatenated Syllable Synthesis

**Idea:** Build sentences by concatenating individual syllable recordings.

```
Sentence: "nǐ hǎo" (你好)

Instead of: [TTS generates full sentence]

We do:     [ni3.wav] + [hao3.wav] = synthetic sentence
           |_0-200ms_|_200-450ms_|
                     ^
                     Known boundary!
```

**Advantages:**
- **Perfect boundary labels**: We know exactly where we concatenated
- **Perfect tone labels**: We know which syllable file we used
- **Separable testing**: Test alignment and classification independently
- **Augmentation-ready**: Can vary speed per syllable

### 5.3 Syllable Lexicon

Create a lexicon of individual syllable recordings covering:

```
mandarin_grader/data/syllables/
├── lexicon.json          # Metadata for all syllables
├── female/
│   ├── a1.wav           # ā (tone 1)
│   ├── a2.wav           # á (tone 2)
│   ├── a3.wav           # ǎ (tone 3)
│   ├── a4.wav           # à (tone 4)
│   ├── ai1.wav
│   ├── ...
│   ├── ni3.wav
│   ├── hao3.wav
│   └── ...
└── male/
    └── ...
```

**Coverage:**
- All finals (a, ai, an, ang, ao, e, ei, en, eng, ...) × 4 tones
- All initial+final combinations used in target sentences
- Neutral tone variants for common particles

**Generation:**
1. Use existing TTS API to generate individual syllables
2. Trim silence from start/end
3. Normalize volume
4. Store with metadata (duration, tone, pinyin)

### 5.4 Sentence Synthesis

```python
def synthesize_sentence(
    syllables: list[str],  # ["ni3", "hao3"]
    lexicon: SyllableLexicon,
    speed_variation: float = 0.1,  # ±10% per syllable
) -> tuple[np.ndarray, list[SyllableSpan]]:
    """
    Concatenate syllables to form a sentence with known boundaries.

    Returns:
        audio: Concatenated audio samples
        spans: Ground-truth syllable boundaries
    """
    audio_parts = []
    spans = []
    current_ms = 0

    for i, syl in enumerate(syllables):
        # Load syllable audio
        syl_audio = lexicon.load(syl)

        # Optional: vary speed slightly
        if speed_variation > 0:
            factor = 1.0 + np.random.uniform(-speed_variation, speed_variation)
            syl_audio = change_speed(syl_audio, factor)

        # Record span
        duration_ms = len(syl_audio) / 16  # assuming 16kHz
        spans.append(SyllableSpan(
            index=i,
            start_ms=current_ms,
            end_ms=current_ms + duration_ms,
            confidence=1.0,  # Ground truth!
        ))

        audio_parts.append(syl_audio)
        current_ms += duration_ms

    # Optional: add small gaps between syllables
    # Optional: add crossfade for smoother transitions

    return np.concatenate(audio_parts), spans
```

### 5.5 Augmentations

| Augmentation | Description | Purpose |
|--------------|-------------|---------|
| Speed variation | ±10-20% per syllable | Simulate natural tempo variation |
| Pitch shift | ±1-2 semitones globally | Simulate different speakers |
| Noise addition | Low-level background noise | Robustness testing |
| Silence gaps | 0-50ms between syllables | Simulate pauses |
| Crossfade | 10-20ms overlap | Smoother transitions |

### 5.6 Dataset Structure

```python
@dataclass
class SyntheticSample:
    id: str
    audio: np.ndarray
    syllables: list[TargetSyllable]
    ground_truth_spans: list[SyllableSpan]  # Known boundaries
    augmentations: dict  # What was applied
```

**Size:**
- 60 sentences from existing dataset
- × 5 augmentation variants = 300 samples
- ~900 syllables with perfect labels

---

## 6. Benchmarks

### 6.1 Benchmark Dataset

**Source:** Synthetic concatenated syllables (Section 5)

**Ground truth available:**
- Exact syllable boundaries (from concatenation)
- Exact tones (from syllable filenames)

**Size:**
- 300 synthetic samples (60 sentences × 5 variants)
- ~900 syllables with perfect labels

**Splits:**
- Tune: 70% (for threshold adjustment)
- Eval: 30% (held out for final metrics)

### 6.2 Metrics

#### Segmentation Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| Boundary error | Mean |predicted - actual| in ms | < 50ms |
| Segment IoU | Intersection over union of spans | > 0.8 |
| Syllable count accuracy | Correct syllable count / total | > 95% |

#### Tone Classification Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| Accuracy | Correct tone / total syllables | > 80% |
| Accuracy (excluding T0) | Accuracy on tones 1-4 only | > 85% |
| T2/T3 confusion rate | T2 misclassified as T3 or vice versa | < 15% |
| Per-tone F1 | F1 score per tone class | > 0.75 |

### 6.3 DL Justification Threshold

**If deterministic approach achieves:**
- Segmentation boundary error < 50ms AND
- Tone accuracy > 80%

**Then:** No DL needed for MVP. Proceed to Dart port.

**If not:** Evaluate specific failure modes:
- Segmentation failures → Consider CTC alignment
- Tone classification failures → Consider neural classifier
- Both → Prioritize the worse performer

### 6.4 Benchmark Protocol

```python
def run_benchmark(scorer, dataset, split="eval"):
    """
    Run full benchmark on held-out data.
    """
    results = {
        "segmentation": {"boundary_errors": [], "ious": []},
        "tone": {"predictions": [], "targets": []},
    }

    for sample in dataset.filter(split):
        # Run scorer
        result = scorer.score(sample.audio, sample.targets)

        # Collect segmentation metrics
        for pred_span, true_span in zip(result.spans, sample.true_spans):
            boundary_error = abs(pred_span.start_ms - true_span.start_ms)
            results["segmentation"]["boundary_errors"].append(boundary_error)
            # ... compute IoU

        # Collect tone metrics
        for pred_tone, true_tone in zip(result.predicted_tones, sample.true_tones):
            results["tone"]["predictions"].append(pred_tone)
            results["tone"]["targets"].append(true_tone)

    # Compute summary metrics
    return compute_metrics(results)
```

---

## 7. Implementation Plan

### 7.1 Module Structure

```
mandarin_grader/
├── __init__.py
├── types.py           # ✓ Done
├── sandhi.py          # ✓ Done
├── pitch.py           # ✓ Done (add YIN alternative to pYIN)
├── contour.py         # ✓ Done
├── data/              # ✓ Done
│   ├── dataloader.py
│   └── audio.py
├── align/             # NEW
│   ├── __init__.py
│   ├── dtw.py         # DTW alignment with TTS reference
│   ├── energy.py      # Energy-based segmentation (fallback)
│   └── types.py       # AlignmentResult, etc.
├── tone/              # NEW
│   ├── __init__.py
│   ├── features.py    # ToneFeatures extraction
│   ├── rules.py       # Rule-based classifier
│   ├── templates.py   # Template matching classifier
│   └── templates.json # Canonical tone templates
├── scorer.py          # ✓ Exists (upgrade to use align + tone)
└── benchmark/         # NEW
    ├── __init__.py
    ├── metrics.py     # Metric computation
    └── runner.py      # Benchmark runner
```

### 7.2 Implementation Phases

#### Phase 0: Synthetic Data Generation (Priority!)

1. **Create syllable lexicon:**
   - Generate individual syllable audio via TTS
   - Cover all syllables used in target sentences
   - All 4 tones + neutral for each

2. **Implement concatenation:**
   - `data/synthesis.py`: SyllableLexicon class
   - Concatenate syllables with known boundaries
   - Add speed/pitch augmentations

3. **Generate benchmark dataset:**
   - Synthesize all 60 sentences
   - Create 5 variants each with augmentations
   - Save with ground-truth labels

**Output:** `data/synthetic/` with audio + labels

#### Phase A: DTW Alignment

1. Implement `align/dtw.py`:
   - MFCC or mel feature extraction
   - DTW with Sakoe-Chiba constraint
   - Boundary projection

2. Test on TTS pairs (same sentence, different voice)

3. Benchmark: boundary error on held-out TTS

#### Phase B: Rule-Based Tone Classifier (1-2 days)

1. Implement `tone/features.py`:
   - Extract ToneFeatures from contour

2. Implement `tone/rules.py`:
   - Rule-based classify_tone()
   - Tune thresholds on TTS data

3. Benchmark: accuracy on TTS-extracted contours

#### Phase C: Integration (1 day)

1. Update `scorer.py`:
   - Wire together alignment + tone scoring
   - Implement scoring formula

2. End-to-end test on TTS data

#### Phase D: Benchmark & Iterate (1-2 days)

1. Run full benchmark suite
2. Identify failure modes
3. Tune rules/thresholds
4. Document results

#### Phase E: (Conditional) Add DL

Only if benchmarks show:
- Segmentation boundary error > 50ms: Add CTC option
- Tone accuracy < 80%: Add neural classifier option

### 7.3 Deliverables

| Deliverable | Description | Priority |
|-------------|-------------|----------|
| `data/syllables/` | Syllable lexicon (TTS-generated) | P0 |
| `data/synthesis.py` | Sentence synthesis from syllables | P0 |
| `data/synthetic/` | Benchmark dataset with labels | P0 |
| `align/dtw.py` | DTW-based alignment | P0 |
| `tone/rules.py` | Rule-based tone classifier | P0 |
| `benchmark/` | Benchmark suite | P0 |
| `BENCHMARK_RESULTS.md` | Results documentation | P0 |
| `tone/templates.py` | Template matching (if rules fail) | P1 |
| CTC aligner | Neural alignment (if DTW fails) | P2 |
| Neural tone classifier | (if rules + templates fail) | P2 |

---

## Appendix A: DTW Implementation Notes

### Fast DTW with Sakoe-Chiba Band

```python
import numpy as np
from numba import jit

@jit(nopython=True)
def dtw_sakoe_chiba(cost_matrix: np.ndarray, band_fraction: float = 0.2):
    """
    DTW with Sakoe-Chiba band constraint.

    Args:
        cost_matrix: [T1, T2] pairwise distances
        band_fraction: Fraction of sequence length for band width

    Returns:
        (total_cost, path)
    """
    T1, T2 = cost_matrix.shape
    band_width = int(max(T1, T2) * band_fraction)

    # Initialize DP matrix with infinity
    dp = np.full((T1 + 1, T2 + 1), np.inf)
    dp[0, 0] = 0

    # Fill DP matrix within band
    for i in range(1, T1 + 1):
        j_min = max(1, i - band_width)
        j_max = min(T2, i + band_width)
        for j in range(j_min, j_max + 1):
            cost = cost_matrix[i-1, j-1]
            dp[i, j] = cost + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])

    # Backtrack to get path
    path = []
    i, j = T1, T2
    while i > 0 or j > 0:
        path.append((i-1, j-1))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            argmin = np.argmin([dp[i-1, j], dp[i, j-1], dp[i-1, j-1]])
            if argmin == 0:
                i -= 1
            elif argmin == 1:
                j -= 1
            else:
                i -= 1
                j -= 1

    return dp[T1, T2], path[::-1]
```

### Feature Extraction for DTW

```python
def extract_dtw_features(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Extract features for DTW alignment.

    Returns: [T, D] feature matrix (MFCCs or mel spectrogram)
    """
    import librosa

    # Option 1: MFCCs (13 coefficients + deltas = 39 dims)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=160)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.vstack([mfcc, delta, delta2]).T  # [T, 39]

    # Option 2: Mel spectrogram (80 dims) - may be better for tonal languages
    # mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80, hop_length=160)
    # features = librosa.power_to_db(mel).T  # [T, 80]

    return features
```

---

## Appendix B: Tone Template Generation

```python
def generate_tone_templates(dataset: SentenceDataset, k: int = 20) -> dict:
    """
    Generate canonical tone templates from TTS data.
    """
    from collections import defaultdict

    contours_by_tone = defaultdict(list)

    for sample in dataset:
        audio = load_audio(sample.audio_path)
        f0, voicing = extract_f0_pyin(audio)
        f0_norm = normalize_f0(hz_to_semitones(f0), voicing)

        # Assuming uniform syllable boundaries for TTS
        syllables = apply_tone_sandhi(sample.syllables)
        n_frames = len(f0_norm)
        frames_per_syl = n_frames // len(syllables)

        for i, syl in enumerate(syllables):
            start = i * frames_per_syl
            end = (i + 1) * frames_per_syl

            syl_contour = resample_contour(
                f0_norm[start:end],
                voicing[start:end],
                k=k
            )

            if np.sum(syl_contour != 0) >= k // 2:  # Mostly voiced
                contours_by_tone[syl.tone_surface].append(syl_contour)

    # Compute mean templates
    templates = {}
    for tone, contours in contours_by_tone.items():
        if len(contours) >= 10:
            templates[tone] = np.mean(contours, axis=0).tolist()

    return templates
```

---

## Appendix C: Rule Tuning Guidelines

### Tone 1 (High Level)
- **slope** threshold: Start at ±0.5, tune based on variance in TTS data
- **range** threshold: Typically < 3 semitones

### Tone 2 (Rising)
- **slope** threshold: Start at > 1.0
- **end - start** difference: > 2 semitones

### Tone 3 (Dipping)
- **min_position**: Between 0.3 and 0.7 (middle third)
- **V-shape check**: start > min AND end > min

### Tone 4 (Falling)
- **slope** threshold: Start at < -1.0
- **start - end** difference: > 2 semitones

### Tone 0 (Neutral)
- **duration**: < 80-100ms
- **voicing_ratio**: < 0.3-0.5

**Tuning process:**
1. Extract features from all TTS syllables
2. Plot feature distributions per tone
3. Find decision boundaries that maximize accuracy
4. Validate on held-out set
