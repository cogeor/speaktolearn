# Mandarin Syllable-Level Speech Grading: Implementation Plan

**Version:** 1.0
**Date:** 2026-02-08
**Status:** Draft

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Phase 1: Foundation (Python Reference)](#2-phase-1-foundation-python-reference)
3. [Phase 2: Alignment Pipeline (CTC)](#3-phase-2-alignment-pipeline-ctc)
4. [Phase 3: Tone Scoring](#4-phase-3-tone-scoring)
5. [Phase 4: Segmental Scoring](#5-phase-4-segmental-scoring)
6. [Phase 5: Mobile Integration](#6-phase-5-mobile-integration)
7. [Phase 6: Calibration and Polish](#7-phase-6-calibration-and-polish)
8. [Appendices](#8-appendices)

---

## 1. Executive Summary

### 1.1 Current State vs Target State

| Aspect | Current (ASR+CER) | Target (Syllable-Level) |
|--------|-------------------|-------------------------|
| Feedback granularity | Sentence-level only | Per-syllable scores |
| Tone analysis | Implicit (ASR transcribes) | Explicit tone classification + contour analysis |
| Segmental analysis | None | Initial/final GOP-like scoring |
| Platform dependency | OS speech recognition API | Custom on-device models |
| Error localization | None | Specific error tags per syllable |
| Pitch extraction | None | Frame-level F0 analysis |
| Forced alignment | None | CTC-based syllable boundaries |

### 1.2 High-Level Phase Summary

| Phase | Name | Duration | Key Deliverables | Go/No-Go Criteria |
|-------|------|----------|------------------|-------------------|
| 1 | Foundation | Weeks 1-4 | Python package, types, sandhi, tests | Sandhi tests pass, mypy strict |
| 2 | Alignment Pipeline | Weeks 5-8 | CTC model, G2P lexicon, alignment API | Alignment accuracy > 90% on test set |
| 3 | Tone Scoring | Weeks 9-12 | Pitch extraction, tone classifier, templates | Tone accuracy > 85% on held-out data |
| 4 | Segmental Scoring | Weeks 13-16 | GOP scoring, initial/final analysis | Correlation with human ratings > 0.7 |
| 5 | Mobile Integration | Weeks 17-20 | Flutter FFI, TFLite/CoreML, UI components | End-to-end scoring < 500ms on mid-tier device |
| 6 | Calibration | Weeks 21-24 | Score calibration, golden tests, fairness | 95% golden tests pass, fairness criteria met |

**Total Timeline:** 24 weeks (6 months)

### 1.3 Success Criteria

1. **Accuracy:** Syllable-level scores correlate > 0.75 with expert human ratings
2. **Latency:** Full scoring pipeline completes in < 500ms on mid-tier mobile devices
3. **Model Size:** Total model footprint < 50MB (Android APK / iOS bundle)
4. **Coverage:** Supports all 400+ valid Mandarin syllable-tone combinations
5. **Robustness:** Consistent performance across male/female voices, children, accents

### 1.4 Key Risks and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| CTC model too large for mobile | High | Medium | Use knowledge distillation, aggressive quantization |
| Tone 2/3 confusion rate too high | Medium | High | Fuse classifier with contour distance, use confusion-aware penalties |
| Pitch extraction fails on noisy audio | Medium | Medium | Robust normalization (median/MAD), confidence gating |
| Sandhi rules incomplete | Low | Medium | Extensive test fixtures, iterative rule additions |
| Model inference latency too slow | High | Low | Batched processing, NNAPI/CoreML acceleration |

### 1.5 Dependencies

- Python 3.11+ with numpy, scipy, librosa
- PyTorch for model training (development only)
- TensorFlow Lite for Android deployment
- CoreML for iOS deployment
- Flutter FFI for native bridge
- Mandarin G2P lexicon (CC-CEDICT based)

---

## 2. Phase 1: Foundation (Python Reference)

**Duration:** Weeks 1-4
**Effort:** 4 person-weeks

### 2.1 Objectives

- Establish typed Python package structure
- Implement deterministic scoring logic
- Build comprehensive test framework
- Create test fixture format and initial fixtures

### 2.2 Package Structure

```
mandarin_grader/
  __init__.py
  types.py          # Core data types
  sandhi.py         # Tone sandhi rules
  pitch.py          # Pitch normalization
  contour.py        # Contour extraction
  tone.py           # Tone scoring
  segmental.py      # Segmental scoring
  fluency.py        # Fluency metrics
  fuse.py           # Score fusion
  scorer.py         # End-to-end scorer
  fixtures/
    prompts/        # Target text definitions
    alignments/     # Syllable span data
    f0_tracks/      # Pitch contours (.npy)
    posteriors/     # Phone posteriors (.npy)
    expected_scores/ # Golden outputs
  tests/
    test_sandhi.py
    test_pitch_norm.py
    test_contours.py
    test_tone_scoring.py
    test_segmental_scoring.py
    test_fusion.py
    test_regression_golden.py
```

### 2.3 Core Types (types.py)

```python
from dataclasses import dataclass
from typing import Literal, NewType
import numpy as np

Tone = Literal[0, 1, 2, 3, 4]  # 0 = neutral
Ms = NewType("Ms", int)

@dataclass(frozen=True)
class TargetSyllable:
    index: int
    hanzi: str
    pinyin: str
    initial: str
    final: str
    tone_underlying: Tone
    tone_surface: Tone
    start_expected_ms: Ms | None = None
    end_expected_ms: Ms | None = None

@dataclass(frozen=True)
class PhoneSpan:
    phone: str
    start_ms: Ms
    end_ms: Ms
    confidence: float  # 0..1

@dataclass(frozen=True)
class SyllableSpan:
    index: int
    start_ms: Ms
    end_ms: Ms
    confidence: float  # 0..1
    phone_spans: list[PhoneSpan] | None = None

@dataclass(frozen=True)
class FrameTrack:
    frame_hz: float
    f0_hz: np.ndarray        # shape [T]
    voicing: np.ndarray      # shape [T], 0..1
    energy: np.ndarray | None = None

@dataclass(frozen=True)
class PosteriorTrack:
    frame_hz: float
    token_names: list[str]
    posteriors: np.ndarray   # shape [T, V]

@dataclass(frozen=True)
class Contour:
    f0_norm: np.ndarray      # shape [K]
    df0: np.ndarray          # shape [K]
    ddf0: np.ndarray         # shape [K]
    duration_ms: int
    voicing_ratio: float

@dataclass(frozen=True)
class ToneResult:
    score: float
    probs: dict[Tone, float]
    tags: list[str]

@dataclass(frozen=True)
class SyllableScores:
    segmental: float
    tone: float
    fluency: float
    overall: float
    tone_probs: dict[Tone, float]
    tags: list[str]

@dataclass(frozen=True)
class SentenceScore:
    overall: float  # 0..100
    syllables: list[SyllableScores]
    warnings: list[str]
```

### 2.4 Sandhi Module (sandhi.py)

**Functions:**

| Function | Description |
|----------|-------------|
| `apply_tone_sandhi(targets)` | Main entry point, applies all rules |
| `apply_3rd_tone_sandhi(targets)` | 3+3 -> 2+3 rule |
| `apply_yi_rule(targets)` | "yi" tone changes based on following syllable |
| `apply_bu_rule(targets)` | "bu" tone changes before 4th tone |

**Test Cases (Required):**

```python
def test_3rd_tone_sandhi_pairs():
    # ni3 hao3 -> ni2 hao3
    targets = [syl("ni", 3), syl("hao", 3)]
    result = apply_tone_sandhi(targets)
    assert result[0].tone_surface == 2
    assert result[1].tone_surface == 3

def test_yi_before_4th_tone():
    # yi1 ge4 -> yi2 ge4
    targets = [syl("yi", 1), syl("ge", 4)]
    result = apply_tone_sandhi(targets)
    assert result[0].tone_surface == 2

def test_yi_before_non_4th():
    # yi1 tian1 -> yi4 tian1
    targets = [syl("yi", 1), syl("tian", 1)]
    result = apply_tone_sandhi(targets)
    assert result[0].tone_surface == 4

def test_bu_before_4th_tone():
    # bu4 shi4 -> bu2 shi4
    targets = [syl("bu", 4), syl("shi", 4)]
    result = apply_tone_sandhi(targets)
    assert result[0].tone_surface == 2

def test_neutral_tone_preservation():
    # de0 should remain de0
    targets = [syl("de", 0)]
    result = apply_tone_sandhi(targets)
    assert result[0].tone_surface == 0

def test_chain_3rd_tones():
    # wo3 xiang3 mai3 -> wo2 xiang2 mai3
    targets = [syl("wo", 3), syl("xiang", 3), syl("mai", 3)]
    result = apply_tone_sandhi(targets)
    assert [t.tone_surface for t in result] == [2, 2, 3]
```

### 2.5 Pitch Normalization Module (pitch.py)

**Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `hz_to_semitones` | `(f0_hz: ndarray, ref_hz: float) -> ndarray` | Convert Hz to semitones |
| `normalize_f0` | `(semitones: ndarray, voicing: ndarray) -> ndarray` | Speaker normalization |
| `robust_stats` | `(values: ndarray, voicing: ndarray) -> tuple[float, float]` | Median/MAD statistics |

**Key Invariants:**

1. **Scaling invariance:** Multiplying all voiced F0 by a constant should not change the normalized contour shape (up to numerical tolerance)
2. **Unvoiced safety:** Unvoiced frames (voicing < 0.5) are excluded from statistics and do not produce NaN values
3. **Empty frame handling:** If all frames are unvoiced, return zero-filled array with warning

**Test Cases:**

```python
def test_scaling_invariance():
    f0 = np.array([100, 150, 200, 250, 200])
    voicing = np.ones(5)
    norm1 = normalize_f0(hz_to_semitones(f0, 100), voicing)
    norm2 = normalize_f0(hz_to_semitones(f0 * 2, 100), voicing)
    assert np.allclose(norm1, norm2, atol=0.1)

def test_unvoiced_exclusion():
    f0 = np.array([100, 0, 200, 0, 300])
    voicing = np.array([1, 0, 1, 0, 1])
    result = normalize_f0(hz_to_semitones(f0, 100), voicing)
    assert not np.isnan(result).any()

def test_all_unvoiced():
    f0 = np.zeros(5)
    voicing = np.zeros(5)
    result = normalize_f0(hz_to_semitones(f0, 100), voicing)
    assert np.allclose(result, 0)
```

### 2.6 Unit Test Coverage Targets

| Module | Line Coverage | Branch Coverage |
|--------|---------------|-----------------|
| types.py | 100% | N/A |
| sandhi.py | 100% | 100% |
| pitch.py | 95% | 90% |
| contour.py | 95% | 85% |
| tone.py | 90% | 85% |
| segmental.py | 90% | 85% |
| fuse.py | 95% | 90% |
| scorer.py | 85% | 80% |

### 2.7 Deliverables

1. **mandarin_grader** Python package with all modules
2. **50+ unit tests** passing with mypy --strict
3. **10 initial test fixtures** covering common cases
4. **CI configuration** for automated testing
5. **README** with development setup instructions

---

## 3. Phase 2: Alignment Pipeline (CTC)

**Duration:** Weeks 5-8
**Effort:** 6 person-weeks

### 3.1 Objectives

- Select or train tiny CTC acoustic model
- Build Mandarin G2P lexicon
- Implement forced alignment algorithm
- Quantize model for mobile deployment

### 3.2 Acoustic Model Requirements

| Requirement | Specification |
|-------------|---------------|
| Architecture | Tiny Conformer or TDNN-CTC |
| Input | 80-dimensional log-mel spectrogram |
| Output | Phone-level posteriors (blank + ~60 phones) |
| Parameters | < 5M parameters |
| Size (FP32) | < 20 MB |
| Size (INT8) | < 10 MB |
| Inference | < 100ms RTF on mobile CPU |

**Model Options (evaluate in order of preference):**

1. **wav2vec2-tiny-zh** (if available, fine-tune)
2. **WeNet Conformer-tiny** (distilled from larger model)
3. **Custom TDNN-CTC** (train from scratch on AISHELL)

### 3.3 G2P Lexicon Format

```json
{
  "format_version": "1.0",
  "description": "Mandarin syllable to phone mapping",
  "entries": {
    "shi4": {
      "initial": "sh",
      "final": "iy4",
      "phones": ["sh", "iy4"],
      "variants": [["sh", "ih4"]]
    },
    "de0": {
      "initial": "",
      "final": "ax0",
      "phones": ["ax0"],
      "variants": []
    }
  }
}
```

**Phone Inventory (~60 phones):**

| Category | Phones |
|----------|--------|
| Initials | b, p, m, f, d, t, n, l, g, k, h, j, q, x, zh, ch, sh, r, z, c, s, y, w |
| Finals (base) | a, o, e, i, u, v, ai, ei, ao, ou, an, en, ang, eng, ong, ia, ie, iu, ian, in, iang, ing, iong, ua, uo, uai, ui, uan, un, uang, ueng, ve, van, vn |
| Tone markers | 0, 1, 2, 3, 4 (appended to final) |
| Special | sil (silence), sp (short pause) |

### 3.4 Forced Alignment Algorithm

**Input:**
- Audio waveform (16kHz mono)
- Target syllable sequence with phones

**Process:**

```
1. Feature Extraction
   - Compute 80-dim log-mel spectrogram (25ms window, 10ms hop)
   - Apply CMVN normalization

2. Acoustic Model Forward Pass
   - Run CTC model to get frame-level posteriors [T, V]
   - Store log-posteriors for scoring

3. Viterbi Forced Alignment
   - Build phone-level FST from target sequence
   - Insert blank tokens between phones
   - Allow optional silence at boundaries
   - Find best path using Viterbi

4. Span Extraction
   - Collapse repeated tokens and blanks
   - Extract phone boundaries
   - Aggregate phones into syllable spans
   - Compute per-span confidence from posteriors

5. Missing/Extra Detection
   - If best path score << threshold: mark unreliable
   - If expected phone has posterior << 0.1: mark as missing
   - If unexpected phone has high posterior: mark as insertion
```

**Output Format:**

```python
@dataclass
class AlignmentResult:
    syllable_spans: list[SyllableSpan]
    phone_spans: list[PhoneSpan]
    overall_confidence: float
    warnings: list[str]  # ["low_confidence_span:2", "possible_insertion:5"]
```

### 3.5 Model Quantization Targets

| Format | Size | Latency | Accuracy Loss |
|--------|------|---------|---------------|
| FP32 (baseline) | 20 MB | 150ms | 0% |
| FP16 | 10 MB | 120ms | < 0.5% |
| INT8 (dynamic) | 5 MB | 100ms | < 1% |
| INT8 (static) | 5 MB | 80ms | < 1.5% |

**Quantization Process:**

1. Collect calibration dataset (100 diverse utterances)
2. Apply post-training quantization with TensorFlow Lite
3. Validate alignment accuracy on held-out set
4. Fine-tune if accuracy drops > 2%

### 3.6 Dependencies

- Training data: AISHELL-1, AISHELL-3, or internal dataset
- Training infrastructure: GPU cluster (4x V100 for ~1 week)
- CC-CEDICT for base lexicon
- k2/Lhotse for FST alignment (development)
- TFLite converter for quantization

### 3.7 Deliverables

1. **Trained CTC model** (FP32 and INT8 versions)
2. **G2P lexicon** with 6000+ syllable entries
3. **Forced alignment module** with Python API
4. **Benchmark results** on test set (accuracy, latency)
5. **Calibration dataset** for quantization
6. **Integration tests** with Phase 1 scorer

---

## 4. Phase 3: Tone Scoring

**Duration:** Weeks 9-12
**Effort:** 5 person-weeks

### 4.1 Objectives

- Implement robust pitch extraction
- Train tiny tone classifier
- Build tone contour templates
- Fuse classifier and contour scores

### 4.2 Pitch Extraction Algorithm Options

| Algorithm | Pros | Cons | Mobile Feasible |
|-----------|------|------|-----------------|
| **YIN** | Simple, fast, well-understood | Requires tuning | Yes |
| CREPE | Very accurate | Large model (20MB+) | No (model too large) |
| CREPE-tiny | Good accuracy | Still 5MB+ | Maybe |
| **pYIN** | Robust, probabilistic | Slightly slower | Yes |
| FCPE | Modern, accurate | Needs porting | Maybe |

**Recommended:** YIN with confidence-weighted interpolation

**YIN Implementation Spec:**

```python
def extract_pitch_yin(
    audio: np.ndarray,
    sample_rate: int = 16000,
    frame_shift_ms: float = 10.0,
    min_f0: float = 50.0,
    max_f0: float = 500.0,
    threshold: float = 0.1
) -> FrameTrack:
    """
    Extract frame-level F0 using YIN algorithm.

    Returns:
        FrameTrack with f0_hz (0 for unvoiced) and voicing confidence
    """
```

### 4.3 Contour Feature Extraction

**Per-Syllable Contour Vector (K=20 points):**

```python
@dataclass
class ContourFeatures:
    # Raw contour (normalized F0, K points)
    f0_norm: np.ndarray       # [K]

    # Derivatives
    df0: np.ndarray           # [K] first derivative
    ddf0: np.ndarray          # [K] second derivative

    # Summary statistics
    slope: float              # linear regression slope
    range_st: float           # max - min in semitones
    min_pos: float            # position of minimum (0-1)
    max_pos: float            # position of maximum (0-1)
    duration_ms: int          # syllable duration
    voicing_ratio: float      # fraction of voiced frames

    # Classifier input vector (K*2 + 6 = 46 dims)
    def to_vector(self) -> np.ndarray:
        return np.concatenate([
            self.f0_norm,
            self.df0,
            [self.slope, self.range_st, self.min_pos,
             self.max_pos, self.duration_ms / 500, self.voicing_ratio]
        ])
```

**Resampling Algorithm:**

```python
def resample_contour(
    f0_norm: np.ndarray,
    voicing: np.ndarray,
    k: int = 20
) -> np.ndarray:
    """
    Resample variable-length contour to fixed K points.

    - Only uses voiced frames
    - Linear interpolation
    - Zero-pads if < 3 voiced frames
    """
```

### 4.4 Tone Classifier Architecture

**Model Spec:**

| Layer | Input | Output | Parameters |
|-------|-------|--------|------------|
| Input | 46 | 46 | 0 |
| Dense + ReLU | 46 | 32 | 1,536 |
| Dense + ReLU | 32 | 16 | 544 |
| Dense + Softmax | 16 | 5 | 85 |
| **Total** | | | **2,165** |

**Size:** < 10 KB (FP32), < 3 KB (INT8)

**Training Data Requirements:**

- 10,000+ labeled syllable contours
- Balanced across all 5 tones (including neutral)
- Multiple speakers (male/female, age ranges)
- Various recording conditions

**Training Process:**

1. Extract contours from AISHELL using forced alignment
2. Augment with pitch shifting, time stretching
3. Train with cross-entropy loss, early stopping
4. Evaluate on held-out speakers
5. Target: > 85% accuracy on clean data, > 75% on noisy

### 4.5 Tone Templates

**Template Format:**

```python
TONE_TEMPLATES: dict[Tone, np.ndarray] = {
    1: np.array([...]),  # High level: flat, high
    2: np.array([...]),  # Rising: low to high
    3: np.array([...]),  # Dipping: mid-low-mid
    4: np.array([...]),  # Falling: high to low
    0: np.array([...]),  # Neutral: reduced, context-dependent
}
```

**Template Generation:**

1. Cluster contours by tone from training data
2. Compute mean contour per cluster
3. Smooth with Savitzky-Golay filter
4. Validate by manual inspection

### 4.6 Contour Distance Computation

```python
def tone_contour_distance(
    learner: ContourFeatures,
    target_tone: Tone,
    templates: dict[Tone, np.ndarray],
    method: Literal["l2", "dtw"] = "l2"
) -> float:
    """
    Compute distance between learner contour and target template.

    Returns normalized distance in [0, 1] range.
    """
    if method == "l2":
        template = templates[target_tone]
        dist = np.sqrt(np.mean((learner.f0_norm - template) ** 2))
        # Normalize by typical range (3-4 semitones)
        return min(1.0, dist / 4.0)
    elif method == "dtw":
        # Dynamic time warping (more tolerant of timing)
        dist, _ = fastdtw(learner.f0_norm, templates[target_tone])
        return min(1.0, dist / (20 * 4))
```

### 4.7 Tone Score Fusion

```python
@dataclass
class ToneConfig:
    w_cls: float = 0.6      # Weight for classifier probability
    w_shape: float = 0.4    # Weight for contour shape
    neutral_duration_threshold_ms: int = 100
    confusion_matrix: np.ndarray | None = None

def score_tone(
    contour: ContourFeatures,
    target_tone: Tone,
    classifier: ToneClassifier,
    templates: dict[Tone, np.ndarray],
    config: ToneConfig
) -> ToneResult:
    """
    Fused tone score combining classifier and contour distance.
    """
    # Get classifier probabilities
    probs = classifier.predict_proba(contour.to_vector())
    p_correct = probs[target_tone]

    # Get contour distance
    distance = tone_contour_distance(contour, target_tone, templates)
    shape_score = 1.0 - distance

    # Special handling for neutral tone
    if target_tone == 0:
        # Neutral tone: emphasize duration and relative pitch
        duration_ok = contour.duration_ms < config.neutral_duration_threshold_ms
        shape_score = 0.8 if duration_ok else 0.4

    # Fuse scores
    score = config.w_cls * p_correct + config.w_shape * shape_score

    # Generate tags
    tags = []
    predicted = max(probs, key=probs.get)
    if predicted != target_tone and p_correct < 0.5:
        tags.append(f"tone_{predicted}_vs_{target_tone}_confusion")

    return ToneResult(score=score, probs=probs, tags=tags)
```

### 4.8 Confusion-Aware Penalties

**Tone Confusion Matrix (from literature):**

|  | T1 | T2 | T3 | T4 | T0 |
|--|----|----|----|----|----|
| T1 | - | 0.3 | 0.2 | 0.4 | 0.1 |
| T2 | 0.3 | - | 0.7 | 0.2 | 0.2 |
| T3 | 0.2 | 0.7 | - | 0.1 | 0.3 |
| T4 | 0.4 | 0.2 | 0.1 | - | 0.1 |
| T0 | 0.1 | 0.2 | 0.3 | 0.1 | - |

**Interpretation:** T2/T3 confusion is most common (0.7 weight), so penalty is reduced. T1/T4 confusion is less common (0.4), so penalty is higher.

### 4.9 Deliverables

1. **YIN pitch extractor** integrated with scorer
2. **Trained tone classifier** (TFLite format)
3. **Tone templates** for all 5 tones
4. **Contour distance functions** (L2 and DTW)
5. **Fusion logic** with configurable weights
6. **Evaluation report** on held-out test set

---

## 5. Phase 4: Segmental Scoring

**Duration:** Weeks 13-16
**Effort:** 5 person-weeks

### 5.1 Objectives

- Implement GOP-like posterior scoring
- Score initials and finals separately
- Add duration sanity checks
- Integrate with alignment spans

### 5.2 GOP-Like Posterior Scoring

**Goodness of Pronunciation (GOP) Concept:**

For each phone segment, compute how well the acoustic evidence supports the expected phone vs. competing phones.

```python
def score_phone_gop(
    posteriors: np.ndarray,      # [T_seg, V] posteriors for segment
    expected_idx: int,           # index of expected phone
    config: SegmentalConfig
) -> float:
    """
    Compute GOP-like score for a phone segment.

    GOP = mean(log p(expected)) - mean(log p(best_competitor))

    Returns score in [0, 1] range.
    """
    # Average log posterior of expected phone
    log_p_expected = np.mean(np.log(posteriors[:, expected_idx] + 1e-10))

    # Average log posterior of best competing phone
    mask = np.ones(posteriors.shape[1], dtype=bool)
    mask[expected_idx] = False
    log_p_competitors = np.log(posteriors[:, mask] + 1e-10)
    log_p_best_comp = np.max(np.mean(log_p_competitors, axis=0))

    # GOP score (higher is better)
    gop = log_p_expected - log_p_best_comp

    # Map to [0, 1] using sigmoid-like transformation
    # GOP > 0 means expected phone is more likely
    score = 1.0 / (1.0 + np.exp(-gop / config.gop_scale))

    return score
```

### 5.3 Initial/Final Scoring for Mandarin

**Mandarin Syllable Structure:** (Initial) + Final

| Component | Example | Weight |
|-----------|---------|--------|
| Initial | sh-, zh-, m-, etc. | 0.4 |
| Final | -i, -ang, -uan, etc. | 0.6 |

**Rationale for weights:** Finals carry more phonemic information and tonal cues in Mandarin.

```python
def score_syllable_segmentals(
    posteriors: PosteriorTrack,
    syllable_span: SyllableSpan,
    target: TargetSyllable,
    config: SegmentalConfig
) -> SegmentalResult:
    """
    Score initial and final components of a syllable.
    """
    initial_score = 0.0
    final_score = 0.0
    tags = []

    if syllable_span.phone_spans:
        for phone_span in syllable_span.phone_spans:
            # Extract posterior segment
            seg_posteriors = extract_segment(
                posteriors,
                phone_span.start_ms,
                phone_span.end_ms
            )

            # Determine if initial or final
            expected_phone = phone_span.phone
            phone_idx = posteriors.token_names.index(expected_phone)

            score = score_phone_gop(seg_posteriors, phone_idx, config)

            if is_initial(expected_phone):
                initial_score = score
            else:
                final_score = max(final_score, score)

            # Tag confusions
            if score < config.confusion_threshold:
                competitor = find_top_competitor(seg_posteriors, phone_idx)
                tags.append(f"{expected_phone}_vs_{competitor}")

    # Handle missing initial (e.g., "a1" has no initial)
    if not target.initial:
        initial_score = 1.0  # No initial to score

    # Weighted combination
    if target.initial:
        overall = config.w_initial * initial_score + config.w_final * final_score
    else:
        overall = final_score

    return SegmentalResult(
        initial_score=initial_score,
        final_score=final_score,
        overall=overall,
        tags=tags
    )
```

### 5.4 Duration Sanity Checks

**Duration Constraints:**

| Condition | Action |
|-----------|--------|
| Syllable < 50ms | Degrade confidence by 0.3 |
| Syllable > 800ms | Flag as "hesitation", degrade by 0.1 |
| Phone < 20ms | Mark as "too_short", reduce score |
| Phone > 400ms | Mark as "elongated" |

```python
def apply_duration_penalties(
    score: float,
    span: SyllableSpan,
    config: SegmentalConfig
) -> tuple[float, list[str]]:
    """
    Apply duration-based penalties to segmental score.
    """
    tags = []
    duration = span.end_ms - span.start_ms

    if duration < config.min_syllable_ms:
        score *= 0.7
        tags.append("too_short")
    elif duration > config.max_syllable_ms:
        score *= 0.9
        tags.append("hesitation")

    return score, tags
```

### 5.5 Confidence Gating

```python
def gate_segmental_score(
    score: float,
    alignment_confidence: float,
    config: SegmentalConfig
) -> float:
    """
    Reduce segmental score impact when alignment is uncertain.

    If alignment confidence < threshold:
    - Interpolate score toward neutral (0.5)
    - Add warning tag
    """
    if alignment_confidence < config.confidence_threshold:
        # Linear interpolation toward neutral
        alpha = alignment_confidence / config.confidence_threshold
        score = alpha * score + (1 - alpha) * 0.5

    return score
```

### 5.6 Integration Points

**With Phase 2 (Alignment):**
- Receive `SyllableSpan` with phone-level boundaries
- Access `PosteriorTrack` from alignment model

**With Phase 3 (Tone):**
- Share syllable span information
- Coordinate confidence gating

**With Phase 1 (Types):**
- Use `SegmentalResult` dataclass
- Contribute to `SyllableScores`

### 5.7 Deliverables

1. **GOP scoring module** with configurable parameters
2. **Initial/final separation** logic for Mandarin
3. **Duration penalty functions**
4. **Confidence gating** integration
5. **Unit tests** for monotonicity and boundary stability
6. **Benchmark** correlation with human ratings

---

## 6. Phase 5: Mobile Integration

**Duration:** Weeks 17-20
**Effort:** 6 person-weeks

### 6.1 Objectives

- Build Dart FFI bridge to native inference
- Deploy TFLite models on Android
- Deploy CoreML models on iOS
- Extend Grade model for syllable-level scores
- Create UI feedback components

### 6.2 Dart FFI Architecture

```
Flutter (Dart)
    |
    v
MandarinScorerPlugin (Dart FFI)
    |
    +---> Android: libmandarin_scorer.so (C++)
    |         |
    |         +---> TFLite Runtime
    |         +---> YIN Pitch (native)
    |
    +---> iOS: MandScorer.framework (Swift/ObjC)
              |
              +---> CoreML Runtime
              +---> Accelerate.framework (YIN)
```

### 6.3 TFLite Integration (Android)

**Model Files:**

| Model | File | Size | Purpose |
|-------|------|------|---------|
| Alignment | ctc_aligner_int8.tflite | 5 MB | Forced alignment |
| Tone | tone_clf_int8.tflite | 3 KB | Tone classification |

**Native Library Structure:**

```cpp
// mandarin_scorer.h
extern "C" {
    typedef struct {
        int index;
        float tone_score;
        float segmental_score;
        float overall;
        int predicted_tone;
        float tone_probs[5];
        char* tags;  // JSON array
    } SyllableResult;

    typedef struct {
        float overall;
        int num_syllables;
        SyllableResult* syllables;
        char* warnings;  // JSON array
    } ScoringResult;

    // Initialize with model paths
    int mand_scorer_init(const char* ctc_model_path,
                         const char* tone_model_path);

    // Score audio against target text
    ScoringResult* mand_scorer_score(
        const float* audio_samples,
        int num_samples,
        int sample_rate,
        const char* target_json  // JSON with syllables
    );

    // Free result memory
    void mand_scorer_free_result(ScoringResult* result);

    // Cleanup
    void mand_scorer_destroy();
}
```

### 6.4 CoreML Integration (iOS)

**Model Files:**

| Model | File | Size | Purpose |
|-------|------|------|---------|
| Alignment | CTCAligner.mlmodelc | 6 MB | Forced alignment |
| Tone | ToneClassifier.mlmodelc | 5 KB | Tone classification |

**Swift Interface:**

```swift
// MandarinScorer.swift
public class MandarinScorer {
    private let alignmentModel: MLModel
    private let toneModel: MLModel

    public init(modelPath: URL) throws

    public func score(
        audioURL: URL,
        targetSyllables: [TargetSyllable]
    ) async throws -> ScoringResult

    public struct ScoringResult: Codable {
        let overall: Float
        let syllables: [SyllableResult]
        let warnings: [String]
    }

    public struct SyllableResult: Codable {
        let index: Int
        let toneScore: Float
        let segmentalScore: Float
        let overall: Float
        let predictedTone: Int
        let toneProbs: [Float]
        let tags: [String]
    }
}
```

### 6.5 Extended Grade Model

**Current Grade Model:**

```dart
@freezed
class Grade with _$Grade {
  const factory Grade({
    required int overall,
    required String method,
    int? accuracy,
    int? completeness,
    String? recognizedText,
    Map<String, dynamic>? details,
  }) = _Grade;
}
```

**Extended Grade Model:**

```dart
@freezed
class Grade with _$Grade {
  const factory Grade({
    required int overall,
    required String method,
    int? accuracy,
    int? completeness,
    String? recognizedText,
    Map<String, dynamic>? details,

    // New syllable-level fields
    List<SyllableGrade>? syllables,
    List<String>? warnings,
  }) = _Grade;
}

@freezed
class SyllableGrade with _$SyllableGrade {
  const factory SyllableGrade({
    required int index,
    required String hanzi,
    required String pinyin,
    required double toneScore,
    required double segmentalScore,
    required double overall,
    required int expectedTone,
    required int predictedTone,
    required Map<int, double> toneProbs,
    required List<String> tags,
  }) = _SyllableGrade;
}
```

### 6.6 New MandarinSyllableScorer Implementation

```dart
/// Mandarin syllable-level pronunciation scorer using on-device models.
class MandarinSyllableScorer implements PronunciationScorer {
  MandarinSyllableScorer({
    required MandarinScorerPlugin plugin,
    required SandhiEngine sandhi,
  }) : _plugin = plugin,
       _sandhi = sandhi;

  final MandarinScorerPlugin _plugin;
  final SandhiEngine _sandhi;

  static const _method = 'mandarin_syllable_v1';

  @override
  Future<Grade> score(TextSequence sequence, Recording recording) async {
    // 1. Parse target syllables from sequence
    final targets = _parseTargetSyllables(sequence);

    // 2. Apply sandhi rules
    final sandhiTargets = _sandhi.apply(targets);

    // 3. Call native scorer
    final result = await _plugin.score(
      audioPath: recording.filePath,
      targets: sandhiTargets,
    );

    // 4. Convert to Grade
    return Grade(
      overall: (result.overall * 100).round(),
      method: _method,
      accuracy: _computeAccuracy(result),
      completeness: _computeCompleteness(result),
      syllables: result.syllables.map(_toSyllableGrade).toList(),
      warnings: result.warnings,
      details: {
        'num_syllables': result.syllables.length,
        'avg_tone_score': _avgTone(result),
        'avg_segmental_score': _avgSegmental(result),
      },
    );
  }
}
```

### 6.7 Integration Points with Existing Codebase

| Current File | Integration Action |
|--------------|-------------------|
| `pronunciation_scorer.dart` | No change (interface) |
| `asr_similarity_scorer.dart` | Keep as fallback option |
| `grade.dart` | Add `syllables` and `warnings` fields |
| `score_pronunciation_use_case.dart` | Inject scorer based on config |

**Dependency Injection:**

```dart
// In DI container
final pronunciationScorer = switch (config.scorerType) {
  ScorerType.asrCer => AsrSimilarityScorer(...),
  ScorerType.mandarinSyllable => MandarinSyllableScorer(...),
};
```

### 6.8 UI Feedback Components

**SyllableFeedbackRow Widget:**

```dart
class SyllableFeedbackRow extends StatelessWidget {
  final SyllableGrade grade;

  Color get _toneColor => switch (grade.toneScore) {
    >= 0.8 => Colors.green,
    >= 0.5 => Colors.orange,
    _ => Colors.red,
  };

  IconData get _toneIcon => switch (grade.expectedTone) {
    1 => Icons.horizontal_rule,  // flat
    2 => Icons.arrow_upward,     // rising
    3 => Icons.swap_vert,        // dipping
    4 => Icons.arrow_downward,   // falling
    _ => Icons.remove,           // neutral
  };

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Text(grade.hanzi, style: TextStyle(color: _toneColor)),
        Icon(_toneIcon, size: 16),
        if (grade.tags.isNotEmpty)
          Tooltip(
            message: grade.tags.join(', '),
            child: Icon(Icons.info_outline, size: 14),
          ),
      ],
    );
  }
}
```

**ToneContourOverlay Widget:**

```dart
class ToneContourOverlay extends StatelessWidget {
  final List<double> expectedContour;
  final List<double> learnerContour;

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: ContourPainter(
        expected: expectedContour,
        learner: learnerContour,
      ),
    );
  }
}
```

### 6.9 Deliverables

1. **mandarin_scorer native library** (Android .so, iOS .framework)
2. **MandarinScorerPlugin** Dart FFI wrapper
3. **Extended Grade model** with syllable fields
4. **MandarinSyllableScorer** implementation
5. **UI feedback widgets** (SyllableFeedbackRow, ToneContourOverlay)
6. **Integration tests** end-to-end on device
7. **Performance benchmarks** on target devices

---

## 7. Phase 6: Calibration and Polish

**Duration:** Weeks 21-24
**Effort:** 4 person-weeks

### 7.1 Objectives

- Collect calibration dataset with human ratings
- Fit isotonic regression for score mapping
- Build golden test suite
- Evaluate fairness across demographics
- Optimize performance

### 7.2 Calibration Dataset Requirements

**Dataset Size:** 500-1000 utterances

**Collection Criteria:**

| Criterion | Specification |
|-----------|---------------|
| Speakers | 50+ unique speakers |
| Gender | 50% male, 50% female |
| Age groups | Children (6-12), Adults (18-50), Seniors (50+) |
| Proficiency | Beginner, Intermediate, Advanced, Native |
| Sentence length | 2-10 syllables |
| Recording conditions | Quiet, moderate noise, mobile mic |

**Human Rating Protocol:**

1. Expert raters (3 per utterance) score 0-100
2. Per-syllable tone correctness (1-5 scale)
3. Per-syllable segmental quality (1-5 scale)
4. Overall fluency rating (1-5 scale)
5. Inter-rater agreement target: Krippendorff's alpha > 0.7

### 7.3 Isotonic Regression Fitting

**Purpose:** Map raw model scores to user-facing 0-100 scale that aligns with human perception.

```python
from sklearn.isotonic import IsotonicRegression

def fit_calibration(
    raw_scores: np.ndarray,      # Model outputs [N]
    human_ratings: np.ndarray    # Human ratings [N]
) -> IsotonicRegression:
    """
    Fit isotonic regression to map raw scores to calibrated scores.

    Isotonic ensures monotonicity: higher raw -> higher calibrated
    """
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(raw_scores, human_ratings)
    return ir

# Apply in inference
calibrated_score = calibration_model.predict([raw_score])[0]
```

**Calibration Targets:**

| Raw Score Range | Expected Calibrated Range |
|-----------------|---------------------------|
| 0.0 - 0.3 | 0 - 30 (poor) |
| 0.3 - 0.6 | 30 - 60 (fair) |
| 0.6 - 0.8 | 60 - 80 (good) |
| 0.8 - 1.0 | 80 - 100 (excellent) |

### 7.4 Golden Test Suite Structure

```
golden_tests/
  manifests/
    test_manifest.json
  audio/
    test_001.wav
    test_002.wav
    ...
  expected/
    test_001.json
    test_002.json
    ...
```

**Manifest Format:**

```json
{
  "version": "1.0",
  "tests": [
    {
      "id": "test_001",
      "audio": "audio/test_001.wav",
      "target_text": "你好",
      "expected": "expected/test_001.json",
      "tags": ["basic", "two_syllables", "3rd_sandhi"]
    }
  ]
}
```

**Expected Output Format:**

```json
{
  "overall": 85,
  "overall_tolerance": 5,
  "syllables": [
    {
      "index": 0,
      "hanzi": "你",
      "tone_score": 0.9,
      "tone_score_tolerance": 0.1,
      "segmental_score": 0.85,
      "segmental_tolerance": 0.1,
      "predicted_tone": 2,
      "tags": []
    },
    {
      "index": 1,
      "hanzi": "好",
      "tone_score": 0.88,
      "tone_score_tolerance": 0.1,
      "segmental_score": 0.82,
      "segmental_tolerance": 0.1,
      "predicted_tone": 3,
      "tags": []
    }
  ]
}
```

**Test Categories:**

| Category | Count | Purpose |
|----------|-------|---------|
| Basic (2-3 syllables) | 50 | Core functionality |
| Sandhi cases | 30 | 3rd tone, yi, bu rules |
| Neutral tone | 20 | Reduced syllable handling |
| Tone confusion | 40 | 2/3, 1/4 edge cases |
| Segmental errors | 30 | Initial/final confusions |
| Noisy audio | 20 | Robustness |
| Edge cases | 10 | Very short/long, silence |
| **Total** | **200** | |

### 7.5 Fairness Evaluation Criteria

**Demographic Groups:**

| Dimension | Groups |
|-----------|--------|
| Gender | Male, Female |
| Age | Child, Adult, Senior |
| L1 | Native, L1 English, L1 Japanese, L1 Korean |
| Pitch range | Low (< 150 Hz), Mid (150-250 Hz), High (> 250 Hz) |

**Fairness Metrics:**

1. **Equal Opportunity:** Same TPR for correct pronunciations across groups
2. **Calibration Error:** |predicted - actual| should be similar across groups
3. **Score Distribution:** Mean/variance should not differ significantly for same proficiency

**Fairness Thresholds:**

| Metric | Threshold |
|--------|-----------|
| Max group score difference (same proficiency) | < 5 points |
| Calibration error difference | < 3 points |
| False rejection rate difference | < 5% |

### 7.6 Performance Optimization

**Latency Targets:**

| Component | Target | Measured |
|-----------|--------|----------|
| Audio loading | < 50ms | - |
| Feature extraction | < 100ms | - |
| Alignment inference | < 150ms | - |
| Pitch extraction | < 50ms | - |
| Tone scoring | < 20ms | - |
| Segmental scoring | < 30ms | - |
| Total | < 500ms | - |

**Optimization Strategies:**

1. **Batch processing:** Process all syllables in single model call
2. **NNAPI delegation:** Use Android Neural Networks API for acceleration
3. **CoreML ANE:** Use Apple Neural Engine on iOS
4. **Lazy loading:** Load models on first use, not app startup
5. **Caching:** Cache alignment results for re-scoring

### 7.7 Deliverables

1. **Calibration dataset** with human ratings
2. **Fitted calibration model** (isotonic regression)
3. **Golden test suite** (200 tests)
4. **Fairness evaluation report**
5. **Performance optimization** achieving < 500ms
6. **Final model artifacts** (TFLite, CoreML)
7. **Release documentation**

---

## 8. Appendices

### Appendix A: Model Size Budgets

| Component | Target Size | Format | Notes |
|-----------|-------------|--------|-------|
| CTC Alignment Model | 5 MB | INT8 TFLite | Quantized from 20 MB FP32 |
| Tone Classifier | 3 KB | INT8 TFLite | Tiny MLP |
| G2P Lexicon | 500 KB | JSON | 6000+ entries |
| Tone Templates | 10 KB | JSON | 5 templates x 20 points |
| Calibration Model | 50 KB | JSON | Isotonic regression |
| **Total** | **~6 MB** | | |

### Appendix B: Latency Targets

| Device Tier | Target Latency | Example Devices |
|-------------|----------------|-----------------|
| High-end | < 300ms | iPhone 14+, Pixel 7+ |
| Mid-tier | < 500ms | iPhone 11, Pixel 5 |
| Low-end | < 1000ms | Budget Android |

**Breakdown (Mid-tier):**

| Stage | Latency |
|-------|---------|
| Load audio + resample | 30 ms |
| Mel spectrogram | 50 ms |
| CTC forward pass | 150 ms |
| Viterbi alignment | 50 ms |
| Pitch extraction (YIN) | 50 ms |
| Contour extraction | 20 ms |
| Tone classification | 10 ms |
| Segmental scoring | 30 ms |
| Fusion + calibration | 10 ms |
| **Total** | **400 ms** |

### Appendix C: Test Fixture JSON Schema

**Target Syllables:**

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "syllables": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "index": { "type": "integer" },
          "hanzi": { "type": "string" },
          "pinyin": { "type": "string" },
          "initial": { "type": "string" },
          "final": { "type": "string" },
          "tone_underlying": { "type": "integer", "minimum": 0, "maximum": 4 },
          "tone_surface": { "type": "integer", "minimum": 0, "maximum": 4 }
        },
        "required": ["index", "hanzi", "pinyin", "initial", "final", "tone_underlying", "tone_surface"]
      }
    }
  }
}
```

**Alignment Spans:**

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "syllable_spans": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "index": { "type": "integer" },
          "start_ms": { "type": "integer" },
          "end_ms": { "type": "integer" },
          "confidence": { "type": "number" },
          "phone_spans": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "phone": { "type": "string" },
                "start_ms": { "type": "integer" },
                "end_ms": { "type": "integer" },
                "confidence": { "type": "number" }
              }
            }
          }
        }
      }
    }
  }
}
```

**Expected Scores:**

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "overall": { "type": "number" },
    "syllables": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "segmental": { "type": "number" },
          "tone": { "type": "number" },
          "fluency": { "type": "number" },
          "overall": { "type": "number" },
          "tone_probs": {
            "type": "object",
            "additionalProperties": { "type": "number" }
          },
          "tags": {
            "type": "array",
            "items": { "type": "string" }
          }
        }
      }
    },
    "warnings": {
      "type": "array",
      "items": { "type": "string" }
    }
  }
}
```

### Appendix D: Integration API Contracts (Dart Interfaces)

**PronunciationScorer (existing):**

```dart
abstract class PronunciationScorer {
  Future<Grade> score(TextSequence sequence, Recording recording);
}
```

**MandarinScorerPlugin (new):**

```dart
abstract class MandarinScorerPlugin {
  /// Initialize the native scorer with model paths
  Future<void> initialize({
    required String ctcModelPath,
    required String toneModelPath,
    required String lexiconPath,
  });

  /// Score audio against target syllables
  Future<NativeScoringResult> score({
    required String audioPath,
    required List<TargetSyllable> targets,
  });

  /// Check if scorer is initialized
  bool get isInitialized;

  /// Release native resources
  Future<void> dispose();
}

class NativeScoringResult {
  final double overall;
  final List<NativeSyllableResult> syllables;
  final List<String> warnings;
}

class NativeSyllableResult {
  final int index;
  final double toneScore;
  final double segmentalScore;
  final double overall;
  final int predictedTone;
  final List<double> toneProbs;
  final List<String> tags;
}
```

**SandhiEngine (new):**

```dart
abstract class SandhiEngine {
  /// Apply Mandarin tone sandhi rules to target syllables
  List<TargetSyllable> apply(List<TargetSyllable> targets);
}

class MandarinSandhiEngine implements SandhiEngine {
  @override
  List<TargetSyllable> apply(List<TargetSyllable> targets) {
    var result = List<TargetSyllable>.from(targets);
    result = _apply3rdToneSandhi(result);
    result = _applyYiRule(result);
    result = _applyBuRule(result);
    return result;
  }
}
```

**Extended Grade (updated):**

```dart
@freezed
class Grade with _$Grade {
  const factory Grade({
    required int overall,
    required String method,
    int? accuracy,
    int? completeness,
    String? recognizedText,
    Map<String, dynamic>? details,
    List<SyllableGrade>? syllables,
    List<String>? warnings,
  }) = _Grade;

  factory Grade.fromJson(Map<String, dynamic> json) => _$GradeFromJson(json);
}

@freezed
class SyllableGrade with _$SyllableGrade {
  const factory SyllableGrade({
    required int index,
    required String hanzi,
    required String pinyin,
    required double toneScore,
    required double segmentalScore,
    required double overall,
    required int expectedTone,
    required int predictedTone,
    required Map<int, double> toneProbs,
    required List<String> tags,
  }) = _SyllableGrade;

  factory SyllableGrade.fromJson(Map<String, dynamic> json) =>
      _$SyllableGradeFromJson(json);
}
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-08 | Claude | Initial draft |

---

*End of Document*
