---

# 1) System Design Spec — Mandarin Syllable-Level Speech Grading (On-Device)

## 1.1 Goals and Non-Goals

### Goals

1. **Mobile-usable**: runs locally on iOS/Android with low latency.
2. **Speaker-independent**: robust across voice type, pitch range, microphone differences.
3. **Reference sentence grading**: compare learner speech to a known prompt (text + reference audio available).
4. **Mandarin-specific**: tone correctness is first-class (tone sandhi + neutral tone).
5. **Actionable feedback**: per-syllable scores + localized error tags (tone vs initials/finals).

### Non-Goals (initial phase)

* Fully open-ended free speech scoring (no reference prompt).
* Accent identification.
* Perfect “phonetic tutoring” at the allophone level; keep feedback practical.

---

## 1.2 High-Level Architecture

**Inputs**

* Prompt text: Chinese characters + (optional) target pinyin with tones
* Reference audio (human-recorded preferred; TTS allowed with caveats)
* Learner audio

**Core pipeline**

1. Front-end: VAD + trimming + level normalization (optional denoise)
2. **Alignment/segmentation** (syllable boundaries)
3. Per-syllable scoring:

   * Segmentals (initial/final) score
   * Tone score
   * Fluency micro-metrics
4. Fusion: syllable → word/phrase → sentence grade
5. Feedback generation: highlight syllables, identify likely errors, show suggestions

**On-device models**

* Alignment acoustic model (small CTC/Conformer or similar)
* Optional phonetic posterior model (PPG-like) if you choose DTW alignment
* Tone classifier (tiny)
* Pitch extractor (algorithmic)

**Outputs**

* Sentence score (0–100)
* Per-syllable:

  * overall score (0–1)
  * tone score (0–1) + predicted tone distribution
  * segmental score (0–1) + initial/final sub-scores
  * tags (e.g., “tone_2_vs_3_confusion”, “final_nasal_missing”, “timing_fast”)

---

## 1.3 Data Representations

### Syllable Target Representation

For each target syllable:

* `hanzi`: character (or multi-character syllable unit if needed)
* `pinyin_base`: e.g., “shi”
* `initial`: e.g., “sh”
* `final`: e.g., “i”
* `tone_underlying`: 1–4 or 0 for neutral
* `tone_surface`: after sandhi rules applied (see 1.6)
* `variants`: optional acceptable variants (erhua, reductions, etc.)

### Acoustic Observations per Frame

* `log_mel[t, n_mels]`
* `voicing_flag[t]`
* `f0_hz[t]` (or semitone normalized)
* `energy[t]` (optional)

---

## 1.4 Front-End (Mobile)

### VAD + Trimming

* Remove leading/trailing silence; optionally keep short margins (100–200 ms).
* If long internal silences, mark them as pauses (fluency metric).

### Normalization

* Peak or RMS normalize to a consistent amplitude range.
* Optional: simple noise suppression if device constraints allow.

### Pitch Extraction

* Extract frame-level F0 and voicing probability.
* Store raw F0 Hz + speaker-normalized F0 (see 1.7).

**Key requirement:** tone scoring uses learner’s **original F0**, not synthesized audio.

---

## 1.5 Alignment & Syllable Segmentation

You have two strong choices; the “best default” for your problem is:

### Recommended: Text-guided forced alignment via CTC

**Why:** you know the target sentence, and you want interpretable mapping from audio → specific syllables.

**Process**

1. Convert prompt text → target syllable sequence + (initial/final phones) using G2P + lexicon.
2. Run acoustic model to produce frame-level posteriors over tokens (phones or syllable tokens + blank).
3. Use CTC forced alignment (Viterbi-like) to find the best path matching the expected token sequence.
4. Collapse into time spans for:

   * each syllable
   * optionally each phone within syllable (initial/final)

**Outputs**

* `Alignment`: list of `SyllableSpan{start_ms, end_ms, phones...}` with confidence
* Missing/extra detection:

  * if alignment fails or yields low confidence for a region, mark as `unreliable_alignment`

### Optional (Phase 2): Reference-audio alignment via PPG + DTW

Use when you want “shadowing similarity” vs a specific reference voice. It’s great as an auxiliary score, but not strictly necessary if forced alignment works well.

---

## 1.6 Mandarin Tone Handling (Targets)

### Surface-tone computation (mandatory)

Before scoring, compute the **expected surface tones** given context:

* 3rd tone sandhi: 3 + 3 → 2 + 3 (surface)
* “一” tone change
* “不” tone change
* Neutral tone behavior (syllable reductions)

Store both:

* `tone_underlying` (pedagogical display)
* `tone_surface` (what you grade)

### Neutral tone scoring rule

Neutral tone is not just “tone 0”; typical cues:

* reduced duration
* reduced pitch excursion, often determined by preceding tone’s pitch trajectory

Your scoring should:

* allow a range of realizations
* emphasize duration + relative pitch rather than strict contour templates

---

## 1.7 Tone Scoring (Per Syllable)

Tone scoring is a **fusion** of:

1. Tone classifier probability (categorical correctness)
2. F0 contour similarity (shape correctness)

### Step A — F0 normalization (speaker independence)

Compute per-utterance pitch normalization:

* Convert Hz → semitones: `st = 12 * log2(f0 / ref)`
* Normalize by subtracting speaker mean (voiced frames), divide by std (z-norm)
* Keep robust stats (median/MAD) for noisy cases

### Step B — Syllable contour extraction

For each syllable span:

* sample normalized F0 into fixed length `K` points (e.g., K=20) over voiced frames
* compute derivatives: ΔF0, Δ²F0
* compute summary features:

  * slope, min/max position, range, voicing ratio, duration

### Step C — Tone classifier

Tiny model input:

* `[K] normalized F0 + [K] ΔF0 + duration + voicing_ratio (+ optional short mel slice)`

Output:

* `p(tone=1..4, neutral)` per syllable

### Step D — Contour similarity

Compare learner contour to expected tone contour via:

* template-based distance (tone-dependent) OR
* reference-audio contour distance (if you have a human reference, preferred)

Use:

* time-normalized L2 distance
* DTW distance on contours (optional; slower but tolerant)

### Step E — Combine

Example:

* `tone_score = w_cls * p(correct_surface_tone) + w_shape * (1 - normalized_distance)`
* Add confusion-aware penalties (2 vs 3 gets smaller penalty than 1 vs 4, configurable)

---

## 1.8 Segmental Scoring (Initial/Final)

### Approach: Posterior-based “GOP-like” scoring per phone

From alignment model posteriors:

* For each phone segment:

  * compute average posterior of expected phone across frames
  * compare to competing phone posteriors to quantify ambiguity

Aggregate:

* `initial_score`: average of phones in initial
* `final_score`: average of phones in final
* `segmental_score = α*initial + (1-α)*final` (Mandarin often weights finals slightly more)

Add duration sanity checks:

* very short segments → degrade confidence
* too long segments → possible hesitations

---

## 1.9 Fluency & Timing Micro-metrics

Per syllable:

* duration z-score (relative to expected range)
* intra-syllable voicing stability
  Between syllables:
* pause duration and placement
* speaking rate stability

These typically shouldn’t dominate the score but help feedback.

---

## 1.10 Scoring Fusion and Calibration

### Per syllable overall

`syllable_score = w_seg*segmental_score + w_tone*tone_score + w_flu*fluency_score`

### Sentence score

* weighted mean of syllables
* penalties:

  * missing syllables
  * insertions
  * repeated syllables
  * low alignment confidence spans

### Calibration layer (strongly recommended)

Map raw scores to user-facing 0–100:

* isotonic regression or logistic mapping
* calibrate with a small labeled dataset (even a few hundred samples helps)

---

## 1.11 Feedback Generation

For each syllable, output:

* traffic light indicator
* top error tag:

  * tone mismatch (show predicted vs expected)
  * likely initial confusion
  * final nasalization confusion
  * timing issue

Display tone feedback as:

* expected surface tone number + simple contour icon
* learner contour overlay (normalized)

---

## 1.12 On-Device Constraints & Implementation Notes

### Model sizes (guideline)

* Alignment model: “tiny” Conformer/CTC; quantized INT8
* Tone classifier: very small (tens–hundreds of KB)
* Pitch extractor: algorithmic, fast

### Runtime targets (typical)

* Real-time or faster-than-real-time decoding on mid-tier devices
* Memory budget: keep < ~50–100MB for models + buffers if possible

### Deployment format

* Android: ONNX Runtime Mobile (NNAPI acceleration)
* iOS: ONNX Runtime Mobile (CoreML execution provider)
* Single .onnx model file for both platforms

---

## 1.13 Security, Privacy, Fairness

* Fully on-device by default
* Avoid saving raw audio unless user opts in
* Evaluate fairness across:

  * male/female pitch ranges
  * child voices (if applicable)
  * accents / regional variants
  * different microphones/noise conditions

---

# 2) Python Scoring & Testing Module Spec (Typed)

This module’s job is to:

1. Implement **deterministic scoring logic** (tone scoring math, contour features, fusion rules).
2. Provide a **test harness** to validate:

   * invariants (e.g., pitch normalization invariance)
   * golden regression tests (scores don’t drift)
   * sandhi correctness
   * boundary/edge-case correctness

It does **not** need to run the full on-device models; instead it consumes:

* alignment spans
* per-frame posteriors (from offline runs)
* per-frame F0/voicing
* target syllable metadata

Think of it as a reference implementation + QA suite.

---

## 2.1 Package Layout

```
mandarin_grader/
  __init__.py
  types.py
  sandhi.py
  pitch.py
  contour.py
  tone.py
  segmental.py
  fluency.py
  fuse.py
  scorer.py
  fixtures/
    prompts/
    alignments/
    f0_tracks/
    posteriors/
    expected_scores/
  tests/
    test_sandhi.py
    test_pitch_norm.py
    test_contours.py
    test_tone_scoring.py
    test_segmental_scoring.py
    test_fusion.py
    test_regression_golden.py
```

---

## 2.2 Core Types (typed)

### `types.py`

* Use `dataclasses` and `typing` protocols.
* Prefer `numpy.ndarray` for arrays (with runtime validation of shapes).

Key types:

* `Tone`: `Literal[0,1,2,3,4]`
* `Ms`: `NewType("Ms", int)`

Data classes:

* `TargetSyllable`

  * `index: int`
  * `hanzi: str`
  * `pinyin: str`
  * `initial: str`
  * `final: str`
  * `tone_underlying: Tone`
  * `tone_surface: Tone`
  * `start_expected_ms: Ms | None` (optional)
  * `end_expected_ms: Ms | None`

* `SyllableSpan`

  * `index: int`
  * `start_ms: Ms`
  * `end_ms: Ms`
  * `confidence: float`  # 0..1
  * `phone_spans: list[PhoneSpan] | None`

* `PhoneSpan`

  * `phone: str`
  * `start_ms: Ms`
  * `end_ms: Ms`
  * `confidence: float`

* `FrameTrack`

  * `frame_hz: float`  # frame rate
  * `f0_hz: np.ndarray`        # shape [T]
  * `voicing: np.ndarray`      # shape [T], 0..1
  * `energy: np.ndarray | None`

* `PosteriorTrack`

  * `frame_hz: float`
  * `token_names: list[str]`
  * `posteriors: np.ndarray`   # shape [T, V]

* `SyllableScores`

  * `segmental: float`
  * `tone: float`
  * `fluency: float`
  * `overall: float`
  * `tone_probs: dict[Tone, float]`
  * `tags: list[str]`

* `SentenceScore`

  * `overall: float`  # 0..100
  * `syllables: list[SyllableScores]`
  * `warnings: list[str]`

---

## 2.3 Sandhi Module

### `sandhi.py`

Responsibilities:

* Convert underlying tones to surface tones using deterministic rules.

Functions:

* `apply_tone_sandhi(targets: list[TargetSyllable]) -> list[TargetSyllable]`
* `apply_bu_yi_rules(...)`

Tests:

* `test_3rd_tone_sandhi_pairs()`
* `test_yi_rule_before_4th_tone()`
* `test_bu_rule_before_4th_tone()`
* `test_neutral_tone_preservation()`

---

## 2.4 Pitch Normalization Module

### `pitch.py`

Responsibilities:

* Convert Hz to semitones (with a stable reference)
* Normalize per speaker utterance
* Robust handling of unvoiced frames

Functions:

* `hz_to_semitones(f0_hz: np.ndarray, ref_hz: float) -> np.ndarray`
* `normalize_f0(semitones: np.ndarray, voicing: np.ndarray) -> np.ndarray`
* `robust_stats(...) -> (mean, std)` using median/MAD option

Key invariants to test:

* Multiplying all voiced F0 by a constant should not change normalized contour shape (up to numerical tolerance).
* Unvoiced frames should be ignored in stats and not create NaNs.

---

## 2.5 Contour Extraction Module

### `contour.py`

Responsibilities:

* Extract fixed-length contour vectors per syllable span
* Compute derivatives and summary features

Functions:

* `extract_syllable_contour(track: FrameTrack, span: SyllableSpan, k: int=20) -> Contour`
* `resample_contour(values: np.ndarray, k: int) -> np.ndarray`

Types:

* `Contour`

  * `f0_norm: np.ndarray` shape [K]
  * `df0: np.ndarray` shape [K]
  * `ddf0: np.ndarray` shape [K]
  * `duration_ms: int`
  * `voicing_ratio: float`

Tests:

* `test_resampling_shape()`
* `test_voicing_ratio_bounds()`
* `test_short_span_behavior()` (very short syllables)

---

## 2.6 Tone Scoring Module

### `tone.py`

Responsibilities:

* Tone classifier interface (pluggable)
* Template/reference contour distance
* Final tone score computation + tags

Interfaces:

* `Protocol ToneClassifier`

  * `predict_proba(contour: Contour) -> dict[Tone, float]`

Functions:

* `tone_contour_distance(contour: Contour, target_tone: Tone, templates: ToneTemplates) -> float`
* `score_tone(contour: Contour, target_tone: Tone, clf: ToneClassifier, templates: ToneTemplates, config: ToneConfig) -> ToneResult`

Types:

* `ToneTemplates`: mapping `Tone -> np.ndarray[K]` (or multiple templates)
* `ToneConfig`:

  * weights `w_cls, w_shape`
  * confusion matrix (optional)
  * neutral tone handling parameters

`ToneResult`

* `score: float`
* `probs: dict[Tone, float]`
* `tags: list[str]`

Tests:

* Classification dominance test: if `p(correct)=1`, tone score should be near 1 even if shape mediocre (depending on weights).
* Shape dominance test: if `p(correct)` low but shape matches template strongly, score should reflect it.
* Confusion sensitivity: 2 vs 3 penalty smaller than 1 vs 4.

---

## 2.7 Segmental Scoring Module

### `segmental.py`

Responsibilities:

* Compute GOP-like scores from posterior tracks and phone spans

Functions:

* `score_phone(post: PosteriorTrack, span: PhoneSpan, expected_token: str, config: SegmentalConfig) -> float`
* `score_syllable_segmentals(post: PosteriorTrack, syllable_span: SyllableSpan, expected_phones: list[str], config: SegmentalConfig) -> SegmentalResult`

`SegmentalConfig`:

* min span duration
* posterior aggregation method (mean log prob vs mean prob)
* competitor margin threshold for “confusion” tagging

Tests:

* Posterior monotonicity: increasing expected posterior should not decrease score.
* Span boundaries: scores stable under ±1 frame jitter (within tolerance).

---

## 2.8 Fluency Module

### `fluency.py`

Responsibilities:

* Duration z-scores
* Pause detection (if spans include gaps)
* Speaking rate metrics

Functions:

* `score_syllable_fluency(span: SyllableSpan, expected_stats: ExpectedTiming, config: FluencyConfig) -> float`

Tests:

* Extremely long pauses lower fluency.
* Reasonable durations produce neutral-to-high fluency.

---

## 2.9 Fusion Module

### `fuse.py`

Responsibilities:

* Combine subscores with weights + confidence gating

Functions:

* `fuse_syllable_scores(seg: float, tone: float, flu: float, conf: float, config: FusionConfig) -> float`

Key logic:

* If alignment confidence below threshold:

  * reduce weight on segmentals
  * optionally produce warning tag instead of harsh penalty

Tests:

* Confidence gating: lower confidence reduces impact of subscores.

---

## 2.10 End-to-end Scorer

### `scorer.py`

Top-level function:

* `score_sentence(targets: list[TargetSyllable], spans: list[SyllableSpan], f0_track: FrameTrack, post: PosteriorTrack, tone_clf: ToneClassifier, templates: ToneTemplates, configs: ConfigBundle) -> SentenceScore`

Responsibilities:

* Apply sandhi to targets
* Normalize pitch and extract contours
* Score each syllable (tone + segmental + fluency)
* Fuse
* Sentence-level aggregation + penalties
* Return structured output

---

## 2.11 Testing Strategy (Very Explicit)

### A) Unit tests (fast)

* Sandhi rules correctness
* Pitch normalization invariants
* Contour extraction shape + edge cases
* Segmental scoring monotonicity
* Fusion gating

### B) Property-based tests (optional but powerful)

Use Hypothesis to generate random F0 tracks and check invariants:

* scaling F0 in Hz doesn’t change normalized contour score
* adding unvoiced frames doesn’t break normalization

### C) Golden regression tests (most important for “ensure they are correct”)

Store fixture bundles:

* `prompt.json`
* `targets.json` (with underlying tones)
* `spans.json`
* `f0.npy`, `voicing.npy`
* `posteriors.npy`
* `expected_scores.json`

Then:

* assert scores match within tolerances
* assert tags match exactly (or stable subset)

### D) “Adversarial” fixtures

Include cases that commonly break Mandarin grading:

* 3rd tone sandhi (two 3rd tones)
* neutral tone words (e.g., 的, 了, 吗 patterns)
* yi/bu changes
* tone 2 vs 3 confusion
* very high/low pitch speakers
* fast speech vs slow speech

---

## 2.12 Typed Python Guidelines (enforced)

* `mypy --strict` target
* Explicit return types everywhere
* Use `Protocol` for model interfaces (tone classifier)
* Validate array shapes at runtime (lightweight checks) to avoid silent bugs

---

# 3) Recommended “Best Choice” Implementation (Summary)

**Best default approach for your problem**

* **CTC forced alignment** to get syllable (and optionally phone) spans
* **Segmental scoring** via posterior-based GOP-like scoring (initial/final)
* **Tone scoring** via:

  * speaker-normalized F0 contour features
  * tiny tone classifier probabilities
  * contour distance to (preferably human) reference contour or tone templates
* **Sandhi-aware targets**
* **Confidence gating** to avoid punishing alignment failures

This is both **state-of-practice** and **mobile-realistic**, and it avoids the Whisper issue you called out (language-model “healing” of ambiguous tone evidence).

---

If you want, I can also provide:

* a concrete JSON schema for fixtures (`targets`, `spans`, configs, expected outputs),
* recommended thresholds/weights that are sane starting points for Mandarin (and why),
* and a minimal “fake tone classifier” implementation for tests (deterministic) so your unit tests don’t depend on ML inference.