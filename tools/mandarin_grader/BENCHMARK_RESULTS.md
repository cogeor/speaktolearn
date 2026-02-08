# Benchmark Results

Generated: 2026-02-09 00:05:53

## Summary

| Scorer | Accuracy | Avg Score | Time (s) | Errors |
|--------|----------|-----------|----------|--------|
| deterministic_uniform_rule_based | 28.3% | 49.6 | 48.49 | 0 |
| deterministic_uniform_template | 29.2% | 40.1 | 50.62 | 0 |

## Per-Tone Accuracy

| Scorer | T0 | T1 | T2 | T3 | T4 |
|--------|----|----|----|----|----| 
| deterministic_uniform_rule_based | 20% | 1% | 38% | 11% | 51% |
| deterministic_uniform_template | 22% | 4% | 37% | 15% | 50% |

## deterministic_uniform_rule_based

- Total syllables: 682
- Correct predictions: 193
- Overall accuracy: 28.3%
- Average score: 49.6
- Score std dev: 9.2

```
Confusion Matrix (rows=expected, cols=predicted):
       T0    T1    T2    T3    T4
     ------------------------------
  T0 |   31     0    46     8    68 
  T1 |   21     1    25     8    47 
  T2 |   19     2    44     5    45 
  T3 |   13     2    47    12    31 
  T4 |   37     1    44    20   105 
```

## deterministic_uniform_template

- Total syllables: 682
- Correct predictions: 199
- Overall accuracy: 29.2%
- Average score: 40.1
- Score std dev: 9.0

```
Confusion Matrix (rows=expected, cols=predicted):
       T0    T1    T2    T3    T4
     ------------------------------
  T0 |   33     6    36    14    64 
  T1 |   20     4    23     6    49 
  T2 |   24     1    42     6    42 
  T3 |   13     4    40    16    32 
  T4 |   43     7    34    19   104 
```

## Analysis

**Best configuration:** deterministic_uniform_template (29.2%)

**Most common confusions:**

- T0 -> T4: 132 occurrences
- T1 -> T4: 96 occurrences
- T2 -> T4: 87 occurrences
- T3 -> T2: 87 occurrences
- T0 -> T2: 82 occurrences

## Root Cause Analysis

The low accuracy (28-29%) is primarily due to pitch extraction issues:

1. **YIN algorithm limitations**: The pure-numpy YIN implementation produces octave errors
   (pitch doubling/halving) which distort the contour shape.

2. **Falling pitch declination**: Natural speech has overall falling pitch across sentences.
   Without proper per-syllable normalization relative to speaker baseline, all syllables
   appear to be falling.

3. **Uniform alignment**: Dividing audio equally without detecting actual syllable boundaries
   causes misalignment between extracted contours and expected tones.

4. **Voicing detection**: Edge frames at syllable boundaries often have unreliable F0 values
   which affect the contour shape analysis.

## Recommendations

[WARNING] Accuracy <80%. The deterministic approach requires significant improvements.

### Short-term fixes:
- Replace YIN with librosa.pyin (requires librosa installation)
- Add octave error correction (detect and fix pitch doublings)
- Trim 10-20% from syllable edges before contour extraction
- Use per-syllable normalization instead of global normalization

### Medium-term improvements:
- Implement energy-based voice activity detection for better syllable segmentation
- Use DTW alignment with TTS reference audio for accurate boundaries
- Derive empirical templates from TTS data (average contours per tone)

### Consider DL if:
- Short-term fixes don't achieve 80% accuracy
- DTW + template matching still shows >50ms boundary errors