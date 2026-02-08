# Loop 11: Create implementation plan for advanced Mandarin scoring

## Objective

Create `SCORING_PLAN.md` - a detailed, phased implementation plan for on-device Mandarin syllable-level speech grading. This document will translate the comprehensive spec in `scoring.md` into actionable development phases with clear milestones, dependencies, and integration points.

## Source Material

- **scoring.md**: Full system design spec covering:
  - Section 1: System design (goals, architecture, data representations, front-end, alignment, tone handling, scoring, fusion, feedback, on-device constraints)
  - Section 2: Python scoring and testing module spec (package layout, types, modules, testing strategy)
  - Section 3: Recommended "best choice" implementation summary

## Current Codebase Analysis

### Existing Scoring Architecture

The current implementation uses a simple ASR-based approach:

| Component | File | Function |
|-----------|------|----------|
| `PronunciationScorer` | `lib/features/scoring/domain/pronunciation_scorer.dart` | Abstract interface for scoring |
| `AsrSimilarityScorer` | `lib/features/scoring/data/asr_similarity_scorer.dart` | Main scorer using ASR + CER |
| `SpeechRecognizer` | `lib/features/scoring/data/speech_recognizer.dart` | Abstract ASR interface |
| `SpeechToTextRecognizer` | `lib/features/scoring/data/speech_to_text_recognizer.dart` | Live speech-to-text using platform SDK |
| `CerCalculator` | `lib/features/scoring/data/cer_calculator.dart` | Character Error Rate calculation |
| `CerResult` | `lib/features/scoring/data/cer_result.dart` | CER result with score/accuracy/completeness |
| `Grade` | `lib/features/scoring/domain/grade.dart` | Freezed model: overall, method, accuracy, completeness, details |

### Current Scoring Flow

1. User records audio (SpeechToTextRecognizer listens during recording)
2. Recognition text cached by audio path
3. `AsrSimilarityScorer.score()` retrieves cached text
4. CER calculated between expected and recognized text
5. Grade returned with overall score (0-100)

### Limitations of Current Approach

- **No syllable-level feedback**: Only sentence-level CER
- **No tone scoring**: ASR transcribes tones implicitly, no explicit tone analysis
- **No segmental analysis**: No initial/final consonant scoring
- **Platform-dependent**: Relies on OS speech recognition API
- **No pitch extraction**: No F0 analysis for tone verification
- **No forced alignment**: No time-aligned syllable boundaries

### What Needs to Be Built (from scoring.md)

1. **CTC Forced Alignment Model**: Map audio to syllable/phone spans
2. **Pitch Extractor**: Frame-level F0 and voicing detection
3. **Tone Classifier**: Small neural net for tone classification
4. **Tone Templates**: Reference contours for contour distance
5. **Segmental Scorer**: GOP-like posterior-based phone scoring
6. **Fusion Layer**: Combine subscores with confidence gating
7. **Sandhi Engine**: Surface tone computation rules
8. **Calibration Layer**: Map raw scores to user-friendly 0-100

## Document Structure for SCORING_PLAN.md

### 1. Executive Summary
- High-level phased roadmap
- Timeline estimates per phase
- Go/no-go decision points

### 2. Phase 1: Foundation (Weeks 1-4)
- Python reference implementation setup
- Core types and data structures
- Sandhi rules implementation
- Pitch normalization utilities
- Unit test framework

### 3. Phase 2: Alignment Pipeline (Weeks 5-8)
- CTC acoustic model selection/training
- G2P lexicon for Mandarin
- Forced alignment implementation
- Phone-level span extraction
- Model quantization for mobile

### 4. Phase 3: Tone Scoring (Weeks 9-12)
- Pitch extraction algorithm (YIN/CREPE-lite)
- Contour feature extraction
- Tone classifier training
- Template-based contour distance
- Tone score fusion

### 5. Phase 4: Segmental Scoring (Weeks 13-16)
- GOP-like posterior extraction
- Initial/final scoring
- Confidence gating
- Integration with alignment spans

### 6. Phase 5: Mobile Integration (Weeks 17-20)
- Dart FFI bridge to native inference
- TFLite/CoreML model deployment
- Flutter scoring service
- Grade model extension
- UI feedback components

### 7. Phase 6: Calibration and Polish (Weeks 21-24)
- Score calibration dataset collection
- Isotonic regression fitting
- Golden test suite
- Fairness evaluation
- Performance optimization

### 8. Appendices
- Model size budgets
- Latency targets
- Test fixture formats
- Integration API contracts

## Tasks

### Task 1: Write introduction and executive summary

**Goal:** Establish context, scope, and phased roadmap overview

**Files:**
| Action | Path |
|--------|------|
| CREATE | `.delegate/work/20260208-214116-tasks-implementation/SCORING_PLAN.md` |

**Steps:**
1. Write document header with title and date
2. Summarize current state (ASR+CER) vs target state (syllable-level grading)
3. Create high-level phase summary table with weeks and deliverables
4. Define success criteria for overall project
5. List key risks and mitigations

**Verify:** Document has clear executive summary with phase table

---

### Task 2: Write Phase 1 (Foundation) section

**Goal:** Detail the Python reference implementation setup

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `.delegate/work/20260208-214116-tasks-implementation/SCORING_PLAN.md` |

**Steps:**
1. Define Python package structure from scoring.md Section 2.1
2. Document core types (TargetSyllable, SyllableSpan, FrameTrack, etc.)
3. Specify sandhi module requirements with test cases
4. Specify pitch normalization module with invariants
5. Define unit test coverage targets
6. Estimate effort (person-weeks) and deliverables

**Verify:** Phase 1 section complete with all modules listed

---

### Task 3: Write Phase 2 (Alignment Pipeline) section

**Goal:** Detail CTC forced alignment implementation

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `.delegate/work/20260208-214116-tasks-implementation/SCORING_PLAN.md` |

**Steps:**
1. Document acoustic model requirements (tiny Conformer/CTC)
2. Specify G2P lexicon format for Mandarin syllables
3. Detail forced alignment algorithm (Viterbi-like)
4. Define output format (SyllableSpan, PhoneSpan)
5. Specify model quantization targets (INT8, <10MB)
6. Document missing/extra syllable detection logic
7. Estimate effort and list dependencies

**Verify:** Phase 2 covers model selection through deployment

---

### Task 4: Write Phase 3 (Tone Scoring) and Phase 4 (Segmental Scoring) sections

**Goal:** Detail the two main scoring subsystems

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `.delegate/work/20260208-214116-tasks-implementation/SCORING_PLAN.md` |

**Steps:**
1. Phase 3: Document pitch extraction algorithm options
2. Phase 3: Specify contour feature extraction (K=20 points, derivatives)
3. Phase 3: Define tone classifier architecture and training data
4. Phase 3: Document template-based distance computation
5. Phase 3: Specify fusion formula (w_cls, w_shape weights)
6. Phase 4: Document GOP-like posterior scoring
7. Phase 4: Specify initial/final weighting for Mandarin
8. Phase 4: Document duration sanity checks
9. Estimate effort for each phase

**Verify:** Both phases have architecture, training, and inference documented

---

### Task 5: Write Phase 5 (Mobile Integration) section

**Goal:** Detail Flutter/Dart integration with native models

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `.delegate/work/20260208-214116-tasks-implementation/SCORING_PLAN.md` |

**Steps:**
1. Document Dart FFI approach for native inference
2. Specify TFLite integration for Android
3. Specify CoreML integration for iOS
4. Define new Grade model fields for syllable-level scores
5. Map integration points to existing codebase:
   - `PronunciationScorer` interface extension
   - New `MandarinSyllableScorer` implementation
   - `Grade` model extension (per-syllable scores, tone_probs, tags)
6. Document UI feedback component requirements
7. Estimate effort and dependencies

**Verify:** Integration plan connects Python reference to Flutter app

---

### Task 6: Write Phase 6 (Calibration) and Appendices sections

**Goal:** Complete document with calibration, testing, and reference material

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `.delegate/work/20260208-214116-tasks-implementation/SCORING_PLAN.md` |

**Steps:**
1. Phase 6: Document calibration dataset requirements
2. Phase 6: Specify isotonic regression fitting process
3. Phase 6: Define golden test suite structure
4. Phase 6: Document fairness evaluation criteria
5. Appendix A: Model size budgets table
6. Appendix B: Latency targets table
7. Appendix C: Test fixture JSON schema
8. Appendix D: Integration API contracts (Dart interfaces)

**Verify:** Document complete with all sections and appendices

---

### Task 7: Review and finalize document

**Goal:** Ensure document is complete, consistent, and actionable

**Files:**
| Action | Path |
|--------|------|
| MODIFY | `.delegate/work/20260208-214116-tasks-implementation/SCORING_PLAN.md` |

**Steps:**
1. Review all sections for completeness against scoring.md
2. Verify phase dependencies are correctly sequenced
3. Check effort estimates sum to realistic timeline
4. Ensure integration points reference correct file paths
5. Add table of contents
6. Final formatting pass

**Verify:** Document ready for implementation team review

## Acceptance Criteria

- [ ] Detailed implementation plan document created at `.delegate/work/20260208-214116-tasks-implementation/SCORING_PLAN.md`
- [ ] Plan covers: CTC alignment, tone scoring, segmental scoring, fusion
- [ ] Plan includes phased approach with milestones (6 phases)
- [ ] Dependencies and model requirements documented (TFLite, CoreML, model sizes)
- [ ] Integration points with current codebase identified (PronunciationScorer, Grade, etc.)

## Deliverable

`SCORING_PLAN.md` at:
```
C:\Users\costa\src\speaktolearn\.delegate\work\20260208-214116-tasks-implementation\SCORING_PLAN.md
```
