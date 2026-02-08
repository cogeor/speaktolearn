# Loop 11: Implementation Record

## Task: Create implementation plan for advanced Mandarin scoring from scoring.md

Completed: 2026-02-08T21:50:00Z

### Changes

- `C:\Users\costa\src\speaktolearn\.delegate\work\20260208-214116-tasks-implementation\SCORING_PLAN.md`: Created comprehensive 6-phase implementation plan document (~1200 lines)

### Document Structure Created

1. **Executive Summary** - Phase overview table, success criteria, risks/mitigations, dependencies
2. **Phase 1: Foundation** - Python package structure, core types, sandhi module, pitch normalization, test coverage targets
3. **Phase 2: Alignment Pipeline** - CTC model requirements, G2P lexicon format, forced alignment algorithm, quantization targets
4. **Phase 3: Tone Scoring** - Pitch extraction (YIN), contour features, tone classifier architecture, template-based distance, fusion formula
5. **Phase 4: Segmental Scoring** - GOP-like posterior scoring, initial/final weighting, duration penalties, confidence gating
6. **Phase 5: Mobile Integration** - Dart FFI architecture, TFLite/CoreML deployment, Grade model extension, UI components
7. **Phase 6: Calibration** - Dataset requirements, isotonic regression, golden test suite, fairness evaluation
8. **Appendices** - Model size budgets, latency targets, JSON schemas, Dart API contracts

### Verification

- [x] Detailed implementation plan document created: SCORING_PLAN.md (1,199 lines)
- [x] Plan covers CTC alignment: Phase 2 with model specs, G2P lexicon, Viterbi algorithm
- [x] Plan covers tone scoring: Phase 3 with YIN pitch, contour features, classifier, templates, fusion
- [x] Plan covers segmental scoring: Phase 4 with GOP scoring, initial/final, duration checks
- [x] Plan covers fusion: Sections 4.7 (tone fusion), 5.5 (segmental gating), scoring.md 2.9 (fuse module)
- [x] Plan includes phased approach: 6 phases over 24 weeks with milestones and go/no-go criteria
- [x] Dependencies and model requirements documented: TFLite, CoreML, INT8 quantization, model sizes
- [x] Integration points identified: PronunciationScorer, Grade, AsrSimilarityScorer, MandarinSyllableScorer

### Key Specifications from Document

| Metric | Target |
|--------|--------|
| Total timeline | 24 weeks (6 months) |
| CTC model size | 5 MB (INT8) |
| Tone classifier size | 3 KB (INT8) |
| Total model footprint | ~6 MB |
| End-to-end latency | < 500ms on mid-tier device |
| Tone accuracy target | > 85% on clean data |
| Alignment accuracy | > 90% on test set |
| Human correlation | > 0.75 for syllable scores |
| Golden tests | 200 test cases |

### Notes

- The plan synthesizes the full scoring.md spec into actionable development phases
- Each phase has clear deliverables, effort estimates, and acceptance criteria
- Integration with existing codebase (PronunciationScorer, Grade) is backward-compatible
- The existing AsrSimilarityScorer will be kept as a fallback option
- Sandhi rules (3rd tone, yi, bu) are given detailed test cases
- Fairness evaluation across gender, age, L1, and pitch range is specified

---
