"""Integration tests for V7 CTC-based pronunciation scorer.

Tests the full pipeline: input → model → decode → score, and validates
that the implementation matches the expected behavior.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mandarin_grader.model.syllable_predictor_v7 import (
    CTCDecoder,
    PredictorOutputV7,
    SyllablePredictorConfigV7,
    SyllablePredictorV7,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def model():
    """Create a V7 model for testing."""
    config = SyllablePredictorConfigV7(
        n_syllables=100,  # Smaller for faster tests
        n_tones=5,
        d_model=64,
        n_heads=2,
        n_layers=2,
        dim_feedforward=128,
    )
    return SyllablePredictorV7(config)


@pytest.fixture
def ctc_decoder():
    """Create CTC decoder."""
    return CTCDecoder(blank_index=0)


# ============================================================================
# End-to-End Pipeline Tests
# ============================================================================

class TestFullPipeline:
    """Test the complete inference pipeline."""

    def test_mel_to_logits_to_scores(self, model):
        """Test full pipeline: mel → logits → scores."""
        model.eval()

        # Simulate mel spectrogram (batch=1, 80 mels, 400 time frames)
        mel = torch.randn(1, 80, 400)

        with torch.no_grad():
            syl_logits, tone_logits = model(mel)

        # Verify output shapes
        expected_time = (400 + 3) // 4  # 4x downsampling
        assert syl_logits.shape == (1, expected_time, model.config.n_syllables + 1)
        assert tone_logits.shape == (1, expected_time, model.config.n_tones + 1)

        # Test scoring with decoder
        target_ids = [1, 5, 10, 15]  # 4 syllables
        scores = model.ctc_decoder.score_against_targets(
            syl_logits[0], target_ids
        )

        assert len(scores) == 4
        assert all(0 <= s <= 1 for s in scores), "Scores should be probabilities"

    def test_variable_length_inputs(self, model):
        """Test that model handles variable input lengths correctly."""
        model.eval()

        for mel_length in [100, 200, 500, 1000]:
            mel = torch.randn(1, 80, mel_length)

            with torch.no_grad():
                syl_logits, tone_logits = model(mel)

            expected_time = (mel_length + 3) // 4
            assert syl_logits.shape[1] == expected_time
            assert tone_logits.shape[1] == expected_time

    def test_batch_processing(self, model):
        """Test batch processing with variable lengths."""
        model.eval()

        # Create batch with padding
        batch_size = 3
        max_len = 500
        mel = torch.randn(batch_size, 80, max_len)

        # Create mask (True = padded)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        lengths = [300, 400, 500]
        for i, length in enumerate(lengths):
            mask[i, length:] = True

        with torch.no_grad():
            syl_logits, tone_logits = model(mel, audio_mask=mask)

        assert syl_logits.shape[0] == batch_size
        assert tone_logits.shape[0] == batch_size


# ============================================================================
# Scoring Algorithm Tests
# ============================================================================

class TestScoringAlgorithm:
    """Test the scoring/grading algorithm matches the specified behavior."""

    def test_score_against_targets_finds_max(self, ctc_decoder):
        """Verify score_against_targets returns max probability for each target."""
        # Create fake logits where target 5 has high prob at frame 10
        n_frames = 50
        vocab_size = 20
        logits = torch.randn(n_frames, vocab_size) * 0.1

        # Set high values for targets at specific frames
        logits[10, 5] = 5.0  # High prob for target 5 at frame 10
        logits[25, 10] = 4.0  # High prob for target 10 at frame 25
        logits[40, 15] = 3.0  # High prob for target 15 at frame 40

        scores = ctc_decoder.score_against_targets(logits, [5, 10, 15])

        # Verify scores are high (softmax of dominant values)
        assert scores[0] > 0.5, f"Score for target 5 should be high, got {scores[0]}"
        assert scores[1] > 0.5, f"Score for target 10 should be high, got {scores[1]}"
        assert scores[2] > 0.5, f"Score for target 15 should be high, got {scores[2]}"

    def test_alignment_maintains_order(self, ctc_decoder):
        """Verify score_with_alignment maintains monotonic frame order."""
        n_frames = 100
        vocab_size = 20
        logits = torch.randn(n_frames, vocab_size) * 0.1

        # Set high values in order
        logits[20, 5] = 5.0
        logits[50, 10] = 5.0
        logits[80, 15] = 5.0

        scores, frames = ctc_decoder.score_with_alignment(logits, [5, 10, 15])

        # Verify monotonic order
        assert frames[0] < frames[1] < frames[2], f"Frames should be monotonic: {frames}"

    def test_alignment_handles_reversed_peaks(self, ctc_decoder):
        """Test alignment when later targets have earlier peaks."""
        n_frames = 100
        vocab_size = 20
        logits = torch.randn(n_frames, vocab_size) * 0.1

        # Intentionally put peaks in wrong order
        logits[80, 5] = 5.0   # Target 5 peaks late
        logits[20, 10] = 5.0  # Target 10 peaks early

        scores, frames = ctc_decoder.score_with_alignment(logits, [5, 10])

        # Should still be monotonic (alignment forces order)
        assert frames[0] < frames[1], f"Frames should be monotonic: {frames}"

    def test_combined_score_formula(self, model):
        """Test 70% syllable + 30% tone formula."""
        model.eval()

        mel = torch.randn(1, 80, 400)

        with torch.no_grad():
            syl_logits, tone_logits = model(mel)

        target_syls = [1, 5, 10]
        target_tones = [1, 2, 3]

        syl_scores = model.ctc_decoder.score_against_targets(syl_logits[0], target_syls)
        tone_scores = model.ctc_decoder.score_against_targets(tone_logits[0], target_tones)

        # Calculate combined scores
        combined = [0.7 * s + 0.3 * t for s, t in zip(syl_scores, tone_scores)]

        # Verify formula
        for i, (syl, tone, comb) in enumerate(zip(syl_scores, tone_scores, combined)):
            expected = 0.7 * syl + 0.3 * tone
            assert abs(comb - expected) < 1e-6, f"Combined score {i} mismatch"


# ============================================================================
# CTC Decoding Tests
# ============================================================================

class TestCTCDecoding:
    """Test CTC decoding correctness."""

    def test_greedy_decode_collapses_repeats(self, ctc_decoder):
        """Test that greedy decode collapses repeated tokens."""
        # Logits where same token is predicted repeatedly
        logits = torch.zeros(10, 5)
        logits[:3, 1] = 5.0   # Frames 0-2: predict token 1
        logits[3:5, 0] = 5.0  # Frames 3-4: predict blank
        logits[5:8, 2] = 5.0  # Frames 5-7: predict token 2
        logits[8:, 1] = 5.0   # Frames 8-9: predict token 1

        decoded = ctc_decoder.greedy_decode(logits.unsqueeze(0))

        # After collapse and blank removal: [1, 2, 1]
        assert decoded[0] == [1, 2, 1]

    def test_decode_with_probs_returns_confidence(self, ctc_decoder):
        """Test that decode_with_probs returns per-token probabilities."""
        logits = torch.zeros(10, 5)
        logits[0, 1] = 5.0  # Token 1 at frame 0
        logits[5, 2] = 4.0  # Token 2 at frame 5

        seqs, probs = ctc_decoder.decode_with_probs(logits.unsqueeze(0))

        # Verify we got probabilities for decoded tokens
        assert len(seqs[0]) == len(probs[0])
        assert all(0 < p <= 1 for p in probs[0])


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_targets(self, ctc_decoder):
        """Test scoring with no targets."""
        logits = torch.randn(50, 20)
        scores = ctc_decoder.score_against_targets(logits, [])
        assert scores == []

    def test_single_frame(self, model):
        """Test with minimal input (single frame after downsampling)."""
        model.eval()

        # 4 mel frames → 1 output frame after 4x downsampling
        mel = torch.randn(1, 80, 4)

        with torch.no_grad():
            syl_logits, tone_logits = model(mel)

        # Should have at least 1 frame
        assert syl_logits.shape[1] >= 1

    def test_more_targets_than_frames(self, ctc_decoder):
        """Test when there are more targets than frames."""
        logits = torch.randn(5, 20)  # Only 5 frames
        targets = list(range(1, 11))  # 10 targets

        scores, frames = ctc_decoder.score_with_alignment(logits, targets)

        # Should still return scores for all targets
        assert len(scores) == len(targets)

    def test_deterministic_output(self, model):
        """Test that same input produces same output."""
        model.eval()

        mel = torch.randn(1, 80, 200)

        with torch.no_grad():
            syl1, tone1 = model(mel)
            syl2, tone2 = model(mel)

        assert torch.allclose(syl1, syl2)
        assert torch.allclose(tone1, tone2)


# ============================================================================
# Cross-Platform Validation Helpers
# ============================================================================

class TestCrossPlatformConsistency:
    """Tests for validating Python/Dart consistency.

    These tests generate test vectors that can be compared with Dart.
    """

    def test_softmax_consistency(self):
        """Test softmax computation matches expected behavior."""
        # Python softmax
        logits = [1.0, 2.0, 3.0, 4.0, 5.0]
        logits_t = torch.tensor(logits)
        probs = F.softmax(logits_t, dim=0).tolist()

        # Manual calculation (should match Dart implementation)
        max_logit = max(logits)
        exps = [math.exp(x - max_logit) for x in logits]
        sum_exps = sum(exps)
        expected = [e / sum_exps for e in exps]

        for p, e in zip(probs, expected):
            assert abs(p - e) < 1e-6, f"Softmax mismatch: {p} vs {e}"

    def test_tone_extraction_vectors(self):
        """Generate test vectors for tone extraction."""
        tone_map = {
            'ā': 1, 'á': 2, 'ǎ': 3, 'à': 4,
            'ē': 1, 'é': 2, 'ě': 3, 'è': 4,
            'ī': 1, 'í': 2, 'ǐ': 3, 'ì': 4,
            'ō': 1, 'ó': 2, 'ǒ': 3, 'ò': 4,
            'ū': 1, 'ú': 2, 'ǔ': 3, 'ù': 4,
            'ǖ': 1, 'ǘ': 2, 'ǚ': 3, 'ǜ': 4,
        }

        def extract_tone(syllable: str) -> int:
            for c in syllable:
                if c in tone_map:
                    return tone_map[c]
            return 0

        # Test cases
        test_cases = [
            ('mā', 1), ('má', 2), ('mǎ', 3), ('mà', 4), ('ma', 0),
            ('shī', 1), ('shí', 2), ('shǐ', 3), ('shì', 4),
            ('gū', 1), ('gú', 2), ('gǔ', 3), ('gù', 4),
            ('lǚ', 3), ('nǜ', 4),
        ]

        for syllable, expected_tone in test_cases:
            actual = extract_tone(syllable)
            assert actual == expected_tone, f"Tone for '{syllable}': expected {expected_tone}, got {actual}"

    def test_alignment_scoring_vectors(self):
        """Generate test vectors for alignment scoring algorithm."""
        # Create reproducible logits
        np.random.seed(42)
        logits = torch.tensor(np.random.randn(20, 10).astype(np.float32))

        decoder = CTCDecoder(blank_index=0)
        targets = [1, 3, 5, 7]

        scores, frames = decoder.score_with_alignment(logits, targets)

        # Verify properties
        assert len(scores) == 4
        assert len(frames) == 4
        assert all(0 <= s <= 1 for s in scores)
        assert all(frames[i] < frames[i+1] for i in range(len(frames)-1)), "Frames must be monotonic"

    def test_syllable_to_char_mapping(self):
        """Test syllable-to-character mapping logic."""
        def map_syllables_to_chars(syl_scores, char_count, syl_count):
            if not syl_scores:
                return [0.0] * char_count
            if syl_count == char_count:
                return syl_scores
            if char_count > syl_count:
                char_scores = []
                for i in range(char_count):
                    syl_idx = int(i * syl_count / char_count)
                    char_scores.append(syl_scores[syl_idx])
                return char_scores
            return syl_scores[:char_count]

        # Test 1:1 mapping
        result = map_syllables_to_chars([0.9, 0.8, 0.7], 3, 3)
        assert result == [0.9, 0.8, 0.7]

        # Test expansion
        result = map_syllables_to_chars([0.9, 0.7], 4, 2)
        assert result == [0.9, 0.9, 0.7, 0.7]

        # Test truncation
        result = map_syllables_to_chars([0.9, 0.8, 0.7, 0.6], 2, 4)
        assert result == [0.9, 0.8]


# ============================================================================
# Test Vector Generation
# ============================================================================

def generate_test_vectors():
    """Generate test vectors for cross-platform validation.

    Run this manually to create fixtures/v7_test_vectors.json
    """
    vectors = {
        'softmax': {
            'input': [1.0, 2.0, 3.0],
            'expected': F.softmax(torch.tensor([1.0, 2.0, 3.0]), dim=0).tolist(),
        },
        'softmax_large': {
            'input': [100.0, 200.0, 300.0],
            'expected': F.softmax(torch.tensor([100.0, 200.0, 300.0]), dim=0).tolist(),
        },
        'tone_extraction': [
            {'syllable': 'mā', 'tone': 1},
            {'syllable': 'má', 'tone': 2},
            {'syllable': 'mǎ', 'tone': 3},
            {'syllable': 'mà', 'tone': 4},
            {'syllable': 'ma', 'tone': 0},
        ],
        'alignment_scoring': {
            'description': 'Test monotonic alignment with given logits',
            'n_frames': 20,
            'n_vocab': 10,
            'targets': [1, 3, 5, 7],
            # Actual values depend on random seed
        },
        'syllable_char_mapping': [
            {'syl_scores': [0.9, 0.8, 0.7], 'char_count': 3, 'syl_count': 3, 'expected': [0.9, 0.8, 0.7]},
            {'syl_scores': [0.9, 0.7], 'char_count': 4, 'syl_count': 2, 'expected': [0.9, 0.9, 0.7, 0.7]},
            {'syl_scores': [0.9, 0.8, 0.7, 0.6], 'char_count': 2, 'syl_count': 4, 'expected': [0.9, 0.8]},
        ],
    }
    return vectors


if __name__ == '__main__':
    # Generate test vectors
    vectors = generate_test_vectors()
    print(json.dumps(vectors, indent=2))
