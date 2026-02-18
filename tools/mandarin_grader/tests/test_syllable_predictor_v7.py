"""Tests for SyllablePredictorV7 (CTC-based model)."""

import pytest
import numpy as np

torch = pytest.importorskip("torch")

from mandarin_grader.model.syllable_predictor_v7 import (
    SyllablePredictorV7,
    SyllablePredictorConfigV7,
    CTCDecoder,
    PredictorOutputV7,
)


class TestSyllablePredictorConfigV7:
    """Tests for V7 config."""

    def test_default_config(self):
        config = SyllablePredictorConfigV7()
        assert config.n_mels == 80
        assert config.sample_rate == 16000
        assert config.blank_index == 0
        assert config.n_tones == 5
        assert config.d_model == 192
        assert config.n_layers == 4

    def test_custom_config(self):
        config = SyllablePredictorConfigV7(
            d_model=256,
            n_layers=6,
            max_audio_frames=2000,
        )
        assert config.d_model == 256
        assert config.n_layers == 6
        assert config.max_audio_frames == 2000


class TestSyllablePredictorV7:
    """Tests for V7 model forward pass."""

    @pytest.fixture
    def model(self):
        config = SyllablePredictorConfigV7()
        return SyllablePredictorV7(config)

    def test_forward_basic(self, model):
        """Test basic forward pass."""
        mel = torch.randn(2, 80, 400)  # 4 seconds
        syl_logits, tone_logits = model(mel)

        # Output shape: [batch, time//4, vocab+1]
        expected_time = 400 // 4  # CNN 4x downsampling
        assert syl_logits.shape == (2, expected_time, model.config.n_syllables + 1)
        assert tone_logits.shape == (2, expected_time, model.config.n_tones + 1)

    def test_forward_variable_lengths(self, model):
        """Test forward pass with various input lengths."""
        for time_frames in [100, 300, 500, 800, 1000]:
            mel = torch.randn(1, 80, time_frames)
            syl_logits, tone_logits = model(mel)

            expected_time = (time_frames + 3) // 4
            assert syl_logits.shape[1] == expected_time
            assert tone_logits.shape[1] == expected_time

    def test_forward_with_mask(self, model):
        """Test forward with audio padding mask."""
        mel = torch.randn(2, 80, 400)
        audio_mask = torch.zeros(2, 400, dtype=torch.bool)
        audio_mask[0, 300:] = True  # Sample 0 has only 300 frames
        audio_mask[1, 350:] = True  # Sample 1 has only 350 frames

        syl_logits, tone_logits = model(mel, audio_mask)

        assert syl_logits.shape[0] == 2
        assert tone_logits.shape[0] == 2

    def test_predict(self, model):
        """Test predict method with CTC decoding."""
        mel = torch.randn(1, 80, 400)
        output = model.predict(mel)

        assert isinstance(output, PredictorOutputV7)
        assert output.syllable_logits is not None
        assert output.tone_logits is not None
        assert output.syllable_ids is not None
        assert output.tone_ids is not None
        assert len(output.syllable_ids) == 1  # Batch size 1

    def test_get_input_lengths(self, model):
        """Test input length computation for CTC loss."""
        mel_lengths = torch.tensor([100, 200, 400, 800])
        output_lengths = model.get_input_lengths(mel_lengths)

        # CNN downsampling is 4x
        expected = (mel_lengths + 3) // 4
        assert torch.all(output_lengths == expected)

    def test_count_parameters(self, model):
        """Test parameter counting."""
        total, trainable = model.count_parameters()
        assert total > 0
        assert trainable == total  # All params should be trainable


class TestCTCDecoder:
    """Tests for CTC decoder."""

    @pytest.fixture
    def decoder(self):
        return CTCDecoder(blank_index=0)

    def test_greedy_decode_simple(self, decoder):
        """Test greedy decoding with simple sequence."""
        # Logits: [batch=1, time=5, vocab=4]
        # Set up: blank=0, classes=1,2,3
        logits = torch.zeros(1, 5, 4)
        # Frame 0: class 1
        logits[0, 0, 1] = 10.0
        # Frame 1: class 1 (repeat)
        logits[0, 1, 1] = 10.0
        # Frame 2: blank
        logits[0, 2, 0] = 10.0
        # Frame 3: class 2
        logits[0, 3, 2] = 10.0
        # Frame 4: class 3
        logits[0, 4, 3] = 10.0

        decoded = decoder.greedy_decode(logits)

        # After collapse: [1, blank, 2, 3] -> remove blank -> [1, 2, 3]
        assert decoded == [[1, 2, 3]]

    def test_greedy_decode_all_blank(self, decoder):
        """Test decoding when all frames are blank."""
        logits = torch.zeros(1, 5, 4)
        logits[:, :, 0] = 10.0  # All blank

        decoded = decoder.greedy_decode(logits)
        assert decoded == [[]]

    def test_greedy_decode_batch(self, decoder):
        """Test batch decoding."""
        logits = torch.zeros(2, 3, 4)
        # Batch 0: [1]
        logits[0, 0, 1] = 10.0
        logits[0, 1, 0] = 10.0  # blank
        logits[0, 2, 0] = 10.0  # blank

        # Batch 1: [2, 3]
        logits[1, 0, 2] = 10.0
        logits[1, 1, 3] = 10.0
        logits[1, 2, 0] = 10.0  # blank

        decoded = decoder.greedy_decode(logits)
        assert decoded == [[1], [2, 3]]

    def test_decode_with_probs(self, decoder):
        """Test decoding with probability extraction."""
        logits = torch.zeros(1, 3, 4)
        logits[0, 0, 1] = 10.0
        logits[0, 1, 0] = 10.0  # blank
        logits[0, 2, 2] = 10.0

        seqs, probs = decoder.decode_with_probs(logits)

        assert seqs == [[1, 2]]
        assert len(probs[0]) == 2
        # Probabilities should be close to 1.0 (softmax of 10 vs 0)
        assert all(p > 0.99 for p in probs[0])

    def test_score_against_targets(self, decoder):
        """Test target-based scoring."""
        # Logits: [time=4, vocab=5]
        logits = torch.zeros(4, 5)
        logits[0, 1] = 5.0  # Target 1 at frame 0
        logits[1, 2] = 3.0  # Target 2 at frame 1
        logits[2, 1] = 7.0  # Target 1 at frame 2 (higher)
        logits[3, 3] = 4.0  # Target 3 at frame 3

        targets = [1, 2, 3]
        scores = decoder.score_against_targets(logits, targets)

        assert len(scores) == 3
        # Target 1 should have high score (appears at frames 0 and 2)
        assert scores[0] > 0.5
        # Target 2 appears at frame 1
        assert scores[1] > 0.3
        # Target 3 appears at frame 3
        assert scores[2] > 0.3

    def test_score_with_alignment(self, decoder):
        """Test alignment-based scoring."""
        # Logits: [time=6, vocab=4]
        logits = torch.zeros(6, 4)
        # Target sequence: [1, 2, 1]
        logits[0, 1] = 5.0  # First target at frame 0
        logits[2, 2] = 4.0  # Second target at frame 2
        logits[4, 1] = 6.0  # Third target at frame 4

        targets = [1, 2, 1]
        scores, frames = decoder.score_with_alignment(logits, targets)

        assert len(scores) == 3
        assert len(frames) == 3
        # Frames should be monotonically increasing
        assert frames[0] < frames[1] < frames[2]

    def test_score_with_alignment_maintains_order(self, decoder):
        """Test that alignment respects temporal order."""
        logits = torch.zeros(4, 3)
        # Target 1 has max at frame 3
        logits[3, 1] = 10.0
        # Target 2 has max at frame 1
        logits[1, 2] = 10.0

        # But alignment must be monotonic: target 1 first, then target 2
        targets = [1, 2]
        scores, frames = decoder.score_with_alignment(logits, targets)

        # First target gets frame 3 (best for target 1)
        # Second target must come AFTER frame 3, but frame 1 is not allowed
        assert frames[0] < frames[1] or frames[1] >= frames[0] + 1


class TestV7Integration:
    """Integration tests for V7 model and decoder."""

    def test_full_pipeline(self):
        """Test full prediction pipeline."""
        config = SyllablePredictorConfigV7()
        model = SyllablePredictorV7(config)

        # Simulate mel input
        mel = torch.randn(1, 80, 300)

        # Forward pass
        syl_logits, tone_logits = model(mel)

        # CTC decode
        decoder = model.ctc_decoder
        syl_decoded = decoder.greedy_decode(syl_logits)
        tone_decoded = decoder.greedy_decode(tone_logits)

        assert isinstance(syl_decoded, list)
        assert isinstance(tone_decoded, list)

    def test_scoring_pipeline(self):
        """Test scoring against target sequence."""
        config = SyllablePredictorConfigV7()
        model = SyllablePredictorV7(config)

        mel = torch.randn(1, 80, 400)

        with torch.no_grad():
            syl_logits, tone_logits = model(mel)

        # Score against targets
        target_syls = [10, 20, 30]  # Fake syllable IDs
        target_tones = [1, 2, 3]  # Tone IDs

        syl_scores = model.ctc_decoder.score_against_targets(syl_logits[0], target_syls)
        tone_scores = model.ctc_decoder.score_against_targets(tone_logits[0], target_tones)

        assert len(syl_scores) == 3
        assert len(tone_scores) == 3
        # All scores should be probabilities
        assert all(0 <= s <= 1 for s in syl_scores)
        assert all(0 <= s <= 1 for s in tone_scores)
