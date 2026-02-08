"""Comprehensive tests for Mandarin tone sandhi rules.

Tests cover:
- Third tone sandhi (consecutive 3rd tones)
- Yi (一) tone change rules
- Bu (不) tone change rules
- Neutral tone handling
- Edge cases
"""

import pytest
from typing import cast

from mandarin_grader.types import TargetSyllable, Tone, Ms
from mandarin_grader.sandhi import apply_tone_sandhi


def syl(
    pinyin: str,
    tone: int,
    hanzi: str = "",
    index: int = 0,
    initial: str = "",
    final: str = "",
) -> TargetSyllable:
    """Create a TargetSyllable for testing.

    Args:
        pinyin: The pinyin spelling (without tone number).
        tone: The tone number (0-4).
        hanzi: The Chinese character (optional).
        index: The syllable index (optional, defaults to 0).
        initial: The initial consonant (optional).
        final: The final vowel part (optional).

    Returns:
        A TargetSyllable with both underlying and surface tones set to the given tone.
    """
    return TargetSyllable(
        index=index,
        hanzi=hanzi,
        pinyin=pinyin,
        initial=initial,
        final=final,
        tone_underlying=cast(Tone, tone),
        tone_surface=cast(Tone, tone),
    )


class TestThirdToneSandhi:
    """Tests for the third tone sandhi rule.

    Rule: When two or more 3rd tones are consecutive, all but the last become 2nd tone.
    """

    def test_pair_becomes_second_third(self):
        """ni3 hao3 -> ni2 hao3"""
        syllables = [
            syl("ni", 3, "你", index=0),
            syl("hao", 3, "好", index=1),
        ]
        result = apply_tone_sandhi(syllables)

        assert len(result) == 2
        assert result[0].tone_surface == 2  # ni changes to 2nd tone
        assert result[1].tone_surface == 3  # hao stays 3rd tone
        # Underlying tones should be unchanged
        assert result[0].tone_underlying == 3
        assert result[1].tone_underlying == 3

    def test_chain_of_three(self):
        """wo3 xiang3 mai3 -> wo2 xiang2 mai3"""
        syllables = [
            syl("wo", 3, "我", index=0),
            syl("xiang", 3, "想", index=1),
            syl("mai", 3, "买", index=2),
        ]
        result = apply_tone_sandhi(syllables)

        assert len(result) == 3
        assert result[0].tone_surface == 2  # wo changes to 2nd tone
        assert result[1].tone_surface == 2  # xiang changes to 2nd tone
        assert result[2].tone_surface == 3  # mai stays 3rd tone

    def test_chain_of_four(self):
        """Four consecutive 3rd tones: first three become 2nd, last stays 3rd."""
        syllables = [
            syl("ni", 3, index=0),
            syl("hao", 3, index=1),
            syl("wo", 3, index=2),
            syl("xiang", 3, index=3),
        ]
        result = apply_tone_sandhi(syllables)

        assert result[0].tone_surface == 2
        assert result[1].tone_surface == 2
        assert result[2].tone_surface == 2
        assert result[3].tone_surface == 3

    def test_single_third_unchanged(self):
        """Single 3rd tone stays 3rd."""
        syllables = [syl("hao", 3, "好")]
        result = apply_tone_sandhi(syllables)

        assert len(result) == 1
        assert result[0].tone_surface == 3

    def test_third_not_adjacent_unchanged(self):
        """ni3 ... hao3 with gap stays unchanged."""
        syllables = [
            syl("ni", 3, "你", index=0),
            syl("shi", 4, "是", index=1),  # 4th tone breaks the sequence
            syl("hao", 3, "好", index=2),
        ]
        result = apply_tone_sandhi(syllables)

        assert len(result) == 3
        assert result[0].tone_surface == 3  # stays 3rd (not consecutive)
        assert result[1].tone_surface == 4
        assert result[2].tone_surface == 3  # stays 3rd (not consecutive)

    def test_two_pairs_of_third_tones(self):
        """Two separate pairs of 3rd tones, split by non-3rd tone."""
        syllables = [
            syl("ni", 3, index=0),
            syl("hao", 3, index=1),
            syl("shi", 4, index=2),
            syl("wo", 3, index=3),
            syl("de", 3, index=4),
        ]
        result = apply_tone_sandhi(syllables)

        assert result[0].tone_surface == 2  # first pair: ni becomes 2nd
        assert result[1].tone_surface == 3  # first pair: hao stays 3rd
        assert result[2].tone_surface == 4
        assert result[3].tone_surface == 2  # second pair: wo becomes 2nd
        assert result[4].tone_surface == 3  # second pair: de stays 3rd

    def test_non_third_tones_unchanged(self):
        """Non-3rd tone syllables remain unchanged."""
        syllables = [
            syl("ta", 1, index=0),
            syl("shi", 4, index=1),
            syl("lao", 2, index=2),
        ]
        result = apply_tone_sandhi(syllables)

        assert result[0].tone_surface == 1
        assert result[1].tone_surface == 4
        assert result[2].tone_surface == 2


class TestYiRule:
    """Tests for yi (一) tone sandhi rule.

    Rules:
    - Before 4th tone: yi1 -> yi2
    - Before 1st, 2nd, 3rd tone: yi1 -> yi4
    """

    def test_yi_before_fourth_tone(self):
        """yi1 ge4 -> yi2 ge4"""
        syllables = [
            syl("yi", 1, "一", index=0),
            syl("ge", 4, "个", index=1),
        ]
        result = apply_tone_sandhi(syllables)

        assert result[0].tone_surface == 2  # yi changes to 2nd before 4th
        assert result[1].tone_surface == 4

    def test_yi_before_first_tone(self):
        """yi1 tian1 -> yi4 tian1"""
        syllables = [
            syl("yi", 1, "一", index=0),
            syl("tian", 1, "天", index=1),
        ]
        result = apply_tone_sandhi(syllables)

        assert result[0].tone_surface == 4  # yi changes to 4th before 1st
        assert result[1].tone_surface == 1

    def test_yi_before_second_tone(self):
        """yi1 nian2 -> yi4 nian2"""
        syllables = [
            syl("yi", 1, "一", index=0),
            syl("nian", 2, "年", index=1),
        ]
        result = apply_tone_sandhi(syllables)

        assert result[0].tone_surface == 4  # yi changes to 4th before 2nd
        assert result[1].tone_surface == 2

    def test_yi_before_third_tone(self):
        """yi1 qi3 -> yi4 qi3"""
        syllables = [
            syl("yi", 1, "一", index=0),
            syl("qi", 3, "起", index=1),
        ]
        result = apply_tone_sandhi(syllables)

        assert result[0].tone_surface == 4  # yi changes to 4th before 3rd
        assert result[1].tone_surface == 3

    def test_yi_at_end_unchanged(self):
        """yi1 at end stays yi1."""
        syllables = [
            syl("di", 4, "第", index=0),
            syl("yi", 1, "一", index=1),
        ]
        result = apply_tone_sandhi(syllables)

        assert result[0].tone_surface == 4
        assert result[1].tone_surface == 1  # yi at end stays 1st

    def test_yi_before_neutral_tone(self):
        """yi1 before neutral tone (0) stays yi1."""
        syllables = [
            syl("yi", 1, "一", index=0),
            syl("ge", 0, "个", index=1),  # neutral tone
        ]
        result = apply_tone_sandhi(syllables)

        assert result[0].tone_surface == 1  # yi stays 1st before neutral
        assert result[1].tone_surface == 0

    def test_yi_with_wrong_underlying_tone(self):
        """yi with underlying tone != 1 is not affected."""
        syllables = [
            syl("yi", 4, "意", index=0),  # yi4 (meaning)
            syl("si", 1, "思", index=1),
        ]
        # Create with tone_underlying = 4
        result = apply_tone_sandhi(syllables)

        assert result[0].tone_surface == 4  # Not affected since underlying != 1
        assert result[1].tone_surface == 1

    def test_yi_case_insensitive(self):
        """yi matching should be case insensitive."""
        # Uppercase 'YI' should also be matched
        syllable_upper = TargetSyllable(
            index=0,
            hanzi="一",
            pinyin="YI",  # uppercase
            initial="",
            final="i",
            tone_underlying=cast(Tone, 1),
            tone_surface=cast(Tone, 1),
        )
        syllables = [syllable_upper, syl("ge", 4, index=1)]
        result = apply_tone_sandhi(syllables)

        assert result[0].tone_surface == 2  # should still apply rule


class TestBuRule:
    """Tests for bu (不) tone sandhi rule.

    Rule: bu4 before 4th tone becomes bu2.
    """

    def test_bu_before_fourth(self):
        """bu4 shi4 -> bu2 shi4"""
        syllables = [
            syl("bu", 4, "不", index=0),
            syl("shi", 4, "是", index=1),
        ]
        result = apply_tone_sandhi(syllables)

        assert result[0].tone_surface == 2  # bu changes to 2nd before 4th
        assert result[1].tone_surface == 4

    def test_bu_before_first_unchanged(self):
        """bu4 before 1st tone stays bu4."""
        syllables = [
            syl("bu", 4, "不", index=0),
            syl("zhi", 1, "知", index=1),
        ]
        result = apply_tone_sandhi(syllables)

        assert result[0].tone_surface == 4  # bu stays 4th
        assert result[1].tone_surface == 1

    def test_bu_before_second_unchanged(self):
        """bu4 before 2nd tone stays bu4."""
        syllables = [
            syl("bu", 4, "不", index=0),
            syl("xing", 2, "行", index=1),
        ]
        result = apply_tone_sandhi(syllables)

        assert result[0].tone_surface == 4
        assert result[1].tone_surface == 2

    def test_bu_before_third_unchanged(self):
        """bu4 before 3rd tone stays bu4."""
        syllables = [
            syl("bu", 4, "不", index=0),
            syl("hao", 3, "好", index=1),
        ]
        result = apply_tone_sandhi(syllables)

        assert result[0].tone_surface == 4
        assert result[1].tone_surface == 3

    def test_bu_at_end_unchanged(self):
        """bu4 at end stays bu4."""
        syllables = [
            syl("shi", 4, "是", index=0),
            syl("bu", 4, "不", index=1),
        ]
        result = apply_tone_sandhi(syllables)

        assert result[0].tone_surface == 4
        assert result[1].tone_surface == 4  # bu at end stays 4th

    def test_bu_before_neutral_unchanged(self):
        """bu4 before neutral tone stays bu4."""
        syllables = [
            syl("bu", 4, "不", index=0),
            syl("de", 0, "的", index=1),
        ]
        result = apply_tone_sandhi(syllables)

        assert result[0].tone_surface == 4
        assert result[1].tone_surface == 0

    def test_bu_case_insensitive(self):
        """bu matching should be case insensitive."""
        syllable_upper = TargetSyllable(
            index=0,
            hanzi="不",
            pinyin="BU",  # uppercase
            initial="b",
            final="u",
            tone_underlying=cast(Tone, 4),
            tone_surface=cast(Tone, 4),
        )
        syllables = [syllable_upper, syl("shi", 4, index=1)]
        result = apply_tone_sandhi(syllables)

        assert result[0].tone_surface == 2  # should still apply rule


class TestNeutralTone:
    """Tests for neutral tone (tone 0) handling."""

    def test_neutral_tone_preserved(self):
        """de0 stays de0."""
        syllables = [syl("de", 0, "的")]
        result = apply_tone_sandhi(syllables)

        assert len(result) == 1
        assert result[0].tone_surface == 0

    def test_neutral_after_third(self):
        """wo3 de0 - wo stays wo3 (no consecutive 3rd tones)."""
        syllables = [
            syl("wo", 3, "我", index=0),
            syl("de", 0, "的", index=1),
        ]
        result = apply_tone_sandhi(syllables)

        assert result[0].tone_surface == 3  # wo stays 3rd (no consecutive 3rd)
        assert result[1].tone_surface == 0

    def test_neutral_does_not_trigger_third_tone_sandhi(self):
        """Neutral tone does not count as 3rd tone for sandhi."""
        syllables = [
            syl("hao", 3, index=0),
            syl("de", 0, index=1),
            syl("hen", 3, index=2),
        ]
        result = apply_tone_sandhi(syllables)

        # No consecutive 3rd tones - neutral in between
        assert result[0].tone_surface == 3
        assert result[1].tone_surface == 0
        assert result[2].tone_surface == 3

    def test_multiple_neutral_tones(self):
        """Multiple neutral tones in sequence."""
        syllables = [
            syl("de", 0, index=0),
            syl("le", 0, index=1),
            syl("ma", 0, index=2),
        ]
        result = apply_tone_sandhi(syllables)

        assert all(s.tone_surface == 0 for s in result)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_list(self):
        """Empty list returns empty."""
        result = apply_tone_sandhi([])
        assert result == []

    def test_single_syllable(self):
        """Single syllable unchanged."""
        syllables = [syl("ma", 1, "妈")]
        result = apply_tone_sandhi(syllables)

        assert len(result) == 1
        assert result[0].tone_surface == 1
        assert result[0].pinyin == "ma"

    def test_original_list_not_modified(self):
        """Original list should not be modified."""
        syllables = [
            syl("ni", 3, index=0),
            syl("hao", 3, index=1),
        ]
        original_tones = [s.tone_surface for s in syllables]

        apply_tone_sandhi(syllables)

        # Original should be unchanged
        assert [s.tone_surface for s in syllables] == original_tones

    def test_all_data_preserved(self):
        """All syllable data except tone_surface should be preserved."""
        syllable = TargetSyllable(
            index=5,
            hanzi="你",
            pinyin="ni",
            initial="n",
            final="i",
            tone_underlying=cast(Tone, 3),
            tone_surface=cast(Tone, 3),
            start_expected_ms=cast(Ms, 100),
            end_expected_ms=cast(Ms, 200),
        )
        syllables = [syllable, syl("hao", 3, index=1)]
        result = apply_tone_sandhi(syllables)

        # Check all fields preserved except tone_surface
        assert result[0].index == 5
        assert result[0].hanzi == "你"
        assert result[0].pinyin == "ni"
        assert result[0].initial == "n"
        assert result[0].final == "i"
        assert result[0].tone_underlying == 3
        assert result[0].start_expected_ms == 100
        assert result[0].end_expected_ms == 200
        # Only tone_surface should change
        assert result[0].tone_surface == 2

    def test_very_long_sentence(self):
        """Test with a longer sentence mixing different tones."""
        # wo3 men2 yi1 qi3 qu4 chi1 fan4 ba0
        syllables = [
            syl("wo", 3, index=0),
            syl("men", 2, index=1),  # breaks 3rd tone sequence
            syl("yi", 1, index=2),
            syl("qi", 3, index=3),
            syl("qu", 4, index=4),
            syl("chi", 1, index=5),
            syl("fan", 4, index=6),
            syl("ba", 0, index=7),
        ]
        result = apply_tone_sandhi(syllables)

        assert result[0].tone_surface == 3  # wo stays 3rd (men2 follows)
        assert result[1].tone_surface == 2  # men unchanged
        assert result[2].tone_surface == 4  # yi1 before qi3 -> yi4
        assert result[3].tone_surface == 3  # qi stays 3rd
        assert result[4].tone_surface == 4  # qu unchanged
        assert result[5].tone_surface == 1  # chi unchanged
        assert result[6].tone_surface == 4  # fan unchanged
        assert result[7].tone_surface == 0  # ba unchanged


class TestCombinedRules:
    """Tests for interactions between multiple sandhi rules."""

    def test_yi_and_third_tone_sandhi(self):
        """yi before multiple 3rd tones."""
        # yi1 xiang3 mai3 -> yi4 xiang2 mai3
        syllables = [
            syl("yi", 1, index=0),
            syl("xiang", 3, index=1),
            syl("mai", 3, index=2),
        ]
        result = apply_tone_sandhi(syllables)

        # yi becomes 4th (before 3rd tone)
        assert result[0].tone_surface == 4
        # Third tone sandhi: xiang becomes 2nd, mai stays 3rd
        assert result[1].tone_surface == 2
        assert result[2].tone_surface == 3

    def test_bu_and_third_tone_sandhi(self):
        """bu followed by 3rd tones."""
        # bu4 xiang3 mai3 -> bu4 xiang2 mai3
        syllables = [
            syl("bu", 4, index=0),
            syl("xiang", 3, index=1),
            syl("mai", 3, index=2),
        ]
        result = apply_tone_sandhi(syllables)

        # bu stays 4th (not before 4th tone)
        assert result[0].tone_surface == 4
        # Third tone sandhi applies
        assert result[1].tone_surface == 2
        assert result[2].tone_surface == 3

    def test_yi_and_bu_in_sequence(self):
        """yi and bu in the same sentence."""
        # yi1 ge4 bu4 shi4 -> yi2 ge4 bu2 shi4
        syllables = [
            syl("yi", 1, index=0),
            syl("ge", 4, index=1),
            syl("bu", 4, index=2),
            syl("shi", 4, index=3),
        ]
        result = apply_tone_sandhi(syllables)

        assert result[0].tone_surface == 2  # yi before 4th
        assert result[1].tone_surface == 4
        assert result[2].tone_surface == 2  # bu before 4th
        assert result[3].tone_surface == 4

    def test_multiple_third_tone_chains_with_yi(self):
        """Complex sentence with multiple rules applying."""
        # ni3 hao3 yi1 ge4 wo3 men5
        syllables = [
            syl("ni", 3, index=0),
            syl("hao", 3, index=1),
            syl("yi", 1, index=2),
            syl("ge", 4, index=3),
            syl("wo", 3, index=4),
            syl("men", 0, index=5),  # neutral tone
        ]
        result = apply_tone_sandhi(syllables)

        # ni3 hao3 -> ni2 hao3
        assert result[0].tone_surface == 2
        assert result[1].tone_surface == 3
        # yi1 ge4 -> yi2 ge4
        assert result[2].tone_surface == 2
        assert result[3].tone_surface == 4
        # wo3 men0 -> no change (neutral doesn't form consecutive 3rd)
        assert result[4].tone_surface == 3
        assert result[5].tone_surface == 0
