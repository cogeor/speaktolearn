"""Tone sandhi rule detection and handling.

This module implements Mandarin Chinese tone sandhi (tone change) rules:
1. Third tone sandhi: consecutive 3rd tones change to 2nd except the last
2. Yi (一) rules: tone changes based on following syllable
3. Bu (不) rule: changes to 2nd tone before 4th tone
"""

from dataclasses import replace
from typing import cast

from .types import TargetSyllable, Tone


def apply_tone_sandhi(targets: list[TargetSyllable]) -> list[TargetSyllable]:
    """Apply all tone sandhi rules to a list of target syllables.

    Applies rules in order:
    1. Third tone sandhi
    2. Yi (一) rule
    3. Bu (不) rule

    Args:
        targets: List of target syllables with underlying tones set.

    Returns:
        New list with tone_surface updated according to sandhi rules.
        The original list is not modified.
    """
    if not targets:
        return []

    result = _apply_3rd_tone_sandhi(targets)
    result = _apply_yi_rule(result)
    result = _apply_bu_rule(result)
    return result


def _apply_3rd_tone_sandhi(targets: list[TargetSyllable]) -> list[TargetSyllable]:
    """Apply third tone sandhi rule.

    When two or more 3rd tones are consecutive, all but the last become 2nd tone.

    Examples:
        ni3 hao3 -> ni2 hao3
        wo3 xiang3 mai3 -> wo2 xiang2 mai3

    Args:
        targets: List of target syllables.

    Returns:
        New list with third tone sandhi applied to tone_surface.
    """
    if len(targets) <= 1:
        return list(targets)

    result: list[TargetSyllable] = []

    # Find consecutive runs of 3rd tones
    i = 0
    while i < len(targets):
        syllable = targets[i]

        # Check if this starts a run of 3rd tones (using surface tone)
        if syllable.tone_surface == 3:
            # Find the end of the consecutive 3rd tone run
            run_start = i
            run_end = i
            while run_end + 1 < len(targets) and targets[run_end + 1].tone_surface == 3:
                run_end += 1

            # If we have 2 or more consecutive 3rd tones
            if run_end > run_start:
                # Change all but the last to 2nd tone
                for j in range(run_start, run_end):
                    modified = replace(targets[j], tone_surface=cast(Tone, 2))
                    result.append(modified)
                # Keep the last one as 3rd tone
                result.append(targets[run_end])
                i = run_end + 1
            else:
                # Single 3rd tone, no change
                result.append(syllable)
                i += 1
        else:
            result.append(syllable)
            i += 1

    return result


def _apply_yi_rule(targets: list[TargetSyllable]) -> list[TargetSyllable]:
    """Apply yi (一) tone sandhi rule.

    Yi (一) tone changes based on the following syllable:
    - Before 4th tone: yi1 -> yi2 (e.g., yi1 ge4 -> yi2 ge4)
    - Before 1st, 2nd, 3rd tone: yi1 -> yi4 (e.g., yi1 tian1 -> yi4 tian1)

    Only applies when the underlying tone is 1 (the citation tone of 一).
    Neutral tone (0) following does not trigger a change.

    Args:
        targets: List of target syllables.

    Returns:
        New list with yi rule applied to tone_surface.
    """
    if len(targets) <= 1:
        return list(targets)

    result: list[TargetSyllable] = []

    for i, syllable in enumerate(targets):
        # Check if this is yi with underlying tone 1
        if syllable.pinyin.lower() == "yi" and syllable.tone_underlying == 1:
            # Check if there's a following syllable
            if i + 1 < len(targets):
                next_tone = targets[i + 1].tone_surface
                if next_tone == 4:
                    # Before 4th tone: yi1 -> yi2
                    modified = replace(syllable, tone_surface=cast(Tone, 2))
                    result.append(modified)
                elif next_tone in (1, 2, 3):
                    # Before 1st, 2nd, 3rd tone: yi1 -> yi4
                    modified = replace(syllable, tone_surface=cast(Tone, 4))
                    result.append(modified)
                else:
                    # Neutral tone (0) or other: no change
                    result.append(syllable)
            else:
                # No following syllable, no change
                result.append(syllable)
        else:
            result.append(syllable)

    return result


def _apply_bu_rule(targets: list[TargetSyllable]) -> list[TargetSyllable]:
    """Apply bu (不) tone sandhi rule.

    Bu (不) changes to 2nd tone before a 4th tone:
    - Before 4th tone: bu4 -> bu2 (e.g., bu4 shi4 -> bu2 shi4)

    Only applies when the underlying tone is 4 (the citation tone of 不).

    Args:
        targets: List of target syllables.

    Returns:
        New list with bu rule applied to tone_surface.
    """
    if len(targets) <= 1:
        return list(targets)

    result: list[TargetSyllable] = []

    for i, syllable in enumerate(targets):
        # Check if this is bu with underlying tone 4
        if syllable.pinyin.lower() == "bu" and syllable.tone_underlying == 4:
            # Check if there's a following syllable with 4th tone
            if i + 1 < len(targets) and targets[i + 1].tone_surface == 4:
                # Before 4th tone: bu4 -> bu2
                modified = replace(syllable, tone_surface=cast(Tone, 2))
                result.append(modified)
            else:
                result.append(syllable)
        else:
            result.append(syllable)

    return result
