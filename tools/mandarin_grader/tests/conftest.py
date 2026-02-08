"""Pytest configuration and fixtures for mandarin_grader tests."""

from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from mandarin_grader.types import (
    FrameTrack,
    Ms,
    PhoneSpan,
    SyllableSpan,
    TargetSyllable,
    Tone,
)


@pytest.fixture
def sample_syllable() -> TargetSyllable:
    """A sample syllable for testing."""
    return TargetSyllable(
        index=0,
        hanzi="你",
        pinyin="ni",
        initial="n",
        final="i",
        tone_underlying=cast(Tone, 3),
        tone_surface=cast(Tone, 3),
        start_expected_ms=Ms(0),
        end_expected_ms=Ms(300),
    )


@pytest.fixture
def sample_frame_track() -> FrameTrack:
    """A sample FrameTrack with realistic F0 values.

    Creates a 100-frame track at 100 Hz (1 second of audio) with:
    - F0 rising from 200 Hz to 250 Hz (typical tone 2 pattern)
    - Voicing probability of 0.9 for all frames
    """
    num_frames = 100
    f0_hz: NDArray[np.floating] = np.linspace(200.0, 250.0, num_frames)
    voicing: NDArray[np.floating] = np.full(num_frames, 0.9)
    return FrameTrack(
        frame_hz=100.0,
        f0_hz=f0_hz,
        voicing=voicing,
        energy=None,
    )


@pytest.fixture
def sample_span() -> SyllableSpan:
    """A sample syllable span for testing."""
    return SyllableSpan(
        index=0,
        start_ms=Ms(100),
        end_ms=Ms(400),
        confidence=0.95,
        phone_spans=(
            PhoneSpan(phone="n", start_ms=Ms(100), end_ms=Ms(150), confidence=0.9),
            PhoneSpan(phone="i", start_ms=Ms(150), end_ms=Ms(400), confidence=0.95),
        ),
    )


@pytest.fixture
def sample_syllables_for_sandhi() -> list[TargetSyllable]:
    """Sample syllables for testing tone sandhi rules.

    Creates ni3 hao3 (你好) which should become ni2 hao3 after sandhi.
    """
    return [
        TargetSyllable(
            index=0,
            hanzi="你",
            pinyin="ni",
            initial="n",
            final="i",
            tone_underlying=cast(Tone, 3),
            tone_surface=cast(Tone, 3),
        ),
        TargetSyllable(
            index=1,
            hanzi="好",
            pinyin="hao",
            initial="h",
            final="ao",
            tone_underlying=cast(Tone, 3),
            tone_surface=cast(Tone, 3),
        ),
    ]
