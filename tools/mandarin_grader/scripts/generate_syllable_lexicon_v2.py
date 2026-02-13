#!/usr/bin/env python3
"""Generate complete syllable lexicon with all tones for DL training.

Improvements over v1:
- Generates ALL valid Mandarin syllables (not just from dataset)
- Generates all 5 tones (1-4 + neutral) in one prompt, then splits
- Uses 4 voices (2 female, 2 male)
- Organizes as: voice/tone{N}/syllable.wav

Usage:
    python generate_syllable_lexicon_v2.py --output data/syllables_v2
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import wave
import struct
from pathlib import Path

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Voice configurations - 2 female, 2 male
VOICES = {
    "female1": "zh-CN-XiaoxiaoNeural",   # News, Warm - clear pronunciation
    "female2": "zh-CN-XiaoyiNeural",     # Cartoon, Lively
    "male1": "zh-CN-YunyangNeural",      # News, Professional - clear
    "male2": "zh-CN-YunxiNeural",        # Novel, Lively
}

# Complete list of valid Mandarin syllables (initials + finals)
# Based on standard Pinyin chart

INITIALS = [
    "",  # Zero initial (for syllables starting with vowels)
    "b", "p", "m", "f",
    "d", "t", "n", "l",
    "g", "k", "h",
    "j", "q", "x",
    "zh", "ch", "sh", "r",
    "z", "c", "s",
    "y", "w",
]

# Finals organized by initial compatibility
FINALS_GENERAL = [
    "a", "o", "e", "ai", "ei", "ao", "ou",
    "an", "en", "ang", "eng", "ong",
]

FINALS_I = [
    "i", "ia", "ie", "iao", "iu", "ian", "in", "iang", "ing", "iong",
]

FINALS_U = [
    "u", "ua", "uo", "uai", "ui", "uan", "un", "uang", "ueng",
]

FINALS_V = [
    "ü", "üe", "üan", "ün",  # Used with j, q, x, y
]

# Special standalone syllables
STANDALONE_SYLLABLES = [
    "a", "o", "e", "ai", "ei", "ao", "ou", "an", "en", "ang", "eng", "er",
    "yi", "ya", "ye", "yao", "you", "yan", "yin", "yang", "ying", "yong",
    "wu", "wa", "wo", "wai", "wei", "wan", "wen", "wang", "weng",
    "yu", "yue", "yuan", "yun",
]

def get_all_valid_syllables() -> list[str]:
    """Generate list of all valid Mandarin syllables."""
    syllables = set()

    # Add standalone syllables
    syllables.update(STANDALONE_SYLLABLES)

    # Generate combinations based on Pinyin rules
    for initial in INITIALS:
        if initial == "":
            continue  # Standalone handled above

        # General finals work with most initials
        if initial in ["b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "zh", "ch", "sh", "r", "z", "c", "s"]:
            for final in FINALS_GENERAL:
                syllables.add(initial + final)
            for final in FINALS_U:
                syllables.add(initial + final)

        # i-finals with specific initials
        if initial in ["b", "p", "m", "d", "t", "n", "l", "j", "q", "x"]:
            for final in FINALS_I:
                syllables.add(initial + final)

        # ü-finals only with j, q, x (written as u)
        if initial in ["j", "q", "x"]:
            for final in FINALS_V:
                # j, q, x + ü is written as ju, qu, xu
                syllables.add(initial + final.replace("ü", "u"))

        # n, l can use ü
        if initial in ["n", "l"]:
            for final in FINALS_V:
                syllables.add(initial + final)

    # Add common syllables that might be missed
    common_additions = [
        "zhi", "chi", "shi", "ri", "zi", "ci", "si",  # Special i sound
        "ju", "qu", "xu", "lü", "nü",  # ü syllables
        "jue", "que", "xue", "lüe", "nüe",
        "juan", "quan", "xuan", "lüan",
        "jun", "qun", "xun", "lün",
        "jiong", "qiong", "xiong",
        "zhuang", "chuang", "shuang", "guang", "kuang", "huang",
        "zhuan", "chuan", "shuan", "ruan", "guan", "kuan", "huan", "zuan", "cuan", "suan",
        "zhui", "chui", "shui", "rui", "gui", "kui", "hui", "zui", "cui", "sui", "dui", "tui",
        "zhun", "chun", "shun", "run", "gun", "kun", "hun", "zun", "cun", "sun", "dun", "tun", "lun",
        "zhuo", "chuo", "shuo", "ruo", "guo", "kuo", "huo", "zuo", "cuo", "suo", "duo", "tuo", "nuo", "luo",
        "lia", "niu", "liu", "miu", "diu",
        "niang", "liang",
        "bing", "ping", "ming", "ding", "ting", "ning", "ling", "jing", "qing", "xing",
        "biao", "piao", "miao", "diao", "tiao", "niao", "liao", "jiao", "qiao", "xiao",
        "bian", "pian", "mian", "dian", "tian", "nian", "lian", "jian", "qian", "xian",
        "nüe", "lüe",
        "me", "ne", "le", "ge", "ke", "he", "zhe", "che", "she", "re", "ze", "ce", "se", "de", "te",
        "fo", "mo", "bo", "po", "lo",
        "ceng", "seng", "zeng", "deng", "teng", "neng", "leng", "geng", "keng", "heng", "zheng", "cheng", "sheng", "reng",
        "ca", "za", "sa",
    ]
    syllables.update(common_additions)

    # Remove invalid combinations
    invalid = {
        "buo", "puo", "muo", "fuo",  # These don't exist
        "fei", "gei", "kei", "hei", "zhei", "shei",  # Some don't exist
    }
    syllables -= invalid

    return sorted(syllables)


# Tone marks for pinyin
TONE_MARKS = {
    "a": ["ā", "á", "ǎ", "à"],
    "e": ["ē", "é", "ě", "è"],
    "i": ["ī", "í", "ǐ", "ì"],
    "o": ["ō", "ó", "ǒ", "ò"],
    "u": ["ū", "ú", "ǔ", "ù"],
    "ü": ["ǖ", "ǘ", "ǚ", "ǜ"],
}


def add_tone_mark(pinyin: str, tone: int) -> str:
    """Add tone mark to pinyin. Tone 0 = no mark (neutral)."""
    if tone == 0 or tone > 4:
        return pinyin

    # Find vowel to mark (rules: a/e first, then last vowel, ou marks o)
    vowels = "aeiouü"
    pinyin_lower = pinyin.lower()

    mark_idx = -1

    # Rule 1: a or e gets the mark
    for i, char in enumerate(pinyin_lower):
        if char in "ae":
            mark_idx = i
            break

    # Rule 2: ou -> mark on o
    if mark_idx == -1 and "ou" in pinyin_lower:
        mark_idx = pinyin_lower.find("o")

    # Rule 3: last vowel
    if mark_idx == -1:
        for i in range(len(pinyin_lower) - 1, -1, -1):
            if pinyin_lower[i] in vowels:
                mark_idx = i
                break

    if mark_idx == -1:
        return pinyin

    char = pinyin_lower[mark_idx]
    if char in TONE_MARKS:
        marked = TONE_MARKS[char][tone - 1]
        return pinyin[:mark_idx] + marked + pinyin[mark_idx + 1:]

    return pinyin


def create_tone_prompt(syllable: str) -> str:
    """Create prompt with tones 1-4 clearly separated.

    Format: "tone1. tone2. tone3. tone4."
    Neutral tone (0) is generated separately.
    """
    tones = []
    for tone in [1, 2, 3, 4]:
        marked = add_tone_mark(syllable, tone)
        tones.append(marked)

    # Join with clear separators
    return ". ".join(tones) + "."


# Common neutral tone characters for better TTS pronunciation
# Maps pinyin to Chinese character that is commonly pronounced with neutral tone
NEUTRAL_TONE_CHARS = {
    # Particles
    "a": "啊", "ba": "吧", "la": "啦", "ma": "吗", "na": "呐",
    "ya": "呀", "wa": "哇",
    "de": "的", "le": "了", "ne": "呢",
    "me": "么", "ge": "个",
    # Suffixes
    "men": "们", "zi": "子", "tou": "头",
    # Common unstressed
    "zhe": "着", "guo": "过",
    "shang": "上", "xia": "下", "li": "里", "bian": "边",
}


def get_neutral_tone_prompt(syllable: str) -> str:
    """Get a prompt for neutral tone that TTS will pronounce correctly.

    Uses Chinese characters when available, otherwise uses carrier phrase.
    """
    # If we have a known character, use it directly
    if syllable in NEUTRAL_TONE_CHARS:
        return NEUTRAL_TONE_CHARS[syllable]

    # For other syllables, use a carrier phrase with tone 1 as prefix
    # "妈X" - using 妈 (mā) as carrier since it's clear and common
    # The TTS will read the second syllable in neutral/unstressed context
    tone1 = add_tone_mark(syllable, 1)
    return f"说{tone1}"  # "say [syllable]" - gives Chinese context


async def generate_and_split_tones(
    syllable: str,
    voice: str,
    output_dir: Path,
    voice_name: str,
) -> dict[int, int]:
    """Generate all 5 tones for a syllable and split into separate files.

    Tones 1-4 are generated together and split.
    Tone 0 (neutral) is generated separately with Chinese character context.

    Returns dict of tone -> duration_ms for each successfully extracted tone.
    """
    import edge_tts

    results = {}

    # Generate tones 1-4 together
    prompt = create_tone_prompt(syllable)
    communicate = edge_tts.Communicate(prompt, voice)

    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]

    if audio_data:
        # Split into tones 1-4
        tone_results = split_tones_from_audio(
            audio_data, syllable, output_dir, voice_name,
            tones=[1, 2, 3, 4], n_expected=4
        )
        results.update(tone_results)

    # Generate neutral tone (0) separately with Chinese context
    neutral_prompt = get_neutral_tone_prompt(syllable)
    communicate = edge_tts.Communicate(neutral_prompt, voice)

    neutral_audio = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            neutral_audio += chunk["data"]

    if neutral_audio:
        neutral_result = save_single_tone_audio(
            neutral_audio, syllable, output_dir, voice_name, tone=0
        )
        if neutral_result:
            results[0] = neutral_result

    return results


def save_single_tone_audio(
    mp3_data: bytes,
    syllable: str,
    output_dir: Path,
    voice_name: str,
    tone: int,
    target_sr: int = 16000,
) -> int | None:
    """Save a single tone audio file.

    Returns duration_ms if successful, None otherwise.
    """
    import tempfile
    import subprocess

    # Find ffmpeg
    ffmpeg_paths = [
        "ffmpeg",
        r"C:\Users\costa\AppData\Local\Overwolf\Extensions\ncfplpkmiejjaklknfnkgcpapnhkggmlcppckhcb\270.0.25\obs\bin\64bit\ffmpeg.exe",
    ]

    ffmpeg_cmd = None
    for path in ffmpeg_paths:
        try:
            subprocess.run([path, "-version"], capture_output=True, check=True)
            ffmpeg_cmd = path
            break
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    if ffmpeg_cmd is None:
        return None

    # Write MP3 to temp file and convert to WAV
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
        tmp_mp3.write(mp3_data)
        tmp_mp3_path = tmp_mp3.name

    tmp_wav_path = tmp_mp3_path.replace(".mp3", ".wav")

    try:
        subprocess.run([
            ffmpeg_cmd, "-y", "-i", tmp_mp3_path,
            "-ar", str(target_sr), "-ac", "1", "-sample_fmt", "s16",
            tmp_wav_path
        ], capture_output=True, check=True)

        # Read WAV
        with wave.open(tmp_wav_path, 'rb') as wf:
            n_frames = wf.getnframes()
            raw_data = wf.readframes(n_frames)

        audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Trim silence from start and end
        audio = trim_silence(audio, target_sr)

        if len(audio) < target_sr * 0.05:  # Less than 50ms
            return None

        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.9

        # Save
        tone_dir = output_dir / voice_name / f"tone{tone}"
        tone_dir.mkdir(parents=True, exist_ok=True)
        output_path = tone_dir / f"{syllable}.wav"

        audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)

        with wave.open(str(output_path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(target_sr)
            wf.writeframes(audio_int16.tobytes())

        return int(len(audio) / target_sr * 1000)

    except Exception as e:
        logger.error(f"Error saving neutral tone for {syllable}: {e}")
        return None

    finally:
        import os
        if os.path.exists(tmp_mp3_path):
            os.unlink(tmp_mp3_path)
        if os.path.exists(tmp_wav_path):
            os.unlink(tmp_wav_path)


def trim_silence(audio: np.ndarray, sr: int, threshold: float = 0.02) -> np.ndarray:
    """Trim silence from start and end of audio."""
    # Find first non-silent sample
    abs_audio = np.abs(audio)
    threshold_val = threshold * np.max(abs_audio) if np.max(abs_audio) > 0 else threshold

    # Compute energy in small windows
    window_size = int(0.01 * sr)  # 10ms windows
    n_windows = len(audio) // window_size

    start_idx = 0
    end_idx = len(audio)

    for i in range(n_windows):
        window = audio[i * window_size:(i + 1) * window_size]
        if np.sqrt(np.mean(window ** 2)) > threshold_val:
            start_idx = max(0, i * window_size - window_size)  # Small margin
            break

    for i in range(n_windows - 1, -1, -1):
        window = audio[i * window_size:(i + 1) * window_size]
        if np.sqrt(np.mean(window ** 2)) > threshold_val:
            end_idx = min(len(audio), (i + 1) * window_size + window_size)
            break

    return audio[start_idx:end_idx]


def split_tones_from_audio(
    mp3_data: bytes,
    syllable: str,
    output_dir: Path,
    voice_name: str,
    tones: list[int] = None,
    n_expected: int = 4,
    target_sr: int = 16000,
) -> dict[int, int]:
    """Split combined audio into individual tone files.

    Uses silence detection to find boundaries between tones.

    Args:
        mp3_data: Raw MP3 audio bytes
        syllable: The syllable being processed
        output_dir: Output directory
        voice_name: Voice identifier
        tones: List of tone numbers in order they appear in audio
        n_expected: Number of segments expected
        target_sr: Target sample rate
    """
    import tempfile
    import subprocess

    if tones is None:
        tones = [1, 2, 3, 4]

    # Find ffmpeg
    ffmpeg_paths = [
        "ffmpeg",
        r"C:\Users\costa\AppData\Local\Overwolf\Extensions\ncfplpkmiejjaklknfnkgcpapnhkggmlcppckhcb\270.0.25\obs\bin\64bit\ffmpeg.exe",
    ]

    ffmpeg_cmd = None
    for path in ffmpeg_paths:
        try:
            subprocess.run([path, "-version"], capture_output=True, check=True)
            ffmpeg_cmd = path
            break
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    if ffmpeg_cmd is None:
        raise RuntimeError("ffmpeg not found")

    # Write MP3 to temp file and convert to WAV
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
        tmp_mp3.write(mp3_data)
        tmp_mp3_path = tmp_mp3.name

    tmp_wav_path = tmp_mp3_path.replace(".mp3", ".wav")

    try:
        subprocess.run([
            ffmpeg_cmd, "-y", "-i", tmp_mp3_path,
            "-ar", str(target_sr), "-ac", "1", "-sample_fmt", "s16",
            tmp_wav_path
        ], capture_output=True, check=True)

        # Read WAV
        with wave.open(tmp_wav_path, 'rb') as wf:
            n_frames = wf.getnframes()
            raw_data = wf.readframes(n_frames)

        audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Find silence boundaries
        segments = find_tone_segments(audio, target_sr, n_expected=n_expected)

        if len(segments) != n_expected:
            logger.warning(f"Expected {n_expected} segments for {syllable}, got {len(segments)}")
            if len(segments) < n_expected:
                return {}

        # Save each tone
        results = {}

        for i, (start, end) in enumerate(segments[:n_expected]):
            tone = tones[i]
            tone_dir = output_dir / voice_name / f"tone{tone}"
            tone_dir.mkdir(parents=True, exist_ok=True)

            # Extract segment with small margin
            margin = int(0.01 * target_sr)  # 10ms margin
            start_idx = max(0, start - margin)
            end_idx = min(len(audio), end + margin)

            segment = audio[start_idx:end_idx]

            # Normalize
            max_val = np.max(np.abs(segment))
            if max_val > 0:
                segment = segment / max_val * 0.9

            # Save
            output_path = tone_dir / f"{syllable}.wav"
            audio_int16 = (segment * 32767).clip(-32768, 32767).astype(np.int16)

            with wave.open(str(output_path), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(target_sr)
                wf.writeframes(audio_int16.tobytes())

            results[tone] = int(len(segment) / target_sr * 1000)

        return results

    finally:
        import os
        if os.path.exists(tmp_mp3_path):
            os.unlink(tmp_mp3_path)
        if os.path.exists(tmp_wav_path):
            os.unlink(tmp_wav_path)


def find_tone_segments(
    audio: np.ndarray,
    sr: int,
    n_expected: int = 4,
    min_silence_ms: int = 100,
    silence_threshold: float = 0.02,
) -> list[tuple[int, int]]:
    """Find segments separated by silence.

    Returns list of (start_sample, end_sample) tuples.
    """
    # Compute frame-level energy
    frame_size = int(0.025 * sr)  # 25ms frames
    hop_size = int(0.010 * sr)    # 10ms hop

    n_frames = (len(audio) - frame_size) // hop_size + 1
    energy = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * hop_size
        frame = audio[start:start + frame_size]
        energy[i] = np.sqrt(np.mean(frame ** 2))

    # Smooth energy
    kernel_size = 5
    energy_smooth = np.convolve(energy, np.ones(kernel_size) / kernel_size, mode='same')

    # Find voiced regions (above threshold)
    threshold = silence_threshold * np.max(energy_smooth)
    is_voiced = energy_smooth > threshold

    # Find segment boundaries
    segments = []
    in_segment = False
    segment_start = 0

    min_silence_frames = int(min_silence_ms / 10)  # Convert ms to frames
    silence_count = 0

    for i, voiced in enumerate(is_voiced):
        if voiced:
            if not in_segment:
                segment_start = i
                in_segment = True
            silence_count = 0
        else:
            if in_segment:
                silence_count += 1
                if silence_count >= min_silence_frames:
                    # End of segment
                    segment_end = i - silence_count
                    if segment_end > segment_start:
                        start_sample = segment_start * hop_size
                        end_sample = segment_end * hop_size + frame_size
                        segments.append((start_sample, end_sample))
                    in_segment = False
                    silence_count = 0

    # Handle last segment
    if in_segment:
        start_sample = segment_start * hop_size
        end_sample = len(audio)
        segments.append((start_sample, end_sample))

    # If we have more segments than expected, merge small ones
    while len(segments) > n_expected and len(segments) > 1:
        # Find shortest segment and merge with neighbor
        min_len = float('inf')
        min_idx = 0
        for i, (s, e) in enumerate(segments):
            if e - s < min_len:
                min_len = e - s
                min_idx = i

        # Merge with next or previous
        if min_idx < len(segments) - 1:
            segments[min_idx] = (segments[min_idx][0], segments[min_idx + 1][1])
            segments.pop(min_idx + 1)
        else:
            segments[min_idx - 1] = (segments[min_idx - 1][0], segments[min_idx][1])
            segments.pop(min_idx)

    return segments


async def generate_lexicon(
    output_dir: Path,
    voices: dict[str, str] | None = None,
    syllables: list[str] | None = None,
) -> None:
    """Generate complete syllable lexicon."""

    if voices is None:
        voices = VOICES

    if syllables is None:
        syllables = get_all_valid_syllables()

    logger.info(f"Generating {len(syllables)} syllables × {len(voices)} voices × 5 tones")
    logger.info(f"Total files: {len(syllables) * len(voices) * 5}")

    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(syllables) * len(voices)
    completed = 0
    failed = []

    for voice_name, voice_id in voices.items():
        logger.info(f"\nProcessing voice: {voice_name} ({voice_id})")

        for syllable in syllables:
            try:
                results = await generate_and_split_tones(
                    syllable=syllable,
                    voice=voice_id,
                    output_dir=output_dir,
                    voice_name=voice_name,
                )

                if len(results) < 5:
                    failed.append((voice_name, syllable, f"only {len(results)} tones"))

            except Exception as e:
                failed.append((voice_name, syllable, str(e)))
                logger.error(f"Failed {voice_name}/{syllable}: {e}")

            completed += 1
            if completed % 20 == 0:
                logger.info(f"Progress: {completed}/{total} ({100*completed/total:.1f}%)")

    # Save metadata
    import json
    metadata = {
        "voices": voices,
        "syllables": syllables,
        "total_syllables": len(syllables),
        "tones": [0, 1, 2, 3, 4],
        "failed": failed,
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"\nComplete! {completed} syllable sets processed")
    logger.info(f"Failed: {len(failed)}")
    if failed:
        logger.info("Failed items:")
        for voice, syl, reason in failed[:20]:
            logger.info(f"  {voice}/{syl}: {reason}")


def main():
    parser = argparse.ArgumentParser(description="Generate syllable lexicon v2")
    parser.add_argument(
        "--output", type=Path,
        default=Path("tools/mandarin_grader/data/syllables_v2"),
        help="Output directory",
    )
    parser.add_argument(
        "--voices", nargs="+",
        default=list(VOICES.keys()),
        help="Voices to generate",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Test mode: only generate 5 syllables",
    )

    args = parser.parse_args()

    voices = {k: VOICES[k] for k in args.voices if k in VOICES}
    syllables = get_all_valid_syllables()

    if args.test:
        syllables = syllables[:5]
        logger.info("Test mode: generating only 5 syllables")

    logger.info(f"Syllables to generate: {len(syllables)}")
    logger.info(f"Voices: {list(voices.keys())}")

    asyncio.run(generate_lexicon(
        output_dir=args.output,
        voices=voices,
        syllables=syllables,
    ))


if __name__ == "__main__":
    main()
