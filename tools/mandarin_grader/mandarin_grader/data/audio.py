"""Audio loading and preprocessing utilities."""

from pathlib import Path
import subprocess

import numpy as np

# Target sample rate for all processing
TARGET_SR = 16000


def load_audio(path: Path, target_sr: int = TARGET_SR) -> np.ndarray:
    """Load audio file and resample to target rate.

    Args:
        path: Path to audio file (supports WAV, MP3, M4A, etc.)
        target_sr: Target sample rate in Hz

    Returns:
        Audio samples as float32 array, normalized to [-1, 1]
    """
    import librosa

    audio, _ = librosa.load(str(path), sr=target_sr, mono=True)
    return audio.astype(np.float32)


def extract_mel(
    audio: np.ndarray,
    sr: int = TARGET_SR,
    n_mels: int = 80,
    hop_length: int = 160,  # 10ms at 16kHz
    win_length: int = 400,  # 25ms at 16kHz
) -> np.ndarray:
    """Extract log-mel spectrogram.

    Args:
        audio: Audio samples as float32 array
        sr: Sample rate in Hz
        n_mels: Number of mel frequency bins
        hop_length: Hop length in samples
        win_length: Window length in samples

    Returns:
        Log-mel spectrogram, shape [n_mels, T]
    """
    import librosa

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        win_length=win_length,
        fmin=20,
        fmax=8000,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def convert_to_wav(
    input_path: Path,
    output_path: Path | None = None,
    target_sr: int = TARGET_SR,
) -> Path:
    """Convert audio file to WAV format using ffmpeg.

    Args:
        input_path: Path to input audio file
        output_path: Path for output WAV file (default: same name with .wav)
        target_sr: Target sample rate

    Returns:
        Path to output WAV file
    """
    if output_path is None:
        output_path = input_path.with_suffix(".wav")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ar",
        str(target_sr),
        "-ac",
        "1",
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def get_duration_ms(audio: np.ndarray, sr: int = TARGET_SR) -> int:
    """Get audio duration in milliseconds."""
    return int(len(audio) / sr * 1000)
