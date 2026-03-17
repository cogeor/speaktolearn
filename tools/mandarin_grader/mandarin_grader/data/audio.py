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


def extract_f0_bins(
    audio: np.ndarray,
    sr: int = TARGET_SR,
    n_bins: int = 64,
    hop_length: int = 160,  # 10ms at 16kHz, matches mel hop
    win_length: int = 400,  # 25ms at 16kHz, matches mel window
    fmin: float = 50.0,
    fmax: float = 500.0,
) -> np.ndarray:
    """Extract F0 pitch and bin into discrete pitch bins aligned to mel frames.

    Uses librosa.pyin for robust pitch estimation with voiced/unvoiced detection.
    Bins are log-spaced between fmin and fmax to better represent the perceptual
    pitch range relevant for Mandarin tones (speech typically 50-500 Hz).

    Args:
        audio: Audio samples as float32 array
        sr: Sample rate in Hz
        n_bins: Number of pitch bins (default 64)
        hop_length: Hop length in samples (must match mel extraction)
        win_length: Window length in samples (must match mel extraction)
        fmin: Minimum frequency in Hz (default 50.0)
        fmax: Maximum frequency in Hz (default 500.0)

    Returns:
        Integer array of bin indices, shape [T], where:
        - 0 = unvoiced / no pitch detected
        - 1 to n_bins = pitch bin index (log-spaced from fmin to fmax)
        Length T matches the number of mel frames for the same audio.
    """
    import librosa

    # pyin returns f0 (Hz or NaN for unvoiced), voiced_flag, voiced_prob
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        hop_length=hop_length,
        win_length=win_length,
    )

    # Log-spaced bin edges: n_bins+1 edges define n_bins intervals
    bin_edges = np.logspace(np.log10(fmin), np.log10(fmax), n_bins + 1)

    # Assign each frame to a bin index (1-indexed; 0 = unvoiced)
    n_frames = len(f0)
    bin_indices = np.zeros(n_frames, dtype=np.int64)

    voiced_mask = voiced_flag & np.isfinite(f0)
    if voiced_mask.any():
        # np.digitize returns 1..n_bins for values in range, 0 or n_bins+1 for out-of-range
        raw_bins = np.digitize(f0[voiced_mask], bin_edges)
        # Clamp to valid range [1, n_bins]
        raw_bins = np.clip(raw_bins, 1, n_bins)
        bin_indices[voiced_mask] = raw_bins

    return bin_indices
