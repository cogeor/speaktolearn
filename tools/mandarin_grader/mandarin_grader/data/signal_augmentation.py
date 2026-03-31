"""Waveform-level augmentation for domain adaptation (studio -> mobile).

Applies signal-domain augmentations before mel extraction to bridge the gap
between clean studio recordings and noisy mobile phone recordings.

Augmentations:
- RIR convolution: Simulate room acoustics using real or synthetic impulse responses
- Additive noise: Mix in background noise from corpus or synthesize Gaussian noise
- Codec compression: Simulate low-bitrate codec degradation via ffmpeg or spectral approximation
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Use scipy for efficient convolution if available
try:
    from scipy.signal import fftconvolve as _fftconvolve
    SCIPY_AVAILABLE = True
except ImportError:
    _fftconvolve = None  # type: ignore[assignment]
    SCIPY_AVAILABLE = False


# -----------------------------------------------------------------------------
# Configuration dataclasses
# -----------------------------------------------------------------------------

@dataclass
class RIRConfig:
    """Room Impulse Response convolution configuration."""
    enabled: bool = True
    prob: float = 0.5
    rir_dir: Optional[str] = None  # Path to OpenSLR-28 RIR directory; None = synthetic


@dataclass
class AdditiveNoiseConfig:
    """Additive background noise configuration."""
    enabled: bool = True
    prob: float = 0.5
    snr_db_range: tuple[float, float] = (5.0, 20.0)
    noise_dir: Optional[str] = None  # Path to MUSAN or similar noise corpus; None = Gaussian


@dataclass
class CodecConfig:
    """Codec compression simulation configuration."""
    enabled: bool = True
    prob: float = 0.2
    bitrate_range: tuple[int, int] = (32000, 64000)  # bits/s for AAC encoding


@dataclass
class WaveformAugmentConfig:
    """Unified configuration for waveform-level augmentation pipeline."""
    rir: RIRConfig = field(default_factory=RIRConfig)
    noise: AdditiveNoiseConfig = field(default_factory=AdditiveNoiseConfig)
    codec: CodecConfig = field(default_factory=CodecConfig)


# -----------------------------------------------------------------------------
# WaveformAugmenter
# -----------------------------------------------------------------------------

class WaveformAugmenter:
    """Applies waveform-level augmentations before mel extraction.

    Augmentations are applied in this order:
    1. RIR convolution (room acoustics)
    2. Additive noise (background noise)
    3. Codec compression (low-bitrate simulation)
    """

    def __init__(self, config: WaveformAugmentConfig):
        self.config = config
        self._rir_files: Optional[list[Path]] = None
        self._noise_files: Optional[list[Path]] = None
        self._ffmpeg_available: Optional[bool] = None

    # ------------------------------------------------------------------
    # Lazy file scanning
    # ------------------------------------------------------------------

    def _get_rir_files(self) -> list[Path]:
        """Lazily scan rir_dir for .wav files."""
        if self._rir_files is None:
            rir_dir = self.config.rir.rir_dir
            if rir_dir:
                d = Path(rir_dir)
                self._rir_files = list(d.rglob("*.wav"))
                if not self._rir_files:
                    logger.warning(f"No .wav files found in RIR directory: {d}")
            else:
                self._rir_files = []
        return self._rir_files

    def _get_noise_files(self) -> list[Path]:
        """Lazily scan noise_dir for .wav files."""
        if self._noise_files is None:
            noise_dir = self.config.noise.noise_dir
            if noise_dir:
                d = Path(noise_dir)
                self._noise_files = list(d.rglob("*.wav"))
                if not self._noise_files:
                    logger.warning(f"No .wav files found in noise directory: {d}")
            else:
                self._noise_files = []
        return self._noise_files

    def _is_ffmpeg_available(self) -> bool:
        """Check whether ffmpeg is on PATH (cached)."""
        if self._ffmpeg_available is None:
            try:
                result = subprocess.run(
                    ["ffmpeg", "-version"],
                    capture_output=True,
                    timeout=5,
                )
                self._ffmpeg_available = result.returncode == 0
            except (FileNotFoundError, subprocess.TimeoutExpired):
                self._ffmpeg_available = False
        return self._ffmpeg_available

    # ------------------------------------------------------------------
    # Individual augmentation methods
    # ------------------------------------------------------------------

    def _apply_rir(
        self,
        audio: NDArray[np.float32],
        sr: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float32]:
        """Convolve audio with a room impulse response.

        Uses a real RIR .wav file from rir_dir if available, otherwise
        generates a synthetic exponential-decay impulse response.
        """
        rir_files = self._get_rir_files()

        if rir_files:
            # Pick a random RIR file and load it
            chosen = rng.choice(len(rir_files))  # type: ignore[arg-type]
            rir_path = rir_files[int(chosen)]
            try:
                import librosa
                rir_audio, rir_sr = librosa.load(str(rir_path), sr=sr, mono=True)
                rir = rir_audio.astype(np.float32)
            except Exception as e:
                logger.debug(f"Failed to load RIR {rir_path}: {e}; using synthetic RIR")
                rir = self._synthetic_rir(sr, rng)
        else:
            rir = self._synthetic_rir(sr, rng)

        # Normalize RIR
        rir_norm = rir / (np.abs(rir).max() + 1e-8)

        # Convolve: use fftconvolve if scipy available, otherwise np.convolve
        if SCIPY_AVAILABLE:
            assert _fftconvolve is not None
            convolved = _fftconvolve(audio, rir_norm, mode="full")
        else:
            convolved = np.convolve(audio, rir_norm, mode="full")

        # Trim to original length
        convolved = convolved[: len(audio)].astype(np.float32)

        # Preserve original RMS level
        orig_rms = np.sqrt(np.mean(audio ** 2)) + 1e-8
        conv_rms = np.sqrt(np.mean(convolved ** 2)) + 1e-8
        convolved = convolved * (orig_rms / conv_rms)

        return convolved

    def _synthetic_rir(
        self,
        sr: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float32]:
        """Generate a simple synthetic RIR using exponential decay.

        The RIR simulates early reflections and reverb tail by combining a
        direct impulse with an exponentially decaying noise burst.
        """
        # RT60 between 100ms and 600ms (samples)
        rt60_ms = rng.uniform(100.0, 600.0)
        rt60_samples = int(rt60_ms / 1000.0 * sr)
        rt60_samples = max(rt60_samples, 16)  # Minimum sensible length

        t = np.arange(rt60_samples, dtype=np.float32)
        # Exponential decay: exp(-6.9 * t / rt60) gives ~60dB decay at rt60
        decay = np.exp(-6.9 * t / rt60_samples)
        noise = rng.standard_normal(rt60_samples).astype(np.float32)
        rir = decay * noise

        # Ensure unit direct path at sample 0
        rir[0] = 1.0

        return rir

    def _apply_noise(
        self,
        audio: NDArray[np.float32],
        sr: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float32]:
        """Add background noise at a random SNR.

        Uses a random clip from noise_dir if available, otherwise uses
        Gaussian noise at the configured SNR range.
        """
        snr_db = rng.uniform(
            self.config.noise.snr_db_range[0],
            self.config.noise.snr_db_range[1],
        )

        signal_power = np.mean(audio ** 2)
        if signal_power < 1e-12:
            return audio

        noise_power_target = signal_power / (10 ** (snr_db / 10))

        noise_files = self._get_noise_files()
        if noise_files:
            chosen = rng.choice(len(noise_files))  # type: ignore[arg-type]
            noise_path = noise_files[int(chosen)]
            try:
                import librosa
                noise_audio, _ = librosa.load(str(noise_path), sr=sr, mono=True)
                noise_audio = noise_audio.astype(np.float32)

                # Tile or trim noise clip to match audio length
                if len(noise_audio) < len(audio):
                    repeats = int(np.ceil(len(audio) / len(noise_audio)))
                    noise_audio = np.tile(noise_audio, repeats)
                # Random start offset within noise clip
                max_offset = len(noise_audio) - len(audio)
                start = int(rng.integers(0, max(1, max_offset + 1)))
                noise_clip = noise_audio[start : start + len(audio)]
            except Exception as e:
                logger.debug(f"Failed to load noise file {noise_path}: {e}; using Gaussian")
                noise_clip = rng.standard_normal(len(audio)).astype(np.float32)
        else:
            noise_clip = rng.standard_normal(len(audio)).astype(np.float32)

        # Scale noise clip to target power
        noise_rms = np.sqrt(np.mean(noise_clip ** 2)) + 1e-8
        scale = np.sqrt(noise_power_target) / noise_rms
        return (audio + noise_clip * scale).astype(np.float32)

    def _apply_codec(
        self,
        audio: NDArray[np.float32],
        sr: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float32]:
        """Simulate codec compression artifacts.

        If ffmpeg is available, encodes to AAC at a low bitrate then decodes
        back to PCM. Falls back to a spectral degradation approximation
        (low-pass filter + quantization noise) if ffmpeg is unavailable.
        """
        bitrate = int(rng.integers(
            self.config.codec.bitrate_range[0],
            self.config.codec.bitrate_range[1] + 1,
        ))

        if self._is_ffmpeg_available():
            return self._codec_via_ffmpeg(audio, sr, bitrate)
        else:
            return self._codec_spectral_approx(audio, sr, bitrate, rng)

    def _codec_via_ffmpeg(
        self,
        audio: NDArray[np.float32],
        sr: int,
        bitrate: int,
    ) -> NDArray[np.float32]:
        """Encode to AAC and decode back via ffmpeg subprocess."""
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp = Path(tmp_dir)
                in_wav = tmp / "input.wav"
                aac_file = tmp / "compressed.aac"
                out_wav = tmp / "output.wav"

                # Write input wav using scipy or wave
                _write_wav(in_wav, audio, sr)

                # Encode to AAC
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-i", str(in_wav),
                        "-c:a", "aac",
                        "-b:a", f"{bitrate}",
                        str(aac_file),
                    ],
                    capture_output=True,
                    check=True,
                    timeout=10,
                )

                # Decode back to WAV
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-i", str(aac_file),
                        "-ar", str(sr),
                        "-ac", "1",
                        str(out_wav),
                    ],
                    capture_output=True,
                    check=True,
                    timeout=10,
                )

                # Read output
                out_audio = _read_wav(out_wav, sr)

                # Trim or zero-pad to match original length
                if len(out_audio) >= len(audio):
                    return out_audio[: len(audio)]
                else:
                    padded = np.zeros(len(audio), dtype=np.float32)
                    padded[: len(out_audio)] = out_audio
                    return padded

        except Exception as e:
            logger.debug(f"ffmpeg codec simulation failed: {e}; falling back to spectral approx")
            # Fallback on any error
            rng = np.random.default_rng()
            return self._codec_spectral_approx(audio, sr, bitrate, rng)

    def _codec_spectral_approx(
        self,
        audio: NDArray[np.float32],
        sr: int,
        bitrate: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float32]:
        """Approximate codec degradation with low-pass filter + quantization noise.

        Lower bitrate -> lower cutoff frequency and more quantization noise.
        Bitrate range: 32000-64000 bits/s maps to cutoff 3000-6000 Hz.
        """
        # Map bitrate to cutoff: linear scaling
        lo_br, hi_br = self.config.codec.bitrate_range
        lo_cut, hi_cut = 3000.0, 6000.0
        t = (bitrate - lo_br) / max(hi_br - lo_br, 1)
        cutoff_hz = lo_cut + t * (hi_cut - lo_cut)

        # Simple low-pass filter via FFT
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), d=1.0 / sr)
        # Soft cutoff using sigmoid-like attenuation
        rolloff = 1.0 / (1.0 + np.exp((freqs - cutoff_hz) / 200.0))
        fft_filtered = fft * rolloff.astype(np.complex64)
        filtered = np.fft.irfft(fft_filtered, n=len(audio)).astype(np.float32)

        # Quantization noise: proportional to inverse bitrate
        # At 32kbps -> ~3% noise, at 64kbps -> ~0.5% noise
        noise_fraction = 0.03 * (1.0 - t) + 0.005 * t
        signal_rms = np.sqrt(np.mean(filtered ** 2)) + 1e-8
        quant_noise = rng.standard_normal(len(filtered)).astype(np.float32)
        quant_noise *= signal_rms * noise_fraction

        return (filtered + quant_noise).astype(np.float32)

    # ------------------------------------------------------------------
    # Main augment method
    # ------------------------------------------------------------------

    def augment(
        self,
        audio: NDArray[np.float32],
        sr: int = 16000,
        rng: Optional[np.random.Generator] = None,
    ) -> NDArray[np.float32]:
        """Apply configured augmentations to waveform.

        Args:
            audio: Waveform samples as float32 array
            sr: Sample rate in Hz
            rng: Random number generator (optional, for reproducibility)

        Returns:
            Augmented waveform as float32 array, same length as input
        """
        if rng is None:
            rng = np.random.default_rng()

        # 1. RIR convolution
        cfg = self.config.rir
        if cfg.enabled and rng.random() < cfg.prob:
            audio = self._apply_rir(audio, sr, rng)

        # 2. Additive noise
        cfg = self.config.noise
        if cfg.enabled and rng.random() < cfg.prob:
            audio = self._apply_noise(audio, sr, rng)

        # 3. Codec compression
        cfg = self.config.codec
        if cfg.enabled and rng.random() < cfg.prob:
            audio = self._apply_codec(audio, sr, rng)

        return audio


# -----------------------------------------------------------------------------
# WAV I/O helpers (avoid mandatory scipy dependency)
# -----------------------------------------------------------------------------

def _write_wav(path: Path, audio: NDArray[np.float32], sr: int) -> None:
    """Write float32 audio to a WAV file."""
    try:
        import soundfile as sf
        sf.write(str(path), audio, sr, subtype="PCM_16")
        return
    except ImportError:
        pass

    try:
        from scipy.io import wavfile
        audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        wavfile.write(str(path), sr, audio_int16)
        return
    except ImportError:
        pass

    # Pure stdlib fallback via wave module
    import wave
    import struct
    audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{len(audio_int16)}h", *audio_int16))


def _read_wav(path: Path, sr: int) -> NDArray[np.float32]:
    """Read a WAV file into a float32 array."""
    try:
        import librosa
        audio, _ = librosa.load(str(path), sr=sr, mono=True)
        return audio.astype(np.float32)
    except ImportError:
        pass

    try:
        from scipy.io import wavfile
        rate, data = wavfile.read(str(path))
        if data.dtype == np.int16:
            audio = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            audio = data.astype(np.float32) / 2147483648.0
        else:
            audio = data.astype(np.float32)
        return audio
    except ImportError:
        pass

    # Stdlib wave fallback
    import wave
    import struct
    with wave.open(str(path), "r") as wf:
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
        data = struct.unpack(f"<{n_frames}h", raw)
        return np.array(data, dtype=np.float32) / 32768.0
