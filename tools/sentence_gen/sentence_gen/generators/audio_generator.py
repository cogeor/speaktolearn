"""Audio generator using OpenAI TTS API."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from ..config import Config
from ..models.dataset import Dataset
from ..models.text_sequence import ExampleAudio, TextSequence
from ..models.voice import VoiceRef


class AudioGenerator:
    """Generates TTS audio for text sequences."""

    def __init__(self, config: Config):
        """Initialize the audio generator.

        Args:
            config: Application configuration containing OpenAI API settings
        """
        self.config = config
        self.client = OpenAI(api_key=config.openai.api_key)

    def generate_all(
        self,
        dataset: Dataset,
        output_dir: Path,
        voices: list[str],
    ) -> None:
        """Generate audio for all sequences in dataset.

        Args:
            dataset: The dataset containing text sequences
            output_dir: Directory to write audio files
            voices: List of voice names to generate (e.g., ["female", "male"])
        """
        voice_configs = {
            "female": self.config.tts.openai_voice_female,
            "male": self.config.tts.openai_voice_male,
        }

        # Create output directories
        for voice in voices:
            (output_dir / voice).mkdir(parents=True, exist_ok=True)

        # Generate audio in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for item in dataset.items:
                for voice in voices:
                    future = executor.submit(
                        self._generate_audio,
                        item=item,
                        voice_name=voice,
                        voice_id=voice_configs[voice],
                        output_dir=output_dir,
                    )
                    futures.append((item, voice, future))

            # Collect results and update dataset
            for item, voice, future in futures:
                audio_path = future.result()
                if audio_path:
                    self._add_audio_ref(item, voice, audio_path, output_dir)

    def _generate_audio(
        self,
        item: TextSequence,
        voice_name: str,
        voice_id: str,
        output_dir: Path,
    ) -> Path | None:
        """Generate audio for a single sequence.

        Args:
            item: The text sequence to generate audio for
            voice_name: Name of the voice (e.g., "female", "male")
            voice_id: OpenAI voice ID (e.g., "nova", "onyx")
            output_dir: Directory to write the audio file

        Returns:
            Path to the generated audio file, or None if generation failed
        """
        output_path = output_dir / voice_name / f"{item.id}.opus"

        # Skip if audio file already exists
        if output_path.exists():
            return output_path

        try:
            # Generate with OpenAI TTS
            response = self.client.audio.speech.create(
                model="tts-1",
                voice=voice_id,
                input=item.text,
                response_format="opus",
            )

            # Save to file
            with open(output_path, "wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)

            return output_path
        except Exception as e:
            print(f"Error generating audio for {item.id}: {e}")
            return None

    def _add_audio_ref(
        self,
        item: TextSequence,
        voice_name: str,
        audio_path: Path,
        base_dir: Path,
    ) -> None:
        """Add audio reference to item's example_audio.

        Args:
            item: The text sequence to update
            voice_name: Name of the voice (e.g., "female", "male")
            audio_path: Path to the audio file
            base_dir: Base directory for computing relative paths
        """
        uri = f"assets://examples/{voice_name}/{item.id}.opus"

        voice_ref = VoiceRef(
            id=voice_name[0] + "1",  # f1, m1
            label={"en": voice_name.title()},
            uri=uri,
        )

        if item.example_audio is None:
            item.example_audio = ExampleAudio()

        # Remove existing voice if present (to allow re-generation)
        item.example_audio.voices = [
            v for v in item.example_audio.voices if v.id != voice_ref.id
        ]
        item.example_audio.voices.append(voice_ref)
