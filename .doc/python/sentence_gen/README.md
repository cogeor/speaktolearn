# tools/sentence_gen/ - Python Data Generation Utility

## Purpose

CLI tool that generates language learning datasets:
1. Generate text sequences (sentences/phrases) via LLM
2. Generate example audio via TTS
3. Validate against schema
4. Export as JSON + audio files for Flutter app

**This is NOT a backend service** - it's a build-time data generation tool.

## Folder Structure

```
tools/sentence_gen/
├── pyproject.toml              # Project config, dependencies
├── .env                        # API keys (not committed)
├── .env.example                # Template for .env
│
├── sentence_gen/
│   ├── __init__.py
│   ├── cli.py                  # Typer CLI entrypoint
│   ├── config.py               # Configuration loading
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── text_sequence.py    # Pydantic models
│   │   ├── dataset.py          # Dataset container
│   │   └── voice.py            # Voice configuration
│   │
│   ├── generators/
│   │   ├── __init__.py
│   │   ├── text_generator.py   # LLM text generation
│   │   └── audio_generator.py  # TTS audio generation
│   │
│   ├── prompts/
│   │   ├── zh_sentences.md     # Mandarin generation prompt
│   │   └── ja_sentences.md     # Japanese (future)
│   │
│   └── exporters/
│       ├── __init__.py
│       └── json_exporter.py    # JSON export
│
├── output/                     # Generated files
│   ├── sentences.zh.json       # Dataset JSON
│   └── examples/               # Audio files
│       ├── female/
│       │   └── ts_000001.opus
│       └── male/
│           └── ts_000001.opus
│
└── tests/
    ├── test_text_generator.py
    ├── test_audio_generator.py
    └── test_models.py
```

---

## Dependencies

```toml
# pyproject.toml
[project]
name = "sentence-gen"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "typer>=0.9.0",
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
    "openai>=1.0.0",
    "httpx>=0.25.0",
    "rich>=13.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
]

[project.scripts]
sentence-gen = "sentence_gen.cli:app"
```

---

## Files

### `cli.py`

**Purpose**: Typer CLI interface.

**Commands**:

```bash
# Generate text sequences
sentence-gen generate --language zh-CN --count 50 --tags hsk1,daily

# Generate audio for existing sequences
sentence-gen audio --input output/sentences.zh.json

# Full pipeline: generate text + audio
sentence-gen full --language zh-CN --count 50 --tags hsk1

# Validate dataset
sentence-gen validate --input output/sentences.zh.json

# Export for Flutter (copy to assets)
sentence-gen export --input output/ --flutter-assets ../apps/mobile_flutter/assets/
```

**Implementation**:

```python
from typing import Annotated, Optional
import typer
from pathlib import Path
from rich.console import Console

from .config import load_config
from .generators.text_generator import TextGenerator
from .generators.audio_generator import AudioGenerator
from .exporters.json_exporter import JsonExporter
from .models.dataset import Dataset

app = typer.Typer(help="Generate language learning datasets")
console = Console()


@app.command()
def generate(
    language: Annotated[str, typer.Option(help="Target language code")] = "zh-CN",
    count: Annotated[int, typer.Option(help="Number of sequences")] = 50,
    tags: Annotated[Optional[str], typer.Option(help="Comma-separated tags")] = None,
    output: Annotated[Path, typer.Option(help="Output directory")] = Path("output"),
    difficulty: Annotated[int, typer.Option(help="Difficulty level (1-5)")] = 1,
) -> None:
    """Generate text sequences using LLM."""
    config = load_config()
    generator = TextGenerator(config)

    tag_list = tags.split(",") if tags else []

    console.print(f"[bold]Generating {count} sequences for {language}...[/bold]")

    dataset = generator.generate(
        language=language,
        count=count,
        tags=tag_list,
        difficulty=difficulty,
    )

    output.mkdir(parents=True, exist_ok=True)
    output_path = output / f"sentences.{language.split('-')[0]}.json"

    exporter = JsonExporter()
    exporter.export(dataset, output_path)

    console.print(f"[green]Generated {len(dataset.items)} sequences to {output_path}[/green]")


@app.command()
def audio(
    input_file: Annotated[Path, typer.Argument(help="Input dataset JSON")],
    output: Annotated[Path, typer.Option(help="Output directory")] = Path("output/examples"),
    voices: Annotated[str, typer.Option(help="Voices to generate")] = "female,male",
) -> None:
    """Generate TTS audio for sequences."""
    config = load_config()
    generator = AudioGenerator(config)

    console.print(f"[bold]Loading dataset from {input_file}...[/bold]")
    dataset = Dataset.from_json_file(input_file)

    voice_list = voices.split(",")
    output.mkdir(parents=True, exist_ok=True)

    with console.status("Generating audio..."):
        generator.generate_all(dataset, output, voices=voice_list)

    console.print(f"[green]Generated audio to {output}[/green]")


@app.command()
def full(
    language: Annotated[str, typer.Option()] = "zh-CN",
    count: Annotated[int, typer.Option()] = 50,
    tags: Annotated[Optional[str], typer.Option()] = None,
    output: Annotated[Path, typer.Option()] = Path("output"),
) -> None:
    """Full pipeline: generate text and audio."""
    # Generate text
    generate(language=language, count=count, tags=tags, output=output)

    # Generate audio
    json_path = output / f"sentences.{language.split('-')[0]}.json"
    audio(input_file=json_path, output=output / "examples")


@app.command()
def validate(
    input_file: Annotated[Path, typer.Argument(help="Dataset JSON to validate")],
) -> None:
    """Validate a dataset against the schema."""
    try:
        dataset = Dataset.from_json_file(input_file)
        console.print(f"[green]Valid dataset with {len(dataset.items)} items[/green]")
    except Exception as e:
        console.print(f"[red]Validation failed: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
```

---

### `config.py`

**Purpose**: Load configuration from environment.

**Implementation**:

```python
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from pathlib import Path


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""
    api_key: str
    model: str = "gpt-4"
    max_tokens: int = 4096
    temperature: float = 0.7


class TTSConfig(BaseModel):
    """TTS configuration."""
    provider: str = "openai"  # or "azure", "google"
    # OpenAI TTS settings
    openai_voice_female: str = "nova"
    openai_voice_male: str = "onyx"
    # Audio settings
    format: str = "opus"
    sample_rate: int = 24000


class Config(BaseSettings):
    """Application configuration."""
    openai: OpenAIConfig
    tts: TTSConfig = Field(default_factory=TTSConfig)

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"


def load_config() -> Config:
    """Load configuration from environment."""
    return Config()
```

**Environment File (.env)**:

```bash
OPENAI__API_KEY=sk-...
OPENAI__MODEL=gpt-4
TTS__PROVIDER=openai
```

---

### `models/text_sequence.py`

**Purpose**: Pydantic models for text sequences.

**Implementation**:

```python
from pydantic import BaseModel, Field
from typing import Optional


class VoiceRef(BaseModel):
    """Reference to a pre-generated voice example."""
    id: str
    label: dict[str, str] = Field(default_factory=dict)
    uri: str
    duration_ms: Optional[int] = None


class ExampleAudio(BaseModel):
    """Example audio for a text sequence."""
    voices: list[VoiceRef] = Field(default_factory=list)


class TextSequence(BaseModel):
    """A text sequence for language learning."""
    id: str
    text: str
    romanization: Optional[str] = None
    gloss: dict[str, str] = Field(default_factory=dict)
    tokens: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    difficulty: Optional[int] = None
    example_audio: Optional[ExampleAudio] = None

    @classmethod
    def create(
        cls,
        index: int,
        text: str,
        romanization: str | None = None,
        translations: dict[str, str] | None = None,
        tags: list[str] | None = None,
        difficulty: int = 1,
    ) -> "TextSequence":
        """Create a new TextSequence with auto-generated ID."""
        return cls(
            id=f"ts_{index:06d}",
            text=text,
            romanization=romanization,
            gloss=translations or {},
            tags=tags or [],
            difficulty=difficulty,
        )
```

---

### `models/dataset.py`

**Purpose**: Dataset container model.

**Implementation**:

```python
from pydantic import BaseModel, Field
from datetime import datetime
from pathlib import Path
import json

from .text_sequence import TextSequence


class Dataset(BaseModel):
    """A complete dataset of text sequences."""
    schema_version: str = "1.0.0"
    dataset_id: str
    language: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    items: list[TextSequence] = Field(default_factory=list)

    @classmethod
    def from_json_file(cls, path: Path) -> "Dataset":
        """Load dataset from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.model_validate(data)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(indent=2)

    def add_item(self, item: TextSequence) -> None:
        """Add a text sequence to the dataset."""
        self.items.append(item)

    def get_by_id(self, id: str) -> TextSequence | None:
        """Get a text sequence by ID."""
        return next((item for item in self.items if item.id == id), None)
```

---

### `generators/text_generator.py`

**Purpose**: Generate text sequences using OpenAI.

**Implementation**:

```python
from openai import OpenAI
from pathlib import Path
import json

from ..config import Config
from ..models.dataset import Dataset
from ..models.text_sequence import TextSequence


class TextGenerator:
    """Generates text sequences using LLM."""

    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.openai.api_key)

    def generate(
        self,
        language: str,
        count: int,
        tags: list[str],
        difficulty: int = 1,
    ) -> Dataset:
        """Generate a dataset of text sequences."""
        prompt = self._load_prompt(language)
        system_prompt = self._build_system_prompt(language, tags, difficulty)

        # Generate in batches to avoid token limits
        batch_size = 20
        all_items: list[TextSequence] = []

        for batch_start in range(0, count, batch_size):
            batch_count = min(batch_size, count - batch_start)
            batch_items = self._generate_batch(
                system_prompt=system_prompt,
                user_prompt=prompt,
                count=batch_count,
                start_index=len(all_items) + 1,
            )
            all_items.extend(batch_items)

        # Create dataset
        dataset_id = f"{language.lower().replace('-', '_')}_v1"
        return Dataset(
            dataset_id=dataset_id,
            language=language,
            items=all_items,
        )

    def _load_prompt(self, language: str) -> str:
        """Load the generation prompt for a language."""
        lang_code = language.split("-")[0].lower()
        prompt_path = Path(__file__).parent.parent / "prompts" / f"{lang_code}_sentences.md"

        if not prompt_path.exists():
            raise ValueError(f"No prompt template for language: {language}")

        return prompt_path.read_text()

    def _build_system_prompt(
        self,
        language: str,
        tags: list[str],
        difficulty: int,
    ) -> str:
        """Build the system prompt for generation."""
        return f"""You are a language learning content generator.
Generate sentences in {language} for language learners.

Requirements:
- Difficulty level: {difficulty}/5
- Tags to include: {', '.join(tags) if tags else 'general'}
- Include romanization (pinyin for Chinese, romaji for Japanese)
- Include English translation
- Keep sentences practical and commonly used
- Vary sentence structures

Output format: JSON array of objects with keys:
- text: the sentence in target language
- romanization: pronunciation guide
- translation: English translation
- tokens: array of individual words/characters
- tags: relevant tags"""

    def _generate_batch(
        self,
        system_prompt: str,
        user_prompt: str,
        count: int,
        start_index: int,
    ) -> list[TextSequence]:
        """Generate a batch of sequences."""
        response = self.client.chat.completions.create(
            model=self.config.openai.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{user_prompt}\n\nGenerate {count} sentences."},
            ],
            max_tokens=self.config.openai.max_tokens,
            temperature=self.config.openai.temperature,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        data = json.loads(content)

        # Handle different response formats
        items_data = data.get("sentences") or data.get("items") or data

        if not isinstance(items_data, list):
            items_data = [items_data]

        items = []
        for i, item in enumerate(items_data):
            sequence = TextSequence.create(
                index=start_index + i,
                text=item.get("text", ""),
                romanization=item.get("romanization"),
                translations={"en": item.get("translation", "")},
                tags=item.get("tags", []),
                difficulty=item.get("difficulty", 1),
            )
            items.append(sequence)

        return items
```

---

### `generators/audio_generator.py`

**Purpose**: Generate TTS audio files.

**Implementation**:

```python
from openai import OpenAI
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor

from ..config import Config, TTSConfig
from ..models.dataset import Dataset
from ..models.text_sequence import TextSequence, VoiceRef, ExampleAudio


class AudioGenerator:
    """Generates TTS audio for text sequences."""

    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.openai.api_key)

    def generate_all(
        self,
        dataset: Dataset,
        output_dir: Path,
        voices: list[str],
    ) -> None:
        """Generate audio for all sequences in dataset."""
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
        """Generate audio for a single sequence."""
        output_path = output_dir / voice_name / f"{item.id}.opus"

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
        """Add audio reference to item."""
        relative_path = audio_path.relative_to(base_dir.parent)
        uri = f"assets://examples/{voice_name}/{item.id}.opus"

        voice_ref = VoiceRef(
            id=voice_name[0] + "1",  # f1, m1
            label={"en": voice_name.title()},
            uri=uri,
        )

        if item.example_audio is None:
            item.example_audio = ExampleAudio()

        # Remove existing voice if present
        item.example_audio.voices = [
            v for v in item.example_audio.voices if v.id != voice_ref.id
        ]
        item.example_audio.voices.append(voice_ref)
```

---

### `prompts/zh_sentences.md`

**Purpose**: Prompt template for Chinese sentence generation.

**Content**:

```markdown
# Chinese Sentence Generation

Generate practical Mandarin Chinese sentences for language learners.

## Guidelines

1. **Sentence Structure**
   - Use common grammatical patterns
   - Start with simpler structures for lower difficulty
   - Include a variety: statements, questions, commands

2. **Vocabulary**
   - Use HSK-appropriate vocabulary for the difficulty level
   - Include high-frequency words
   - Mix formal and informal registers

3. **Topics**
   - Daily life: greetings, shopping, directions
   - Travel: transportation, hotels, restaurants
   - Work: meetings, emails, presentations
   - Social: friends, family, hobbies

4. **Romanization**
   - Use standard pinyin with tone marks
   - Include tone numbers as alternative: wǒ (wo3)

5. **Tokens**
   - Split by natural word boundaries
   - Keep measure words with nouns

## Examples

Difficulty 1 (Beginner):
- 你好！ (nǐ hǎo) - Hello!
- 谢谢。 (xiè xie) - Thank you.

Difficulty 2 (Elementary):
- 我想喝水。 (wǒ xiǎng hē shuǐ) - I want to drink water.
- 这个多少钱？ (zhè ge duō shǎo qián) - How much is this?

Difficulty 3 (Intermediate):
- 请问，去火车站怎么走？ (qǐng wèn, qù huǒ chē zhàn zěn me zǒu) - Excuse me, how do I get to the train station?
```

---

### `exporters/json_exporter.py`

**Purpose**: Export dataset to JSON file.

**Implementation**:

```python
from pathlib import Path
import json
from datetime import datetime

from ..models.dataset import Dataset


class JsonExporter:
    """Exports datasets to JSON format."""

    def export(self, dataset: Dataset, output_path: Path) -> None:
        """Export dataset to JSON file."""
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Custom serializer for datetime
        def default_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat() + "Z"
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        # Write JSON
        with open(output_path, "w", encoding="utf-8") as f:
            data = dataset.model_dump()
            json.dump(data, f, indent=2, ensure_ascii=False, default=default_serializer)

    def export_for_flutter(
        self,
        dataset: Dataset,
        audio_dir: Path,
        flutter_assets: Path,
    ) -> None:
        """Export dataset and audio to Flutter assets directory."""
        # Export JSON
        json_path = flutter_assets / "datasets" / f"sentences.{dataset.language.split('-')[0]}.json"
        self.export(dataset, json_path)

        # Copy audio files
        import shutil
        audio_dest = flutter_assets / "examples"
        if audio_dir.exists():
            shutil.copytree(audio_dir, audio_dest, dirs_exist_ok=True)
```

---

## Tests

### `test_models.py`

```python
import pytest
from sentence_gen.models.text_sequence import TextSequence
from sentence_gen.models.dataset import Dataset


def test_text_sequence_create():
    seq = TextSequence.create(
        index=1,
        text="你好",
        romanization="nǐ hǎo",
        translations={"en": "Hello"},
        tags=["greeting"],
    )

    assert seq.id == "ts_000001"
    assert seq.text == "你好"
    assert seq.romanization == "nǐ hǎo"
    assert seq.gloss["en"] == "Hello"
    assert "greeting" in seq.tags


def test_dataset_add_item():
    dataset = Dataset(dataset_id="test_v1", language="zh-CN")

    seq = TextSequence.create(index=1, text="你好")
    dataset.add_item(seq)

    assert len(dataset.items) == 1
    assert dataset.get_by_id("ts_000001") is not None


def test_dataset_json_round_trip(tmp_path):
    dataset = Dataset(dataset_id="test_v1", language="zh-CN")
    dataset.add_item(TextSequence.create(index=1, text="你好"))

    json_path = tmp_path / "test.json"
    with open(json_path, "w") as f:
        f.write(dataset.to_json())

    loaded = Dataset.from_json_file(json_path)

    assert loaded.dataset_id == dataset.dataset_id
    assert len(loaded.items) == 1
    assert loaded.items[0].text == "你好"
```

### `test_text_generator.py`

```python
import pytest
from unittest.mock import Mock, patch

from sentence_gen.generators.text_generator import TextGenerator
from sentence_gen.config import Config, OpenAIConfig


@pytest.fixture
def mock_config():
    return Config(
        openai=OpenAIConfig(api_key="test-key", model="gpt-4")
    )


def test_generate_creates_dataset(mock_config):
    generator = TextGenerator(mock_config)

    mock_response = Mock()
    mock_response.choices = [
        Mock(message=Mock(content='{"sentences": [{"text": "你好", "romanization": "nǐ hǎo", "translation": "Hello"}]}'))
    ]

    with patch.object(generator.client.chat.completions, "create", return_value=mock_response):
        dataset = generator.generate(
            language="zh-CN",
            count=1,
            tags=["greeting"],
        )

    assert len(dataset.items) == 1
    assert dataset.items[0].text == "你好"
    assert dataset.language == "zh-CN"
```

---

## Usage Examples

### Generate Dataset

```bash
# Basic generation
sentence-gen generate --language zh-CN --count 50

# With tags and difficulty
sentence-gen generate --language zh-CN --count 100 --tags hsk1,daily --difficulty 1

# Full pipeline with audio
sentence-gen full --language zh-CN --count 50 --tags hsk1
```

### Export to Flutter

```bash
# Export to Flutter assets
sentence-gen export \
  --input output/ \
  --flutter-assets ../apps/mobile_flutter/assets/
```

### Validation

```bash
# Validate generated dataset
sentence-gen validate output/sentences.zh.json
```

---

## Notes

### Rate Limiting

OpenAI APIs have rate limits. The generator:
- Batches text generation (20 per request)
- Parallelizes audio generation (4 concurrent)
- Add delays if hitting limits

### Audio Format

- **Opus in OGG**: Best compression for speech
- **Bitrate**: ~24kbps (OpenAI TTS default)
- **File size**: ~3-5KB per second of audio

### Extending for New Languages

1. Create `prompts/{lang}_sentences.md`
2. Add voice configuration in `config.py`
3. Test with small batch first
4. Verify romanization is correct for the language
