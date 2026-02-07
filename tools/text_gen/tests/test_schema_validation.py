"""Schema validation tests.

Validates that Pydantic models produce output matching the JSON Schema contract.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, validate, ValidationError

from text_gen.models import Dataset, TextSequence, ExampleAudio, VoiceRef


# Paths to schema and example files
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SCHEMA_PATH = PROJECT_ROOT / "shared" / "data_schema" / "sentences.schema.json"
EXAMPLE_PATH = PROJECT_ROOT / "shared" / "data_schema" / "examples" / "sentences.zh.example.json"


@pytest.fixture
def schema() -> dict:
    """Load the JSON Schema."""
    with open(SCHEMA_PATH) as f:
        return json.load(f)


@pytest.fixture
def sample_dataset() -> Dataset:
    """Create a sample dataset using Pydantic models."""
    # Create voice references
    female_voice = VoiceRef(
        id="f1",
        label={"en": "Female", "zh": "女声"},
        uri="assets://examples/female/ts_000001.opus",
        duration_ms=850,
    )
    male_voice = VoiceRef(
        id="m1",
        label={"en": "Male", "zh": "男声"},
        uri="assets://examples/male/ts_000001.opus",
        duration_ms=920,
    )

    # Create example audio
    example_audio = ExampleAudio(voices=[female_voice, male_voice])

    # Create text sequence
    text_sequence = TextSequence(
        id="ts_000001",
        text="你好！",
        romanization="ni hao",
        gloss={"en": "Hello!", "de": "Hallo!"},
        tokens=["你", "好"],
        tags=["hsk1", "greeting"],
        difficulty=1,
        example_audio=example_audio,
    )

    # Create dataset
    dataset = Dataset(
        schema_version="1.0.0",
        dataset_id="test_dataset_v1",
        language="zh-CN",
        generated_at=datetime(2026, 2, 6, 10, 0, 0, tzinfo=timezone.utc),
        items=[text_sequence],
    )

    return dataset


def test_schema_is_valid(schema: dict) -> None:
    """Verify the JSON Schema itself is valid."""
    Draft202012Validator.check_schema(schema)


def test_pydantic_model_validates_against_schema(
    schema: dict, sample_dataset: Dataset
) -> None:
    """Verify Pydantic model output matches the JSON Schema."""
    # Export to JSON and parse back to dict
    json_str = sample_dataset.model_dump_json()
    data = json.loads(json_str)

    # Validate against schema - will raise ValidationError if invalid
    validate(instance=data, schema=schema)


def test_example_file_validates_against_schema(schema: dict) -> None:
    """Verify the example JSON file matches the JSON Schema."""
    with open(EXAMPLE_PATH) as f:
        example_data = json.load(f)

    # Validate against schema - will raise ValidationError if invalid
    validate(instance=example_data, schema=schema)


def test_minimal_dataset_validates(schema: dict) -> None:
    """Verify a minimal dataset with only required fields validates."""
    # Create minimal text sequence (only required fields: id, text)
    text_sequence = TextSequence(
        id="ts_000001",
        text="Hello",
    )

    dataset = Dataset(
        schema_version="1.0.0",
        dataset_id="minimal_v1",
        language="en",
        generated_at=datetime(2026, 2, 6, 10, 0, 0, tzinfo=timezone.utc),
        items=[text_sequence],
    )

    json_str = dataset.model_dump_json()
    data = json.loads(json_str)

    validate(instance=data, schema=schema)


def test_dataset_with_all_optional_fields_validates(schema: dict) -> None:
    """Verify a fully populated dataset validates against schema."""
    voice = VoiceRef(
        id="f1",
        label={"en": "Female"},
        uri="assets://audio/ts_000001.opus",
        duration_ms=1000,
    )

    text_sequence = TextSequence(
        id="ts_000001",
        text="测试",
        romanization="ce shi",
        gloss={"en": "test"},
        tokens=["测", "试"],
        tags=["test"],
        difficulty=3,
        example_audio=ExampleAudio(voices=[voice]),
    )

    dataset = Dataset(
        schema_version="1.0.0",
        dataset_id="full_v1",
        language="zh-CN",
        generated_at=datetime(2026, 2, 6, 10, 0, 0, tzinfo=timezone.utc),
        items=[text_sequence],
    )

    json_str = dataset.model_dump_json()
    data = json.loads(json_str)

    validate(instance=data, schema=schema)


def test_invalid_id_format_fails_schema(schema: dict) -> None:
    """Verify invalid ID format is rejected by schema."""
    text_sequence = TextSequence(
        id="invalid_id",  # Wrong format, should be ts_NNNNNN
        text="Test",
    )

    dataset = Dataset(
        schema_version="1.0.0",
        dataset_id="test_v1",
        language="en",
        generated_at=datetime(2026, 2, 6, 10, 0, 0, tzinfo=timezone.utc),
        items=[text_sequence],
    )

    json_str = dataset.model_dump_json()
    data = json.loads(json_str)

    with pytest.raises(ValidationError):
        validate(instance=data, schema=schema)


def test_invalid_language_format_fails_schema(schema: dict) -> None:
    """Verify invalid language code format is rejected by schema."""
    text_sequence = TextSequence(
        id="ts_000001",
        text="Test",
    )

    dataset = Dataset(
        schema_version="1.0.0",
        dataset_id="test_v1",
        language="invalid-language-code",  # Wrong format
        generated_at=datetime(2026, 2, 6, 10, 0, 0, tzinfo=timezone.utc),
        items=[text_sequence],
    )

    json_str = dataset.model_dump_json()
    data = json.loads(json_str)

    with pytest.raises(ValidationError):
        validate(instance=data, schema=schema)


def test_difficulty_out_of_range_fails_schema(schema: dict) -> None:
    """Verify difficulty out of range (1-5) is rejected by schema."""
    text_sequence = TextSequence(
        id="ts_000001",
        text="Test",
        difficulty=10,  # Invalid, max is 5
    )

    dataset = Dataset(
        schema_version="1.0.0",
        dataset_id="test_v1",
        language="en",
        generated_at=datetime(2026, 2, 6, 10, 0, 0, tzinfo=timezone.utc),
        items=[text_sequence],
    )

    json_str = dataset.model_dump_json()
    data = json.loads(json_str)

    with pytest.raises(ValidationError):
        validate(instance=data, schema=schema)
