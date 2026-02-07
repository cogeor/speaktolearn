"""Unit tests for Pydantic models."""

import pytest
from text_gen.models.text_sequence import TextSequence
from text_gen.models.dataset import Dataset


def test_text_sequence_create():
    """Verify TextSequence.create() generates correct ID format ts_NNNNNN."""
    seq = TextSequence.create(
        index=1,
        text="你好",
        romanization="ni hao",
        translations={"en": "Hello"},
        tags=["greeting"],
    )

    assert seq.id == "ts_000001"
    assert seq.text == "你好"
    assert seq.romanization == "ni hao"
    assert seq.gloss["en"] == "Hello"
    assert "greeting" in seq.tags


def test_text_sequence_create_large_index():
    """Verify ID format works for larger indices."""
    seq = TextSequence.create(index=123456, text="test")
    assert seq.id == "ts_123456"

    seq2 = TextSequence.create(index=42, text="test")
    assert seq2.id == "ts_000042"


def test_dataset_add_item():
    """Verify items list management with add_item and get_by_id."""
    dataset = Dataset(dataset_id="test_v1", language="zh-CN")

    seq1 = TextSequence.create(index=1, text="你好")
    seq2 = TextSequence.create(index=2, text="再见")

    dataset.add_item(seq1)
    assert len(dataset.items) == 1

    dataset.add_item(seq2)
    assert len(dataset.items) == 2

    # Verify get_by_id works
    found = dataset.get_by_id("ts_000001")
    assert found is not None
    assert found.text == "你好"

    found2 = dataset.get_by_id("ts_000002")
    assert found2 is not None
    assert found2.text == "再见"

    # Verify get_by_id returns None for non-existent ID
    not_found = dataset.get_by_id("ts_999999")
    assert not_found is None


def test_dataset_json_round_trip(tmp_path):
    """Verify dataset can be serialized to JSON and loaded back."""
    dataset = Dataset(dataset_id="test_v1", language="zh-CN")
    dataset.add_item(TextSequence.create(index=1, text="你好", romanization="ni hao"))
    dataset.add_item(TextSequence.create(index=2, text="谢谢", romanization="xie xie"))

    # Serialize to JSON and save to file
    json_path = tmp_path / "test.json"
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(dataset.to_json())

    # Load back from JSON
    loaded = Dataset.from_json_file(json_path)

    # Verify loaded dataset matches original
    assert loaded.dataset_id == dataset.dataset_id
    assert loaded.language == dataset.language
    assert loaded.schema_version == dataset.schema_version
    assert len(loaded.items) == 2

    # Verify items are correctly deserialized
    assert loaded.items[0].text == "你好"
    assert loaded.items[0].romanization == "ni hao"
    assert loaded.items[1].text == "谢谢"
    assert loaded.items[1].romanization == "xie xie"
