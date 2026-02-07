"""Dataset container model for text sequences."""

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
        """Load dataset from JSON file.

        Args:
            path: Path to the JSON file

        Returns:
            A Dataset instance populated from the JSON data
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)

    def to_json(self) -> str:
        """Serialize to JSON string.

        Returns:
            JSON string representation of the dataset
        """
        return self.model_dump_json(indent=2)

    def add_item(self, item: TextSequence) -> None:
        """Add a text sequence to the dataset.

        Args:
            item: The TextSequence to add
        """
        self.items.append(item)

    def get_by_id(self, id: str) -> TextSequence | None:
        """Get a text sequence by ID.

        Args:
            id: The sequence ID to find

        Returns:
            The TextSequence if found, None otherwise
        """
        return next((item for item in self.items if item.id == id), None)
