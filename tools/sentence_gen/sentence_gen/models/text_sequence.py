"""Text sequence model for language learning content."""

from pydantic import BaseModel, Field
from typing import Optional

from .voice import VoiceRef


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
        """Create a new TextSequence with auto-generated ID.

        Args:
            index: Numeric index for ID generation (produces ts_NNNNNN format)
            text: The text content in target language
            romanization: Pronunciation guide (pinyin, romaji, etc.)
            translations: Dictionary of translations keyed by language code
            tags: List of tags for categorization
            difficulty: Difficulty level (default 1)

        Returns:
            A new TextSequence instance with generated ID
        """
        return cls(
            id=f"ts_{index:06d}",
            text=text,
            romanization=romanization,
            gloss=translations or {},
            tags=tags or [],
            difficulty=difficulty,
        )
