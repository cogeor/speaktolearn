"""Voice reference model for audio examples."""

from pydantic import BaseModel, Field
from typing import Optional


class VoiceRef(BaseModel):
    """Reference to a pre-generated voice example."""

    id: str
    label: dict[str, str] = Field(default_factory=dict)
    uri: str
    duration_ms: Optional[int] = None
