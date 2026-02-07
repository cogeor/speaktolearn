"""Configuration loading for sentence-gen."""

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


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

    model_config = {
        "env_file": ".env",
        "env_nested_delimiter": "__",
    }


def load_config() -> Config:
    """Load configuration from environment."""
    return Config()
