"""Text sequence generator using OpenAI LLM."""

from openai import OpenAI
from pathlib import Path
import json

from ..config import Config
from ..models.dataset import Dataset
from ..models.text_sequence import TextSequence


class TextGenerator:
    """Generates text sequences using LLM."""

    def __init__(self, config: Config) -> None:
        """Initialize the text generator.

        Args:
            config: Application configuration with OpenAI settings
        """
        self.config = config
        self.client = OpenAI(api_key=config.openai.api_key)

    def generate(
        self,
        language: str,
        count: int,
        tags: list[str],
        difficulty: int = 1,
    ) -> Dataset:
        """Generate a dataset of text sequences.

        Args:
            language: Target language code (e.g., "zh-CN")
            count: Number of sequences to generate
            tags: Tags to include in generation (e.g., ["hsk1", "daily"])
            difficulty: Difficulty level (1-5)

        Returns:
            Dataset containing generated TextSequence items
        """
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
        """Load the generation prompt for a language.

        Args:
            language: Language code (e.g., "zh-CN")

        Returns:
            The prompt template content

        Raises:
            ValueError: If no prompt template exists for the language
        """
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
        """Build the system prompt for generation.

        Args:
            language: Target language code
            tags: Tags to incorporate in generation
            difficulty: Difficulty level (1-5)

        Returns:
            System prompt string for the LLM
        """
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
        """Generate a batch of sequences.

        Args:
            system_prompt: The system prompt for the LLM
            user_prompt: The user prompt (language-specific template)
            count: Number of sequences in this batch
            start_index: Starting index for ID generation

        Returns:
            List of generated TextSequence items
        """
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
