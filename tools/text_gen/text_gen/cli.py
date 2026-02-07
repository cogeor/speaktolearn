"""CLI entrypoint for sentence-gen command."""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console

from .config import load_config
from .exporters.json_exporter import JsonExporter
from .generators.audio_generator import AudioGenerator
from .generators.text_generator import TextGenerator
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
    language: Annotated[str, typer.Option(help="Target language code")] = "zh-CN",
    count: Annotated[int, typer.Option(help="Number of sequences")] = 50,
    tags: Annotated[Optional[str], typer.Option(help="Comma-separated tags")] = None,
    output: Annotated[Path, typer.Option(help="Output directory")] = Path("output"),
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


@app.command()
def export(
    input: Annotated[Path, typer.Option("--input", help="Input directory with JSON and audio")],
    flutter_assets: Annotated[Path, typer.Option("--flutter-assets", help="Flutter assets directory")],
) -> None:
    """Export dataset and audio to Flutter assets."""
    # Find the JSON file in input directory
    json_files = list(input.glob("sentences.*.json"))
    if not json_files:
        console.print(f"[red]No sentences JSON file found in {input}[/red]")
        raise typer.Exit(1)

    json_file = json_files[0]
    console.print(f"[bold]Loading dataset from {json_file}...[/bold]")
    dataset = Dataset.from_json_file(json_file)

    # Audio directory is input/examples
    audio_dir = input / "examples"

    exporter = JsonExporter()
    exporter.export_for_flutter(dataset, audio_dir, flutter_assets)

    console.print(f"[green]Exported to {flutter_assets}[/green]")


if __name__ == "__main__":
    app()
