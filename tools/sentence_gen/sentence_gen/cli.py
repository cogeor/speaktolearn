"""CLI entrypoint for sentence-gen command."""

import typer

app = typer.Typer(help="Generate language learning datasets")


@app.command()
def placeholder() -> None:
    """Placeholder command - implementation pending."""
    typer.echo("sentence-gen CLI - implementation pending")


if __name__ == "__main__":
    app()
