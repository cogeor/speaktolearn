"""JSON exporter for datasets."""

from pathlib import Path
import json
import shutil
from datetime import datetime

from ..models.dataset import Dataset


class JsonExporter:
    """Exports datasets to JSON format."""

    def export(self, dataset: Dataset, output_path: Path) -> None:
        """Export dataset to JSON file.

        Args:
            dataset: The dataset to export
            output_path: Path where the JSON file will be written
        """
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Custom serializer for datetime (ISO 8601 format)
        def default_serializer(obj):
            if isinstance(obj, datetime):
                # Use Z suffix for UTC, otherwise use offset
                iso = obj.isoformat()
                if iso.endswith('+00:00'):
                    return iso.replace('+00:00', 'Z')
                return iso
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
        """Export dataset and audio to Flutter assets directory.

        Args:
            dataset: The dataset to export
            audio_dir: Directory containing the generated audio files
            flutter_assets: Flutter assets directory to copy files to
        """
        # Export JSON to datasets subdirectory
        lang_code = dataset.language.split("-")[0]
        json_path = flutter_assets / "datasets" / f"sentences.{lang_code}.json"
        self.export(dataset, json_path)

        # Copy audio files if they exist
        audio_dest = flutter_assets / "examples"
        if audio_dir.exists():
            shutil.copytree(audio_dir, audio_dest, dirs_exist_ok=True)
