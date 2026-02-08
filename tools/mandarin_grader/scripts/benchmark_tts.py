#!/usr/bin/env python3
"""Benchmark deterministic scorer on TTS assets.

This script evaluates the DeterministicScorer on the 60 TTS audio files
generated for the app. Since these are TTS recordings, the expected tone
accuracy should be near-perfect (ground truth).

Metrics reported:
- Tone classification accuracy (per tone and overall)
- Score distributions
- Confusion matrix
- Per-configuration comparison (uniform vs DTW, rule vs template)

Usage:
    cd tools/mandarin_grader
    python scripts/benchmark_tts.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mandarin_grader import DeterministicScorer, ScorerConfig
from mandarin_grader.align import DTWAligner, UniformAligner
from mandarin_grader.data import SentenceDataset
from mandarin_grader.sandhi import apply_tone_sandhi
from mandarin_grader.tone import RuleBasedClassifier, TemplateClassifier


def load_audio_pydub(path: Path, sr: int = 16000) -> np.ndarray:
    """Load audio using pydub (ffmpeg backend)."""
    from pydub import AudioSegment

    audio_seg = AudioSegment.from_file(str(path))
    # Convert to mono and resample
    audio_seg = audio_seg.set_channels(1).set_frame_rate(sr)
    # Get raw samples as numpy array
    samples = np.array(audio_seg.get_array_of_samples())
    # Normalize to [-1, 1]
    max_val = float(2 ** (audio_seg.sample_width * 8 - 1))
    return samples.astype(np.float32) / max_val


def load_audio(path: Path, sr: int = 16000) -> np.ndarray:
    """Load audio with fallback to pydub if librosa not available."""
    try:
        from mandarin_grader.data import load_audio as librosa_load
        return librosa_load(path, sr)
    except ImportError:
        return load_audio_pydub(path, sr)


@dataclass
class BenchmarkResult:
    """Results from a single scorer configuration."""

    scorer_name: str
    total_syllables: int = 0
    correct_tones: int = 0
    tone_counts: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    tone_correct: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    confusion_matrix: dict[tuple[int, int], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    scores: list[float] = field(default_factory=list)
    processing_time_s: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if self.total_syllables == 0:
            return 0.0
        return self.correct_tones / self.total_syllables

    @property
    def per_tone_accuracy(self) -> dict[int, float]:
        result = {}
        for tone in range(5):
            count = self.tone_counts[tone]
            if count > 0:
                result[tone] = self.tone_correct[tone] / count
            else:
                result[tone] = 0.0
        return result


def run_benchmark(
    dataset: SentenceDataset,
    scorer: DeterministicScorer,
    sr: int = 16000,
) -> BenchmarkResult:
    """Run benchmark on dataset with given scorer.

    Args:
        dataset: Dataset of TTS samples.
        scorer: Scorer instance to evaluate.
        sr: Sample rate.

    Returns:
        BenchmarkResult with metrics.
    """
    result = BenchmarkResult(scorer_name=scorer.name)
    start_time = time.time()

    for sample in dataset:
        try:
            # Load audio
            audio = load_audio(sample.audio_path, sr=sr)
            if audio is None or len(audio) == 0:
                result.errors.append(f"{sample.id}: failed to load audio")
                continue

            # Get targets with sandhi applied
            targets = sample.syllables
            if not targets:
                result.errors.append(f"{sample.id}: no syllables")
                continue

            sandhi_targets = apply_tone_sandhi(targets)

            # Score
            score_result = scorer.score(audio, targets, sr=sr)
            result.scores.append(score_result.overall)

            # Check tone predictions
            for i, (target, syl_score) in enumerate(
                zip(sandhi_targets, score_result.syllables)
            ):
                expected_tone = target.tone_surface
                result.total_syllables += 1
                result.tone_counts[expected_tone] += 1

                # Get predicted tone from tags
                predicted_tone = expected_tone  # Default to correct if no mismatch tag
                for tag in syl_score.tags:
                    if tag.startswith("tone_") and "_vs_" in tag:
                        # Format: tone_X_vs_Y where X is predicted, Y is expected
                        parts = tag.split("_")
                        if len(parts) >= 4:
                            predicted_tone = int(parts[1])
                            break

                if predicted_tone == expected_tone:
                    result.correct_tones += 1
                    result.tone_correct[expected_tone] += 1

                result.confusion_matrix[(expected_tone, predicted_tone)] += 1

        except Exception as e:
            result.errors.append(f"{sample.id}: {e}")

    result.processing_time_s = time.time() - start_time
    return result


def format_confusion_matrix(result: BenchmarkResult) -> str:
    """Format confusion matrix as string."""
    lines = []
    lines.append("Confusion Matrix (rows=expected, cols=predicted):")
    lines.append("       T0    T1    T2    T3    T4")
    lines.append("     " + "-" * 30)

    for expected in range(5):
        row = f"  T{expected} |"
        for predicted in range(5):
            count = result.confusion_matrix.get((expected, predicted), 0)
            row += f" {count:4d} "
        lines.append(row)

    return "\n".join(lines)


def format_results(results: list[BenchmarkResult]) -> str:
    """Format all results as markdown."""
    lines = []
    lines.append("# Benchmark Results")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Scorer | Accuracy | Avg Score | Time (s) | Errors |")
    lines.append("|--------|----------|-----------|----------|--------|")

    for r in results:
        avg_score = np.mean(r.scores) if r.scores else 0.0
        lines.append(
            f"| {r.scorer_name} | {r.accuracy:.1%} | {avg_score:.1f} | "
            f"{r.processing_time_s:.2f} | {len(r.errors)} |"
        )

    lines.append("")

    # Per-tone accuracy
    lines.append("## Per-Tone Accuracy")
    lines.append("")
    lines.append("| Scorer | T0 | T1 | T2 | T3 | T4 |")
    lines.append("|--------|----|----|----|----|----| ")

    for r in results:
        acc = r.per_tone_accuracy
        lines.append(
            f"| {r.scorer_name} | "
            f"{acc[0]:.0%} | {acc[1]:.0%} | {acc[2]:.0%} | "
            f"{acc[3]:.0%} | {acc[4]:.0%} |"
        )

    lines.append("")

    # Detailed results per scorer
    for r in results:
        lines.append(f"## {r.scorer_name}")
        lines.append("")
        lines.append(f"- Total syllables: {r.total_syllables}")
        lines.append(f"- Correct predictions: {r.correct_tones}")
        lines.append(f"- Overall accuracy: {r.accuracy:.1%}")
        lines.append(f"- Average score: {np.mean(r.scores):.1f}")
        lines.append(f"- Score std dev: {np.std(r.scores):.1f}")
        lines.append("")
        lines.append("```")
        lines.append(format_confusion_matrix(r))
        lines.append("```")
        lines.append("")

        if r.errors:
            lines.append("### Errors")
            lines.append("")
            for err in r.errors[:10]:  # First 10
                lines.append(f"- {err}")
            if len(r.errors) > 10:
                lines.append(f"- ... and {len(r.errors) - 10} more")
            lines.append("")

    # Analysis
    lines.append("## Analysis")
    lines.append("")

    if len(results) >= 2:
        best = max(results, key=lambda r: r.accuracy)
        lines.append(f"**Best configuration:** {best.scorer_name} ({best.accuracy:.1%})")
        lines.append("")

        # Common confusions
        all_confusions: Counter = Counter()
        for r in results:
            for (exp, pred), count in r.confusion_matrix.items():
                if exp != pred:
                    all_confusions[(exp, pred)] += count

        if all_confusions:
            lines.append("**Most common confusions:**")
            lines.append("")
            for (exp, pred), count in all_confusions.most_common(5):
                lines.append(f"- T{exp} -> T{pred}: {count} occurrences")
            lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")

    if results:
        best_acc = max(r.accuracy for r in results)
        if best_acc >= 0.8:
            lines.append(
                "[OK] Accuracy >=80% achieved. Deterministic approach is viable."
            )
        else:
            lines.append(
                "[WARNING] Accuracy <80%. Consider investigating confusions or adding DL."
            )
            lines.append("")
            lines.append("Potential improvements:")
            lines.append("- Tune classifier thresholds based on confusion patterns")
            lines.append("- Use DTW alignment with TTS reference audio")
            lines.append("- Derive empirical templates from TTS data")

    return "\n".join(lines)


def main() -> None:
    """Run benchmark on TTS assets."""
    # Find paths relative to script
    project_root = Path(__file__).parent.parent.parent.parent
    assets_dir = project_root / "apps" / "mobile_flutter" / "assets"
    sentences_json = assets_dir / "datasets" / "sentences.zh.json"
    audio_dir = assets_dir / "examples"

    print(f"Project root: {project_root}")
    print(f"Sentences JSON: {sentences_json}")
    print(f"Audio dir: {audio_dir}")

    if not sentences_json.exists():
        print(f"ERROR: {sentences_json} not found")
        sys.exit(1)

    if not audio_dir.exists():
        print(f"ERROR: {audio_dir} not found")
        sys.exit(1)

    # Load dataset
    print("\nLoading dataset...")
    dataset = SentenceDataset.from_app_assets(
        sentences_json=sentences_json,
        audio_dir=audio_dir,
        voice="female",
    )
    print(f"Loaded {len(dataset)} samples")

    if len(dataset) == 0:
        print("ERROR: No samples loaded")
        sys.exit(1)

    # Define scorer configurations to test
    configs = [
        ("uniform_rule", UniformAligner(), RuleBasedClassifier()),
        ("uniform_template", UniformAligner(), TemplateClassifier()),
    ]

    results = []

    for name, aligner, classifier in configs:
        print(f"\nBenchmarking {name}...")
        scorer = DeterministicScorer(
            config=ScorerConfig(),
            aligner=aligner,
            classifier=classifier,
        )

        result = run_benchmark(dataset, scorer)
        results.append(result)
        print(f"  Accuracy: {result.accuracy:.1%}")
        print(f"  Avg score: {np.mean(result.scores):.1f}")
        print(f"  Time: {result.processing_time_s:.2f}s")
        if result.errors:
            print(f"  Errors: {len(result.errors)}")

    # Generate report
    print("\n" + "=" * 60)
    report = format_results(results)
    print(report)

    # Save report
    output_path = Path(__file__).parent.parent / "BENCHMARK_RESULTS.md"
    output_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
