#!/usr/bin/env python3
"""Generate syllable coverage statistics for the synthetic data lexicon.

This script analyzes the syllable lexicon and sentence dataset to report:
- Count of each syllable in the lexicon
- Tone distribution
- Under-represented syllables
- Missing syllables

Usage:
    python syllable_stats.py --lexicon data/syllables
    python syllable_stats.py --sentences sentences.zh.json --output stats.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Add parent package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mandarin_grader.data.lexicon import SyllableLexicon, extract_unique_syllables
from mandarin_grader.data.dataloader import parse_romanization


def analyze_lexicon(lexicon: SyllableLexicon) -> dict:
    """Analyze syllable coverage in the lexicon.

    Args:
        lexicon: SyllableLexicon to analyze

    Returns:
        Dictionary with analysis results
    """
    entries = list(lexicon)

    # Count by voice
    voice_counts = Counter(e.voice_id for e in entries)

    # Count by tone
    tone_counts = Counter(e.tone for e in entries)

    # Count by syllable
    syllable_counts = Counter(e.syllable_key for e in entries)

    # Group by pinyin (ignoring tone)
    pinyin_counts = Counter(e.pinyin for e in entries)

    # Duration statistics
    durations = [e.duration_ms for e in entries if e.duration_ms > 0]
    duration_stats = {
        "min_ms": min(durations) if durations else 0,
        "max_ms": max(durations) if durations else 0,
        "mean_ms": sum(durations) / len(durations) if durations else 0,
    }

    return {
        "total_entries": len(entries),
        "voice_counts": dict(voice_counts),
        "tone_counts": dict(tone_counts),
        "syllable_counts": dict(syllable_counts),
        "pinyin_counts": dict(pinyin_counts),
        "duration_stats": duration_stats,
        "unique_syllables": len(syllable_counts),
        "unique_pinyins": len(pinyin_counts),
    }


def analyze_sentence_coverage(
    sentences_json: Path,
    lexicon: SyllableLexicon,
    voice: str = "female",
) -> dict:
    """Analyze syllable coverage for sentences in the dataset.

    Args:
        sentences_json: Path to sentences.zh.json
        lexicon: SyllableLexicon to check against
        voice: Voice to check coverage for

    Returns:
        Dictionary with coverage analysis
    """
    with open(sentences_json, encoding="utf-8") as f:
        data = json.load(f)

    # Track syllable occurrences in sentences
    syllable_in_sentences = Counter()
    missing_syllables = set()
    covered_syllables = set()

    for item in data.get("items", []):
        romanization = item.get("romanization", "")
        text = item.get("text", "")

        if not romanization:
            continue

        syllables = parse_romanization(romanization, text)
        for syl in syllables:
            # Get base pinyin
            from mandarin_grader.data.lexicon import _remove_tone_marks
            base_pinyin = _remove_tone_marks(syl.pinyin)
            tone = syl.tone_surface if syl.tone_surface else syl.tone_underlying

            key = f"{base_pinyin}{tone}"
            syllable_in_sentences[key] += 1

            if lexicon.has(base_pinyin, tone, voice):
                covered_syllables.add(key)
            else:
                missing_syllables.add(key)

    # Find under-represented syllables (appear less than 3 times)
    under_represented = {
        k: v for k, v in syllable_in_sentences.items()
        if v < 3 and k in covered_syllables
    }

    return {
        "total_sentences": len(data.get("items", [])),
        "total_syllable_occurrences": sum(syllable_in_sentences.values()),
        "unique_syllables_in_sentences": len(syllable_in_sentences),
        "covered_syllables": len(covered_syllables),
        "missing_syllables": sorted(missing_syllables),
        "under_represented": dict(sorted(under_represented.items(), key=lambda x: x[1])),
        "syllable_frequency": dict(syllable_in_sentences.most_common()),
    }


def generate_markdown_report(
    lexicon_analysis: dict,
    sentence_coverage: dict | None = None,
) -> str:
    """Generate a Markdown report of the analysis.

    Args:
        lexicon_analysis: Results from analyze_lexicon
        sentence_coverage: Results from analyze_sentence_coverage (optional)

    Returns:
        Markdown formatted report
    """
    lines = [
        "# Syllable Lexicon Statistics",
        "",
        "## Overview",
        "",
        f"- **Total entries:** {lexicon_analysis['total_entries']}",
        f"- **Unique syllables (pinyin+tone):** {lexicon_analysis['unique_syllables']}",
        f"- **Unique base pinyins:** {lexicon_analysis['unique_pinyins']}",
        "",
        "## Voice Distribution",
        "",
    ]

    for voice, count in sorted(lexicon_analysis['voice_counts'].items()):
        lines.append(f"- {voice}: {count} entries")

    lines.extend([
        "",
        "## Tone Distribution",
        "",
        "| Tone | Count | Description |",
        "|------|-------|-------------|",
    ])

    tone_names = {
        0: "Neutral",
        1: "High level (ˉ)",
        2: "Rising (ˊ)",
        3: "Dipping (ˇ)",
        4: "Falling (ˋ)",
    }

    for tone in sorted(lexicon_analysis['tone_counts'].keys()):
        count = lexicon_analysis['tone_counts'][tone]
        name = tone_names.get(tone, "Unknown")
        lines.append(f"| {tone} | {count} | {name} |")

    lines.extend([
        "",
        "## Duration Statistics",
        "",
        f"- **Min:** {lexicon_analysis['duration_stats']['min_ms']}ms",
        f"- **Max:** {lexicon_analysis['duration_stats']['max_ms']}ms",
        f"- **Mean:** {lexicon_analysis['duration_stats']['mean_ms']:.1f}ms",
        "",
    ])

    if sentence_coverage:
        coverage_pct = (
            sentence_coverage['covered_syllables'] /
            sentence_coverage['unique_syllables_in_sentences'] * 100
            if sentence_coverage['unique_syllables_in_sentences'] > 0 else 0
        )

        lines.extend([
            "## Sentence Dataset Coverage",
            "",
            f"- **Total sentences:** {sentence_coverage['total_sentences']}",
            f"- **Total syllable occurrences:** {sentence_coverage['total_syllable_occurrences']}",
            f"- **Unique syllables in dataset:** {sentence_coverage['unique_syllables_in_sentences']}",
            f"- **Covered by lexicon:** {sentence_coverage['covered_syllables']} ({coverage_pct:.1f}%)",
            "",
        ])

        if sentence_coverage['missing_syllables']:
            lines.extend([
                "### Missing Syllables",
                "",
                "The following syllables appear in sentences but are not in the lexicon:",
                "",
            ])
            for syl in sentence_coverage['missing_syllables'][:20]:
                lines.append(f"- `{syl}`")
            if len(sentence_coverage['missing_syllables']) > 20:
                lines.append(f"- ... and {len(sentence_coverage['missing_syllables']) - 20} more")
            lines.append("")

        if sentence_coverage['under_represented']:
            lines.extend([
                "### Under-Represented Syllables",
                "",
                "Syllables appearing less than 3 times in the dataset:",
                "",
                "| Syllable | Count |",
                "|----------|-------|",
            ])
            for syl, count in list(sentence_coverage['under_represented'].items())[:30]:
                lines.append(f"| `{syl}` | {count} |")
            lines.append("")

        # Top 20 most frequent
        lines.extend([
            "### Most Frequent Syllables",
            "",
            "| Rank | Syllable | Count |",
            "|------|----------|-------|",
        ])
        for i, (syl, count) in enumerate(list(sentence_coverage['syllable_frequency'].items())[:20], 1):
            lines.append(f"| {i} | `{syl}` | {count} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate syllable coverage statistics"
    )
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=Path("tools/mandarin_grader/data/syllables"),
        help="Path to syllable lexicon directory",
    )
    parser.add_argument(
        "--sentences",
        type=Path,
        default=Path("apps/mobile_flutter/assets/datasets/sentences.zh.json"),
        help="Path to sentences.zh.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for Markdown report (default: print to stdout)",
    )
    parser.add_argument(
        "--voice",
        default="female",
        help="Voice to check coverage for",
    )

    args = parser.parse_args()

    # Load lexicon
    print(f"Loading lexicon from {args.lexicon}...")
    lexicon = SyllableLexicon.load(args.lexicon)
    print(f"Loaded {len(lexicon)} entries")

    # Analyze lexicon
    print("Analyzing lexicon...")
    lexicon_analysis = analyze_lexicon(lexicon)

    # Analyze sentence coverage
    sentence_coverage = None
    if args.sentences.exists():
        print(f"Analyzing sentence coverage from {args.sentences}...")
        sentence_coverage = analyze_sentence_coverage(
            args.sentences, lexicon, args.voice
        )

    # Generate report
    report = generate_markdown_report(lexicon_analysis, sentence_coverage)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to {args.output}")
    else:
        print("\n" + report)


if __name__ == "__main__":
    main()
