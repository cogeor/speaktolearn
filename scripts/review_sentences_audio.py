#!/usr/bin/env python3
"""Review generated sentence/audio alignment outside Flutter.

Usage examples:
  python scripts/review_sentences_audio.py --level hsk1 --voice female --play
  python scripts/review_sentences_audio.py --id ts_000040 --voice male --play
  python scripts/review_sentences_audio.py --level hsk4 --contains "jin tian"
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TONE_MARK_CHARS = set("āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜü")
TONE_NUMBER_RE = re.compile(r"[a-zvü]+[1-5]", re.IGNORECASE)


@dataclass
class Row:
    seq_id: str
    text: str
    romanization: str
    hsk: str | None
    voice_uri: str | None
    file_path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("apps/mobile_flutter/assets/datasets/sentences.zh.json"),
        help="Path to dataset JSON",
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=Path("apps/mobile_flutter/assets"),
        help="Flutter assets root directory",
    )
    parser.add_argument(
        "--voice",
        default="female",
        help="Voice selector: female|male|f1|m1|exact voice id",
    )
    parser.add_argument(
        "--level",
        action="append",
        default=[],
        help="HSK filter tag (repeatable), e.g. --level hsk1 --level hsk4",
    )
    parser.add_argument(
        "--id",
        action="append",
        default=[],
        help="Sequence ID filter (repeatable), e.g. --id ts_000001",
    )
    parser.add_argument(
        "--contains",
        default="",
        help="Substring filter for text or romanization",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max rows (0 = all)")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle row order")
    parser.add_argument("--play", action="store_true", help="Play clips sequentially")
    parser.add_argument(
        "--strict-tone-mark",
        action="store_true",
        help="Flag rows whose romanization is not tone-mark style",
    )
    return parser.parse_args()


def read_dataset(dataset_path: Path) -> dict[str, Any]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    with dataset_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_hsk(tags: list[str]) -> str | None:
    for tag in tags:
        if re.fullmatch(r"hsk[1-6]", tag):
            return tag
    return None


def resolve_voice_uri(voices: list[dict[str, Any]], voice_selector: str) -> str | None:
    selector = voice_selector.strip().lower()
    if not voices:
        return None

    def by_id(v: dict[str, Any]) -> bool:
        return (v.get("id") or "").strip().lower() == selector

    def by_gender(v: dict[str, Any], gender: str) -> bool:
        uri = (v.get("uri") or "").lower()
        vid = (v.get("id") or "").lower()
        label = json.dumps(v.get("label", {})).lower()
        return f"/{gender}/" in uri or gender in vid or gender in label

    if selector in {"female", "male"}:
        match = next((v for v in voices if by_gender(v, selector)), None)
        if match:
            return match.get("uri")
    else:
        match = next((v for v in voices if by_id(v)), None)
        if match:
            return match.get("uri")

    return (voices[0] or {}).get("uri")


def asset_uri_to_path(uri: str, assets_root: Path) -> Path | None:
    if uri.startswith("assets://"):
        rel = uri[len("assets://") :]
        return assets_root / rel
    if uri.startswith("file://"):
        return Path(uri[len("file://") :])
    return None


def pinyin_style(romanization: str) -> str:
    if any(ch in TONE_MARK_CHARS for ch in romanization.lower()):
        return "tone-mark"
    if TONE_NUMBER_RE.search(romanization):
        return "tone-number"
    return "other"


def select_rows(data: dict[str, Any], args: argparse.Namespace) -> list[Row]:
    rows: list[Row] = []
    items = data.get("items", [])

    id_set = {x.strip() for x in args.id}
    level_set = {x.strip().lower() for x in args.level}
    contains = args.contains.strip().lower()

    for item in items:
        seq_id = item.get("id", "")
        text = item.get("text", "")
        romanization = item.get("romanization", "")
        tags = item.get("tags", []) or []
        hsk = get_hsk(tags)

        if id_set and seq_id not in id_set:
            continue
        if level_set and (hsk or "").lower() not in level_set:
            continue
        if contains:
            if contains not in text.lower() and contains not in romanization.lower():
                continue

        voices = (item.get("example_audio") or {}).get("voices") or []
        voice_uri = resolve_voice_uri(voices, args.voice)
        file_path = asset_uri_to_path(voice_uri, args.assets_root) if voice_uri else None

        rows.append(
            Row(
                seq_id=seq_id,
                text=text,
                romanization=romanization,
                hsk=hsk,
                voice_uri=voice_uri,
                file_path=file_path,
            )
        )

    if args.shuffle:
        random.shuffle(rows)
    else:
        rows.sort(key=lambda r: r.seq_id)

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    return rows


def play_with_ffplay(path: Path) -> bool:
    ffplay = shutil.which("ffplay")
    if not ffplay:
        return False
    cmd = [ffplay, "-v", "error", "-nodisp", "-autoexit", str(path)]
    subprocess.run(cmd, check=False)
    return True


def play_with_powershell_mediaplayer(path: Path) -> bool:
    if sys.platform != "win32":
        return False
    ps = shutil.which("powershell") or shutil.which("pwsh")
    if not ps:
        return False
    uri = path.resolve().as_uri()
    script = (
        "Add-Type -AssemblyName presentationCore; "
        "$p = New-Object System.Windows.Media.MediaPlayer; "
        f"$p.Open([Uri]'{uri}'); "
        "$p.Play(); "
        "while(-not $p.NaturalDuration.HasTimeSpan){ Start-Sleep -Milliseconds 100 }; "
        "while($p.Position -lt $p.NaturalDuration.TimeSpan){ Start-Sleep -Milliseconds 100 }; "
        "$p.Stop(); $p.Close();"
    )
    subprocess.run([ps, "-NoProfile", "-Command", script], check=False)
    return True


def play_audio(path: Path) -> bool:
    if play_with_ffplay(path):
        return True
    if play_with_powershell_mediaplayer(path):
        return True
    return False


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    args = parse_args()
    data = read_dataset(args.dataset)
    rows = select_rows(data, args)

    if not rows:
        print("No rows matched filters.")
        return 1

    print(f"dataset_id: {data.get('dataset_id')}")
    print(f"generated_at: {data.get('generated_at')}")
    print(f"rows: {len(rows)}")
    print("")

    played = 0
    missing_audio = 0
    tone_issues = 0

    for idx, row in enumerate(rows, start=1):
        style = pinyin_style(row.romanization)
        tone_bad = args.strict_tone_mark and style != "tone-mark"
        if tone_bad:
            tone_issues += 1

        exists = bool(row.file_path and row.file_path.exists())
        if not exists:
            missing_audio += 1

        print(f"[{idx}/{len(rows)}] {row.seq_id} {row.hsk or ''}")
        print(f"  text: {row.text}")
        print(f"  pinyin: {row.romanization} ({style})")
        print(f"  voice: {row.voice_uri}")
        print(f"  file: {row.file_path} {'OK' if exists else 'MISSING'}")
        if tone_bad:
            print("  !! tone-mark check failed")

        if args.play and exists and row.file_path:
            ok = play_audio(row.file_path)
            played += 1 if ok else 0
            if not ok:
                print("  !! no playback backend found (need ffplay or Windows MediaPlayer)")

        if args.play:
            resp = input("  Enter=next | q=quit > ").strip().lower()
            if resp == "q":
                break

        print("")

    print("Summary:")
    print(f"  total_rows: {len(rows)}")
    print(f"  missing_audio: {missing_audio}")
    print(f"  tone_mark_issues: {tone_issues}")
    if args.play:
        print(f"  played: {played}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
