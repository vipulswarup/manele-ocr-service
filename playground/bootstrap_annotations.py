#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create an empty JSONL transcription sheet from a folder of line crops."
    )
    parser.add_argument("line_dir", type=Path, help="Directory containing line crop images.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("annotations.jsonl"),
        help="Output JSONL file to fill with transcriptions.",
    )
    parser.add_argument(
        "--script-hint",
        default="",
        help="Optional page-level script hint to pre-fill into each record.",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Store absolute image paths instead of paths relative to the line directory.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    line_dir = args.line_dir.expanduser().resolve()
    if not line_dir.is_dir():
        raise SystemExit(f"Not a directory: {line_dir}")

    allowed = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
    images = [p for p in sorted(line_dir.iterdir()) if p.is_file() and p.suffix.lower() in allowed]
    if not images:
        raise SystemExit(f"No line crop images found in {line_dir}")

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for image_path in images:
            record = {
                "image": str(image_path if args.absolute_paths else image_path.name),
                "text": "",
                "script": args.script_hint,
                "notes": "",
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(images)} annotation rows to {output_path}")


if __name__ == "__main__":
    main()
