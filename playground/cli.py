from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a document OCR engine on a file or folder and print text to stdout.",
    )
    parser.add_argument(
        "engine",
        choices=["surya", "olmocr"],
        help="OCR engine to run.",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Image, PDF, or directory of images/PDFs (Surya). PDF or image for olmocr when installed.",
    )
    parser.add_argument(
        "--page-range",
        dest="page_range",
        default=None,
        metavar="RANGE",
        help="Surya only: pages for PDFs, e.g. 0,5-10,20",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    path: Path = args.path.expanduser()
    if not path.exists():
        print(f"Path not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.engine == "surya":
            from playground.engines import surya as surya_engine

            text = surya_engine.run(path, page_range=args.page_range)
        elif args.engine == "olmocr":
            from playground.engines import olmocr as olmocr_engine

            text = olmocr_engine.run(path)
    except RuntimeError as err:
        print(err, file=sys.stderr)
        sys.exit(1)

    print(text, end="" if text.endswith("\n") else "\n")

