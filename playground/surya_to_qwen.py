#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Surya OCR on an input page/file and summarize the OCR text with a local Ollama Qwen model."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input image, PDF, or directory accepted by the existing Surya runner.",
    )
    parser.add_argument(
        "--page-range",
        dest="page_range",
        default=None,
        metavar="RANGE",
        help="Optional page range for PDFs, e.g. 0 or 0-2.",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5:7b",
        help="Installed Ollama text model used for summarization.",
    )
    parser.add_argument(
        "--language-or-script",
        default="unknown",
        help="Optional hint passed into the summarizer prompt.",
    )
    parser.add_argument(
        "--ocr-output",
        type=Path,
        default=None,
        help="Optional text file path to save raw Surya OCR output.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional JSON file path to save the summary payload.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    from playground.engines import surya as surya_engine
    from playground.summarise_text_ollama import summarize_ocr_text
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input path not found: {input_path}")

    ocr_text = surya_engine.run(input_path, page_range=args.page_range).strip()
    if not ocr_text:
        raise SystemExit("Surya returned empty OCR text.")

    if args.ocr_output:
        ocr_output = args.ocr_output.expanduser().resolve()
        ocr_output.parent.mkdir(parents=True, exist_ok=True)
        ocr_output.write_text(ocr_text + "\n", encoding="utf-8")

    summary = summarize_ocr_text(
        model=args.model,
        ocr_text=ocr_text,
        language_or_script=args.language_or_script,
    )

    payload = {
        "input_path": str(input_path),
        "ocr_engine": "surya",
        "language_or_script": args.language_or_script,
        "ocr_text": ocr_text,
        "summary": summary,
    }

    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.summary_output:
        summary_output = args.summary_output.expanduser().resolve()
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
