#!/usr/bin/env python3
import argparse
import base64
import json
import subprocess
from pathlib import Path

import fitz  # PyMuPDF
import requests


PROMPT = """You are reading a scanned citizen complaint letter.

Tasks:
1. Identify the language or script.
2. Summarize the letter in simple English in 3-4 sentences.
3. State the main complaint in one sentence.
4. If any part is unclear, say "uncertain" rather than guessing.
5. Do not invent names, IDs, or facts not visible in the image.

Return strict JSON in this format:
{
  "language": "...",
  "summary_english": "...",
  "main_complaint": "...",
  "uncertainty_notes": "..."
}
"""


def check_ollama_running() -> None:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=10)
        r.raise_for_status()
    except Exception as exc:
        raise SystemExit(
            "Ollama is not reachable on http://localhost:11434. "
            "Start it first with: ollama serve"
        ) from exc


def check_model_exists(model: str) -> None:
    r = requests.get("http://localhost:11434/api/tags", timeout=20)
    r.raise_for_status()
    data = r.json()
    models = {m["name"] for m in data.get("models", [])}
    if model not in models:
        raise SystemExit(
            f"Model '{model}' is not installed.\n"
            f"Install it with:\n  ollama pull {model}"
        )


def pdf_to_images(pdf_path: Path, output_dir: Path, dpi: int = 200) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths: list[Path] = []

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img_path = output_dir / f"page-{page_index + 1}.png"
        pix.save(str(img_path))
        image_paths.append(img_path)

    return image_paths


def encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def ask_ollama(model: str, image_path: Path) -> dict:
    payload = {
        "model": model,
        "prompt": PROMPT,
        "images": [encode_image(image_path)],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0
        }
    }

    r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()

    raw_response = data.get("response", "").strip()
    if not raw_response:
        return {
            "language": "unknown",
            "summary_english": "",
            "main_complaint": "",
            "uncertainty_notes": "No response from model",
            "raw_response": raw_response,
        }

    try:
        parsed = json.loads(raw_response)
        parsed["raw_response"] = raw_response
        return parsed
    except json.JSONDecodeError:
        return {
            "language": "unknown",
            "summary_english": "",
            "main_complaint": "",
            "uncertainty_notes": "Model did not return valid JSON",
            "raw_response": raw_response,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise scanned complaint letters using Ollama")
    parser.add_argument("pdf", type=Path, help="Input PDF file")
    parser.add_argument("--model", default="qwen2.5vl:7b", help="Ollama model name")
    parser.add_argument("--workdir", type=Path, default=Path("letter_work"), help="Working directory")
    args = parser.parse_args()

    pdf_path = args.pdf.resolve()
    workdir = args.workdir.resolve()
    image_dir = workdir / "pages"
    output_json = workdir / "results.json"

    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    check_ollama_running()
    check_model_exists(args.model)

    images = pdf_to_images(pdf_path, image_dir)

    results = []
    for i, image_path in enumerate(images, start=1):
        print(f"Processing page {i}: {image_path.name}")
        result = ask_ollama(args.model, image_path)
        results.append({
            "page": i,
            "image": str(image_path),
            "result": result
        })

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nDone. Results written to:\n{output_json}")


if __name__ == "__main__":
    main()