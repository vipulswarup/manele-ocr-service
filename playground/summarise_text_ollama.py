#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROMPT_TEMPLATE = """You are summarizing OCR output from a citizen complaint letter submitted to UIDAI.

Rules:
1. Work only from the provided OCR text.
2. If text is noisy or incomplete, say so explicitly.
3. Do not infer a generic Aadhaar complaint unless supported by the OCR text.
4. Return strict JSON only.

Return this exact schema:
{{
  "language_or_script": "{language_or_script}",
  "summary_english": "...",
  "main_complaint": "...",
  "uncertainty_notes": "..."
}}

OCR text:
\"\"\"
{ocr_text}
\"\"\"
"""


def build_prompt(*, ocr_text: str, language_or_script: str) -> str:
    return PROMPT_TEMPLATE.format(
        language_or_script=language_or_script or "unknown",
        ocr_text=ocr_text,
    )


def check_ollama_running() -> None:
    import requests

    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        response.raise_for_status()
    except Exception as exc:
        raise SystemExit(
            "Ollama is not reachable on http://localhost:11434. Start it first with: ollama serve"
        ) from exc


def check_model_exists(model: str) -> None:
    import requests

    response = requests.get("http://localhost:11434/api/tags", timeout=20)
    response.raise_for_status()
    payload = response.json()
    models = {item["name"] for item in payload.get("models", [])}
    if model not in models:
        raise SystemExit(f"Model '{model}' is not installed. Install it with: ollama pull {model}")


def load_text_payload(path: Path) -> tuple[str, str]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise SystemExit(f"No text found in {path}")

    if path.suffix.lower() == ".json":
        payload = json.loads(raw)
        cleaned_text = str(payload.get("cleaned_text", "")).strip()
        language_or_script = str(
            payload.get("detected_script_display")
            or payload.get("detected_script")
            or payload.get("detected_language")
            or "unknown"
        )
        if not cleaned_text:
            raise SystemExit(f"`cleaned_text` was empty in {path}")
        return cleaned_text, language_or_script

    return raw, "unknown"


def ask_ollama(model: str, prompt: str) -> dict[str, str]:
    import requests

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        },
        timeout=300,
    )
    response.raise_for_status()
    payload = response.json()
    raw_response = str(payload.get("response", "")).strip()

    if not raw_response:
        return {
            "language_or_script": "unknown",
            "summary_english": "",
            "main_complaint": "",
            "uncertainty_notes": "No response from model",
            "raw_response": raw_response,
        }

    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        return {
            "language_or_script": "unknown",
            "summary_english": "",
            "main_complaint": "",
            "uncertainty_notes": "Model did not return valid JSON",
            "raw_response": raw_response,
        }

    parsed["raw_response"] = raw_response
    return parsed


def summarize_ocr_text(*, model: str, ocr_text: str, language_or_script: str) -> dict[str, str]:
    check_ollama_running()
    check_model_exists(model)
    prompt = build_prompt(
        language_or_script=language_or_script,
        ocr_text=ocr_text,
    )
    return ask_ollama(model, prompt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize OCR text using a local Ollama text model.")
    parser.add_argument("input", type=Path, help="Plain-text OCR file or JSON emitted by handwritten_page.py")
    parser.add_argument("--model", default="qwen2.5:7b", help="Installed Ollama text model")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    ocr_text, language_or_script = load_text_payload(input_path)
    summary = summarize_ocr_text(
        model=args.model,
        ocr_text=ocr_text,
        language_or_script=language_or_script,
    )
    payload = json.dumps(summary, ensure_ascii=False, indent=2)

    if args.output_json:
        args.output_json.expanduser().resolve().write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
