from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playground.engines.paddle import recognize_line


SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    "devanagari": ((0x0900, 0x097F),),
    "urdu_arabic": (
        (0x0600, 0x06FF),
        (0x0750, 0x077F),
        (0x08A0, 0x08FF),
        (0xFB50, 0xFDFF),
        (0xFE70, 0xFEFF),
    ),
    "kannada": ((0x0C80, 0x0CFF),),
    "gurmukhi": ((0x0A00, 0x0A7F),),
}

PADDLE_ROUTE_DEFAULTS: dict[str, dict[str, str]] = {
    "devanagari": {
        "model_name": "devanagari_PP-OCRv3_mobile_rec",
        "language": "hi",
    },
    "urdu_arabic": {
        "model_name": "arabic_PP-OCRv3_mobile_rec",
        "language": "ur",
    },
    "kannada": {
        "model_name": "ka_PP-OCRv3_mobile_rec",
        "language": "kn",
    },
}

DISPLAY_NAMES = {
    "devanagari": "Devanagari",
    "urdu_arabic": "Urdu (Arabic script)",
    "kannada": "Kannada",
    "gurmukhi": "Gurmukhi",
}


@dataclass(slots=True)
class CandidateResult:
    route: str
    backend: str
    text: str
    raw_score: float
    script_affinity: float
    combined_score: float
    model_name: str | None = None
    model_dir: str | None = None


@dataclass(slots=True)
class LineResult:
    file: str
    best_script: str
    best_text: str
    best_score: float
    uncertain: bool
    candidates: list[CandidateResult]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run routed handwritten OCR over a folder of line crops and emit page-level JSON. "
            "This keeps summarization grounded in OCR text instead of raw-image VLM guesses."
        )
    )
    parser.add_argument("line_dir", type=Path, help="Directory containing ordered line crop images.")
    parser.add_argument(
        "--script-hint",
        choices=sorted(DISPLAY_NAMES),
        default=None,
        help="Restrict routing to one expected script when the page is known to be monolingual.",
    )
    parser.add_argument(
        "--gurmukhi-model-dir",
        type=Path,
        default=None,
        help=(
            "Optional local PaddleOCR text recognition model directory for Gurmukhi. "
            "No maintained off-the-shelf Gurmukhi line recognizer is wired by default."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the full JSON result.",
    )
    parser.add_argument(
        "--uncertain-threshold",
        type=float,
        default=0.55,
        help="Mark lines below this combined score as uncertain.",
    )
    return parser


def _iter_line_images(line_dir: Path) -> list[Path]:
    allowed = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
    images = [p for p in sorted(line_dir.iterdir()) if p.suffix.lower() in allowed and p.is_file()]
    if not images:
        raise RuntimeError(f"No line crop images found in {line_dir}")
    return images


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text, flags=re.UNICODE)
    return text.strip()


def _char_in_script(char: str, script: str) -> bool:
    code = ord(char)
    for start, end in SCRIPT_RANGES[script]:
        if start <= code <= end:
            return True
    return False


def _script_affinity(text: str, script: str) -> float:
    meaningful = [ch for ch in text if ch.isalpha() or _char_in_script(ch, script)]
    if not meaningful:
        return 0.0
    hits = sum(1 for ch in meaningful if _char_in_script(ch, script))
    return hits / len(meaningful)


def _combined_score(text: str, raw_score: float, affinity: float) -> float:
    if not text:
        return 0.0
    length_bonus = min(len(text.replace(" ", "")), 20) / 20.0
    combined = (0.55 * raw_score) + (0.35 * affinity) + (0.10 * length_bonus)
    if affinity < 0.25:
        combined *= 0.45
    return max(0.0, min(combined, 1.0))


def _build_routes(script_hint: str | None, gurmukhi_model_dir: Path | None) -> list[dict[str, str | None]]:
    routes: list[dict[str, str | None]] = []
    for script, route in PADDLE_ROUTE_DEFAULTS.items():
        if script_hint and script != script_hint:
            continue
        routes.append(
            {
                "script": script,
                "backend": "paddle",
                "model_name": route["model_name"],
                "model_dir": None,
            }
        )

    if gurmukhi_model_dir and (not script_hint or script_hint == "gurmukhi"):
        routes.append(
            {
                "script": "gurmukhi",
                "backend": "paddle",
                "model_name": None,
                "model_dir": str(gurmukhi_model_dir.expanduser().resolve()),
            }
        )

    if not routes:
        raise RuntimeError("No OCR routes are enabled. Provide a valid script hint or a custom model directory.")
    return routes


def _run_candidate(image_path: Path, route: dict[str, str | None]) -> CandidateResult:
    script = str(route["script"])
    backend = str(route["backend"])
    if backend != "paddle":
        raise RuntimeError(f"Unsupported backend: {backend}")

    result = recognize_line(
        image_path,
        model_name=route["model_name"],
        model_dir=route["model_dir"],
    )
    text = _normalize_text(result.text)
    affinity = _script_affinity(text, script)
    combined = _combined_score(text, result.score, affinity)

    return CandidateResult(
        route=script,
        backend=backend,
        text=text,
        raw_score=result.score,
        script_affinity=affinity,
        combined_score=combined,
        model_name=result.model_name,
        model_dir=result.model_dir,
    )


def run_page(
    line_dir: Path,
    *,
    script_hint: str | None = None,
    gurmukhi_model_dir: Path | None = None,
    uncertain_threshold: float = 0.55,
) -> dict[str, object]:
    line_dir = line_dir.expanduser().resolve()
    if not line_dir.is_dir():
        raise RuntimeError(f"Expected a directory of line crops, got: {line_dir}")

    routes = _build_routes(script_hint, gurmukhi_model_dir)
    line_images = _iter_line_images(line_dir)
    line_results: list[LineResult] = []

    for image_path in line_images:
        candidates = [_run_candidate(image_path, route) for route in routes]
        best = max(candidates, key=lambda item: item.combined_score)
        line_results.append(
            LineResult(
                file=image_path.name,
                best_script=best.route,
                best_text=best.text,
                best_score=best.combined_score,
                uncertain=best.combined_score < uncertain_threshold,
                candidates=candidates,
            )
        )

    script_totals: dict[str, float] = {}
    for line in line_results:
        script_totals[line.best_script] = script_totals.get(line.best_script, 0.0) + line.best_score

    detected_script = max(script_totals, key=script_totals.get) if script_totals else "unknown"
    cleaned_lines = [line.best_text for line in line_results if line.best_text]

    page_text = "\n".join(cleaned_lines)
    language = next(
        (route["language"] for script, route in PADDLE_ROUTE_DEFAULTS.items() if script == detected_script),
        None,
    )

    return {
        "input_dir": str(line_dir),
        "detected_script": detected_script,
        "detected_script_display": DISPLAY_NAMES.get(detected_script, detected_script),
        "detected_language": language,
        "cleaned_text": page_text,
        "uncertain_line_count": sum(1 for line in line_results if line.uncertain),
        "lines": [
            {
                **asdict(line),
                "candidates": [asdict(candidate) for candidate in line.candidates],
            }
            for line in line_results
        ],
    }


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    result = run_page(
        args.line_dir,
        script_hint=args.script_hint,
        gurmukhi_model_dir=args.gurmukhi_model_dir,
        uncertain_threshold=args.uncertain_threshold,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json:
        args.output_json.expanduser().resolve().write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
