from __future__ import annotations

from pathlib import Path

from surya.detection import DetectionPredictor
from surya.foundation import FoundationPredictor
from surya.input.load import load_from_file, load_from_folder
from surya.recognition import RecognitionPredictor
from surya.scripts.config import CLILoader
from surya.settings import settings


def run(path: Path, page_range: str | None = None) -> str:
    path = path.resolve()
    parsed_range = CLILoader.parse_range_str(page_range) if page_range else None

    if path.is_dir():
        images, _ = load_from_folder(str(path), parsed_range)
        highres_images, _ = load_from_folder(
            str(path), parsed_range, dpi=settings.IMAGE_DPI_HIGHRES
        )
    else:
        images, _ = load_from_file(str(path), parsed_range)
        highres_images, _ = load_from_file(
            str(path), parsed_range, dpi=settings.IMAGE_DPI_HIGHRES
        )

    foundation = FoundationPredictor()
    recognition = RecognitionPredictor(foundation)
    detection = DetectionPredictor()
    predictions = recognition(
        images,
        det_predictor=detection,
        highres_images=highres_images,
    )

    pages = ["\n".join(line.text for line in p.text_lines) for p in predictions]
    return "\n\n".join(pages).rstrip() + ("\n" if pages else "")
