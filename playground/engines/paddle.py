from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PaddleRecognitionResult:
    text: str
    score: float
    model_name: str | None = None
    model_dir: str | None = None


def recognize_line(
    image_path: Path,
    *,
    model_name: str | None = None,
    model_dir: str | None = None,
    batch_size: int = 1,
) -> PaddleRecognitionResult:
    try:
        from paddleocr import TextRecognition
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PaddleOCR is not installed. Install it in a separate environment with "
            "`pip install paddleocr` plus the matching PaddlePaddle build for your GPU/CPU."
        ) from exc

    if not model_name and not model_dir:
        raise RuntimeError("Either `model_name` or `model_dir` must be provided.")

    kwargs: dict[str, str] = {}
    if model_name:
        kwargs["model_name"] = model_name
    if model_dir:
        kwargs["model_dir"] = str(Path(model_dir).expanduser().resolve())

    model = TextRecognition(**kwargs)
    outputs = list(model.predict(input=str(image_path.resolve()), batch_size=batch_size))
    if not outputs:
        return PaddleRecognitionResult(
            text="",
            score=0.0,
            model_name=model_name,
            model_dir=model_dir,
        )

    result = outputs[0]
    payload = {}
    if hasattr(result, "get"):
        payload = dict(result)
    if not payload and hasattr(result, "json"):
        payload = dict(getattr(result, "json", {}).get("res", {}) or {})
    text = str(payload.get("rec_text", "")).strip()
    score = float(payload.get("rec_score", 0.0) or 0.0)

    return PaddleRecognitionResult(
        text=text,
        score=score,
        model_name=model_name,
        model_dir=model_dir,
    )
