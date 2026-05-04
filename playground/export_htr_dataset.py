#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export filled JSONL transcriptions into per-script datasets for PaddleOCR and Kraken."
        )
    )
    parser.add_argument("annotations", type=Path, help="Filled JSONL file from bootstrap_annotations.py")
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("."),
        help="Directory used to resolve relative image paths from the JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training_data"),
        help="Root directory for exported PaddleOCR/Kraken datasets.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio per script.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for train/val split.")
    return parser


def _load_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            row = json.loads(raw_line)
            text = str(row.get("text", "")).strip()
            script = str(row.get("script", "")).strip()
            image = str(row.get("image", "")).strip()
            if not image:
                raise SystemExit(f"Missing image at line {line_no}")
            if not text or not script:
                continue
            rows.append({"image": image, "text": text, "script": script})
    if not rows:
        raise SystemExit("No annotated rows with both `text` and `script` were found.")
    return rows


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_paddle_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{row['image']}\t{row['text']}\n")


def _write_dict(path: Path, rows: list[dict[str, str]]) -> None:
    chars = sorted({char for row in rows for char in row["text"] if char.strip()})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(chars) + "\n", encoding="utf-8")


def _export_script(
    script: str,
    rows: list[dict[str, str]],
    *,
    image_root: Path,
    output_dir: Path,
    val_ratio: float,
    rng: random.Random,
) -> dict[str, int]:
    shuffled = rows[:]
    rng.shuffle(shuffled)

    val_count = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
    val_rows = shuffled[:val_count]
    train_rows = shuffled[val_count:]
    if not train_rows:
        train_rows, val_rows = shuffled, []

    script_root = output_dir / script
    paddle_root = script_root / "paddle"
    kraken_root = script_root / "kraken"

    exported_train: list[dict[str, str]] = []
    exported_val: list[dict[str, str]] = []

    for split_name, split_rows, exported in (
        ("train", train_rows, exported_train),
        ("val", val_rows, exported_val),
    ):
        for index, row in enumerate(split_rows):
            source = Path(row["image"])
            if not source.is_absolute():
                source = (image_root / source).resolve()
            if not source.is_file():
                raise SystemExit(f"Image not found: {source}")

            file_name = f"{script}_{split_name}_{index:05d}{source.suffix.lower()}"
            paddle_rel = Path("images") / split_name / file_name
            paddle_abs = paddle_root / paddle_rel
            kraken_abs = kraken_root / split_name / file_name

            _copy(source, paddle_abs)
            _copy(source, kraken_abs)
            kraken_abs.with_suffix(kraken_abs.suffix + ".gt.txt").write_text(
                row["text"],
                encoding="utf-8",
            )

            exported.append({"image": str(paddle_rel), "text": row["text"]})

    _write_paddle_manifest(paddle_root / "train.txt", exported_train)
    _write_paddle_manifest(paddle_root / "val.txt", exported_val)
    _write_dict(script_root / "dict.txt", rows)

    return {
        "train": len(exported_train),
        "val": len(exported_val),
        "total": len(rows),
    }


def main() -> None:
    args = _build_parser().parse_args()
    annotations_path = args.annotations.expanduser().resolve()
    image_root = args.image_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(annotations_path)
    by_script: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_script[row["script"]].append(row)

    rng = random.Random(args.seed)
    summary: dict[str, dict[str, int]] = {}
    for script, script_rows in sorted(by_script.items()):
        summary[script] = _export_script(
            script,
            script_rows,
            image_root=image_root,
            output_dir=output_dir,
            val_ratio=args.val_ratio,
            rng=rng,
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
