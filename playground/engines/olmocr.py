from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


def run(path: Path) -> str:
    if not shutil.which("olmocr"):
        raise RuntimeError(
            "The `olmocr` command was not found. Install olmOCR in a separate environment "
            "and ensure the CLI is on PATH. See https://github.com/allenai/olmocr"
        )

    path = path.resolve()
    if path.is_dir():
        raise RuntimeError(
            "olmocr runner expects a single PDF or image path. "
            "Pass one file or invoke `olmocr` directly for batch workspaces."
        )

    with tempfile.TemporaryDirectory(prefix="playground_olmocr_") as workspace:
        cmd = [
            "olmocr",
            workspace,
            "--markdown",
            "--pdfs",
            str(path),
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"olmocr failed with exit code {e.returncode}. "
                "Ensure a GPU or remote --server setup matches https://github.com/allenai/olmocr"
            ) from e

        md_dir = Path(workspace) / "markdown"
        if not md_dir.is_dir():
            return ""

        expected = md_dir / f"{path.stem}.md"
        if expected.is_file():
            return expected.read_text(encoding="utf-8")

        md_files = sorted(md_dir.glob("*.md"))
        if not md_files:
            return ""
        if len(md_files) == 1:
            return md_files[0].read_text(encoding="utf-8")
        return "\n\n".join(
            f"## {p.name}\n{p.read_text(encoding='utf-8')}".strip() for p in md_files
        )
