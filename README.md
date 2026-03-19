# manele-ocr-service

A Redis-backed worker that consumes OCR jobs from a queue and runs [OCRmyPDF](https://ocrmypdf.readthedocs.io/) via Docker to produce searchable PDFs.

## Overview

- **Worker** (`worker.py`): Long-running process that blocks on the Redis list `ocr_jobs`. For each job it runs the `ocr-base` Docker image with ocrmypdf, reading and writing PDFs under a local `data` directory.
- **Docker image** (`Dockerfile`): Ubuntu-based image with Tesseract (eng, hin, ara, pan, urd), OCRmyPDF, Ghostscript, ImageMagick, and related tools. Build and tag as `ocr-base` for the worker.

## Requirements

- Python 3 (tested with 3.14)
- Redis server (default: localhost:6379)
- Docker (to run the `ocr-base` container)
- `data` directory in the project root for input/output PDFs

## Setup

1. **Python dependencies**

   ```bash
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **Redis**

   Start Redis locally (e.g. `redis-server`) or use a remote instance. The worker connects to `localhost:6379` by default.

3. **OCR Docker image**

   Build and tag the image used by the worker:

   ```bash
   docker build -t ocr-base .
   ```

4. **Data directory**

   ```bash
   mkdir -p data
   ```

   Place input PDFs in `data/`. The worker expects job payloads to reference filenames relative to this directory (e.g. `document.pdf`).

## Running the worker

```bash
source venv/bin/activate
python worker.py
```

The worker runs until interrupted. It will log when it connects to Redis, when it receives jobs, and when OCR completes or fails.

## Job format

Push a JSON object to the Redis list `ocr_jobs`. Required field:

| Field | Description |
|-------|-------------|
| `file` | Input PDF filename (e.g. `document.pdf`). Must exist under `data/`. |

Output is written to `data/<basename>_ocr.pdf` (e.g. `document_ocr.pdf`).

Example (Redis CLI):

```bash
redis-cli LPUSH ocr_jobs '{"file": "document.pdf"}'
```

## Configuration

Currently hardcoded in `worker.py`:

- **Redis**: `host='localhost'`, `port=6379`
- **Queue**: `ocr_jobs`
- **Data directory**: `./data` (absolute path)
- **OCR options**: English language, page rotation, deskew, optimize level 2, 4 jobs

Adjust these in `worker.py` or refactor to env/config as needed.

## License

Not specified.
