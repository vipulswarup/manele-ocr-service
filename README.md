# manele-ocr-service

Local playground for trying document OCR stacks without a web UI or HTTP API. The default Python environment is set up for [SuryaOCR](https://github.com/datalab-to/surya). [olmOCR](https://github.com/allenai/olmocr) is supported as an optional separate install.

## Setup

Python 3.10 or newer is required for Surya.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install olmOCR in **another** virtual environment when you need it; its dependency stack can conflict with Surya.

## Usage

```bash
python -m playground surya path/to/file.pdf
python -m playground surya path/to/page.png --page-range 0-2
python -m playground surya path/to/folder/of/images
```

Text is written to standard output. Model weights download on first Surya run (large download; let the process finish).

### Smoke test on `data/`

With sample PDFs under `data/`, OCR only the first page of each file in one run (one model load):

```bash
python -m playground surya data --page-range 0
```

Do not pipe that command to `head` or similar while weights are still downloading; closing the pipe can send SIGPIPE and abort the download.

### Troubleshooting

If you see `ModuleNotFoundError: No module named 'requests'`, reinstall so the explicit `requests` pin from `requirements.txt` is applied (`surya-ocr` imports it but does not list it on PyPI).

If you see `AttributeError: 'SuryaDecoderConfig' object has no attribute 'pad_token_id'`, you have `transformers` 5.x; this project pins `transformers<5` in `requirements.txt` for compatibility with `surya-ocr` 0.17.x. Run `pip install -r requirements.txt` again.

If `olmocr` is on your `PATH` (from another env or a global install):

```bash
python -m playground olmocr path/to/file.pdf
```

## Layout

| Path | Role |
|------|------|
| `requirements.txt` | SuryaOCR (`surya-ocr`) |
| `playground/cli.py` | Argument parsing and dispatch |
| `playground/engines/surya.py` | Surya detection plus recognition on images or PDF pages |
| `playground/engines/olmocr.py` | Wraps the `olmocr` CLI with a temp workspace |

## License

Not specified.
