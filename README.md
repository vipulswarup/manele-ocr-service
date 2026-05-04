# manele-ocr-service

Local playground for OCR experiments on scanned documents. The repo started as a generic OCR sandbox; it now includes a more practical path for handwritten multilingual complaint letters:

1. segment pages into line crops
2. route each line through script-appropriate recognizers
3. summarize from OCR text only
4. export corrected lines for fine-tuning

## Recommended Approach

For handwritten Aadhaar complaint letters, the bottleneck is line recognition, not layout. The working direction in this repo is:

`PDF/Image -> docTR line crops -> routed line OCR -> confidence filtering -> text-only summary`

Why this direction:

- Vision LLMs over raw page images tend to hallucinate complaint content.
- Generic printed-text OCR performs inconsistently on handwriting.
- The repo already has usable line crops, which is the right unit for HTR fine-tuning.

## Script Routing

`playground/handwritten_page.py` runs line crops through multiple recognizers and picks the best candidate using:

- native model confidence
- Unicode script affinity
- a small length bonus to avoid empty/near-empty wins

Default routes currently wired:

- `devanagari` -> `devanagari_PP-OCRv3_mobile_rec`
- `urdu_arabic` -> `arabic_PP-OCRv3_mobile_rec`
- `kannada` -> `ka_PP-OCRv3_mobile_rec`
- `gurmukhi` -> custom local Paddle model dir only

This gives you a reproducible baseline immediately, while leaving room for custom fine-tuned models where off-the-shelf coverage is weak.

## Setup

### Existing Surya setup

Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Optional PaddleOCR environment

For the routed handwritten baseline, install PaddleOCR in a separate environment so it does not disturb the current Surya setup:

```bash
python3 -m venv .venv-paddle
source .venv-paddle/bin/activate
pip install paddleocr
```

Install the matching `paddlepaddle` or `paddlepaddle-gpu` build for your Linux GPU server before running large jobs.

### Optional Ollama text summarization

Use a text model already available in your local Ollama instance:

```bash
ollama serve
ollama pull qwen2.5:7b
```

## Usage

### 1. Existing engines

```bash
python -m playground surya path/to/file.pdf
python -m playground olmocr path/to/file.pdf
```

These are still useful as baselines, but they are not the recommended final path for handwritten Indic complaint letters.

### 2. Routed handwritten OCR from line crops

Run OCR over an existing folder of line crops:

```bash
python playground/handwritten_page.py line_crops_grouped --output-json page_ocr.json
```

If you already know the page script:

```bash
python playground/handwritten_page.py line_crops_grouped --script-hint devanagari
python playground/handwritten_page.py line_crops_grouped --script-hint urdu_arabic
python playground/handwritten_page.py line_crops_grouped --script-hint kannada
```

For Gurmukhi, point to a local fine-tuned Paddle recognition model:

```bash
python playground/handwritten_page.py line_crops_grouped \
  --script-hint gurmukhi \
  --gurmukhi-model-dir /path/to/gurmukhi_rec_model \
  --output-json page_ocr.json
```

Output JSON includes:

- detected script
- cleaned page text
- uncertain line count
- per-line candidate scores for debugging

### 3. Grounded English summary from OCR text

Summarize from OCR text, not from the original image:

```bash
python playground/summarise_text_ollama.py page_ocr.json --output-json page_summary.json
```

This avoids the image-level hallucination problem seen in `playground/summarise_letters_ollama.py`.

## Preparing Fine-Tuning Data

If the routed baseline is still weak on your handwriting, start building a real HTR dataset from the line crops you already have.

### 1. Bootstrap an annotation sheet

```bash
python playground/bootstrap_annotations.py line_crops_grouped --output annotations.jsonl
```

This creates JSONL rows like:

```json
{"image":"line_00.png","text":"","script":"","notes":""}
```

Fill `text` and `script` manually during transcription.

### 2. Export per-script training sets

```bash
python playground/export_htr_dataset.py annotations.jsonl \
  --image-root line_crops_grouped \
  --output-dir training_data
```

This exports, per script:

- `training_data/<script>/paddle/train.txt`
- `training_data/<script>/paddle/val.txt`
- `training_data/<script>/kraken/train/*.gt.txt`
- `training_data/<script>/kraken/val/*.gt.txt`
- `training_data/<script>/dict.txt`

The Paddle manifests are ready for recognition fine-tuning with a custom dictionary. The Kraken export uses its legacy line-image plus `.gt.txt` format.

## Practical Training Strategy

Use the routed baseline for triage, then fine-tune where needed:

- Devanagari: start with PaddleOCR `devanagari_PP-OCRv3_mobile_rec`, then fine-tune on your complaint-letter lines.
- Urdu: treat `arabic_PP-OCRv3_mobile_rec` as a weak baseline only; for real Nastaliq handwriting you should expect to fine-tune.
- Kannada: start with `ka_PP-OCRv3_mobile_rec`, then fine-tune on your own lines.
- Gurmukhi: plan on custom training from the start.

Two training backends make sense here:

- PaddleOCR when you want a production-friendly recognizer with straightforward inference and export.
- Kraken when you want line-level HTR tooling that is explicitly designed around transcribed line images.

## Files

| Path | Role |
|------|------|
| `playground/handwritten_page.py` | Routed line-level OCR over existing line crops |
| `playground/summarise_text_ollama.py` | Text-only grounded summarization |
| `playground/bootstrap_annotations.py` | Empty JSONL transcription sheet generator |
| `playground/export_htr_dataset.py` | Per-script export for PaddleOCR and Kraken |
| `playground/engines/paddle.py` | Small PaddleOCR line-recognition wrapper |
| `playground/engines/surya.py` | Existing Surya runner |
| `playground/engines/olmocr.py` | Existing olmOCR wrapper |

## Notes

- `playground/summarise_letters_ollama.py` remains in the repo as an image-based experiment, but it should not be your production summarization path.
- The new handwritten route assumes segmentation is already solved and works on folders of line crops.
- `requirements.txt` still describes the Surya environment only; PaddleOCR is intentionally optional and best kept separate.

## License

Not specified.
