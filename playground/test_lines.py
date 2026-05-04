from pathlib import Path
from collections import defaultdict

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import Image, ImageDraw

IMAGE_PATH = "letter_work/pages/page-1.png"
OUTPUT_DIR = Path("line_crops_grouped")
DEBUG_IMAGE = "grouped_lines_debug.png"

OUTPUT_DIR.mkdir(exist_ok=True)

# Load page
doc = DocumentFile.from_images(IMAGE_PATH)
model = ocr_predictor(pretrained=True)
result = model(doc)

img = Image.open(IMAGE_PATH).convert("RGB")
w, h = img.size
page = result.pages[0]

# Collect all detected word boxes
word_boxes = []
for block in page.blocks:
    for line in block.lines:
        for word in line.words:
            ((x1, y1), (x2, y2)) = word.geometry
            word_boxes.append({
                "x1": x1 * w,
                "y1": y1 * h,
                "x2": x2 * w,
                "y2": y2 * h,
            })

# Sort top-to-bottom, then left-to-right
word_boxes.sort(key=lambda b: ((b["y1"] + b["y2"]) / 2, b["x1"]))

# Group words into lines based on vertical overlap / center proximity
grouped_lines = []
y_threshold = 18  # adjust later if needed

for box in word_boxes:
    cy = (box["y1"] + box["y2"]) / 2
    placed = False

    for line in grouped_lines:
        line_cy = line["avg_cy"]
        if abs(cy - line_cy) <= y_threshold:
            line["boxes"].append(box)
            # recompute average center y
            centers = [((b["y1"] + b["y2"]) / 2) for b in line["boxes"]]
            line["avg_cy"] = sum(centers) / len(centers)
            placed = True
            break

    if not placed:
        grouped_lines.append({
            "avg_cy": cy,
            "boxes": [box],
        })

# Sort lines top-to-bottom
grouped_lines.sort(key=lambda l: l["avg_cy"])

# Save merged line crops
draw = ImageDraw.Draw(img)

count = 0
for idx, line in enumerate(grouped_lines):
    boxes = sorted(line["boxes"], key=lambda b: b["x1"])

    x1 = min(b["x1"] for b in boxes)
    y1 = min(b["y1"] for b in boxes)
    x2 = max(b["x2"] for b in boxes)
    y2 = max(b["y2"] for b in boxes)

    pad = 10
    left = max(0, int(x1 - pad))
    top = max(0, int(y1 - pad))
    right = min(w, int(x2 + pad))
    bottom = min(h, int(y2 + pad))

    if right <= left or bottom <= top:
        continue

    crop = img.crop((left, top, right, bottom))
    out_path = OUTPUT_DIR / f"line_{idx:02d}.png"
    crop.save(out_path)

    draw.rectangle((left, top, right, bottom), outline="red", width=2)
    print(f"Saved: {out_path}")

    count += 1

img.save(DEBUG_IMAGE)
print(f"\nSaved {count} grouped line crops to: {OUTPUT_DIR.resolve()}")
print(f"Saved debug image to: {DEBUG_IMAGE}")