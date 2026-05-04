from pathlib import Path
from PIL import Image
import torch

from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor

CROPS_DIR = Path("line_crops_grouped")

foundation = FoundationPredictor()
predictor = RecognitionPredictor(foundation)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

images = []
paths = []
bboxes = []

for img_path in sorted(CROPS_DIR.glob("*.png")):
    image = Image.open(img_path).convert("RGB")
    w, h = image.size

    images.append(image)
    paths.append(img_path.name)

    # 👇 full-image bounding box
    bboxes.append([[0, 0, w, h]])

# Run recognition with provided boxes
results = predictor(images, bboxes=bboxes)

print("\n--- LINE RESULTS ---\n")

all_text = []

for i, res in enumerate(results):
    text = res.text.strip() if hasattr(res, "text") else ""
    print(f"{paths[i]}: {text}")
    all_text.append(text)

print("\n--- FINAL RECONSTRUCTED TEXT ---\n")
print("\n".join(all_text))