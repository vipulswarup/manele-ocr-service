from pathlib import Path
from PIL import Image
import torch
from transformers import (
    ViTImageProcessor,
    AutoTokenizer,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

CROPS_DIR = Path("line_crops_grouped")
MODEL_NAME = "sabaridsnfuji/Hindi_Offline_Handwritten_OCR"
ENCODER_NAME = "google/vit-base-patch16-224-in21k"
DECODER_NAME = "surajp/RoBERTa-hindi-guj-san"

print("Loading feature extractor and tokenizer...")
feature_extractor = ViTImageProcessor.from_pretrained(ENCODER_NAME)
tokenizer = AutoTokenizer.from_pretrained(DECODER_NAME)
processor = TrOCRProcessor(image_processor=feature_extractor, tokenizer=tokenizer)

print("Loading model...")
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print(f"Using device: {device}")

results = []

for img_path in sorted(CROPS_DIR.glob("*.png")):
    image = Image.open(img_path).convert("RGB")

    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=64,
        )

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    print(f"{img_path.name}: {text}")
    results.append((img_path.name, text))

print("\n--- FINAL RECONSTRUCTED TEXT ---\n")
for name, text in results:
    print(f"{name}: {text}")