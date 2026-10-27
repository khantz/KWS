from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

CNN1D_PATH = Path(r"C:\Users\M S I\PycharmProjects\KWS\results\tflite_evaluation\plots\cnn1d_header.png")
DSCNN_PATH = Path(r"C:\Users\M S I\PycharmProjects\KWS\results\tflite_evaluation\plots\dscnn_header.png")
OUT_PATH   = Path(r"C:\Users\M S I\PycharmProjects\KWS\results\tflite_evaluation\plots\Figure_4_X_Model_Headers.png")

# check inputs exist
for p in (CNN1D_PATH, DSCNN_PATH):
    if not p.is_file():
        raise FileNotFoundError(f"Missing input: {p}")

img1 = Image.open(CNN1D_PATH).convert("RGB")
img2 = Image.open(DSCNN_PATH).convert("RGB")

# match heights
h = max(img1.height, img2.height)
img1 = img1.resize((int(img1.width * h / img1.height), h))
img2 = img2.resize((int(img2.width * h / img2.height), h))

gap = 30
canvas = Image.new("RGB",
                   (img1.width + gap + img2.width, h),
                   color="white")

canvas.paste(img1, (0, 0))
canvas.paste(img2, (img1.width + gap, 0))

draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("arial.ttf", 28)
except Exception:
    font = ImageFont.load_default()

draw.text((10, 10), "(a) CNN1D", fill="black", font=font)
draw.text((img1.width + gap + 10, 10), "(b) DS-CNN", fill="black", font=font)

# ---- Save with explicit format so extension cannot be missing ----
canvas.save(OUT_PATH, format="PNG", dpi=(600, 600))
print(f"Saved: {OUT_PATH}")