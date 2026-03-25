"""
=============================================================
  GlaucoScan - Eye Analysis System
  - YOLO Classification  : Glaucoma Positive / Negative
  - MobileNetV2 Regression: Eyes Quality Score (1-10)
=============================================================
"""

import os, io, base64
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify, render_template_string
from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
YOLO_WEIGHT_PATH       = r"C:\Users\Arif\Desktop\Project_Eyes\hillel-yaffe-glaucoma\hillel-yaffe-glaucoma\runs\classify\train\weights\best.pt"
REGRESSION_WEIGHT_PATH = r"C:\Users\Arif\Desktop\Project_Eyes\outputs\best_model.pth"
CSV_PATH               = r"C:\Users\Arif\Desktop\Project_Eyes\dataset\Labels.csv"

# Medical scale: Eyes quality sentiasa 1-10
# Model output [0,1] × (MAX-MIN) + MIN → paparan /10
SCORE_MIN = 1.0
SCORE_MAX = 10.0

# Auto-detect SCORE_MIN dari CSV sahaja (MAX tetap 10 — medical standard)
try:
    import pandas as pd
    _df        = pd.read_csv(CSV_PATH)
    _score_col = _df.columns[3]
    SCORE_MIN  = float(_df[_score_col].min())
    print(f"[INFO] SCORE_MIN dari CSV : {SCORE_MIN:.3f}")
    print(f"[INFO] SCORE_MAX (medical): {SCORE_MAX:.3f}")
except Exception as e:
    print(f"[WARN] CSV tidak jumpa, guna default SCORE_MIN=1, SCORE_MAX=10 ({e})")

IMG_SIZE = 224
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# MODEL 1: YOLO CLASSIFICATION
# ─────────────────────────────────────────────
print(f"[INFO] Device        : {DEVICE}")
print(f"[INFO] Loading YOLO  : {YOLO_WEIGHT_PATH}")
yolo_model = YOLO(YOLO_WEIGHT_PATH)
print(f"[INFO] YOLO classes  : {yolo_model.names}")
print("[INFO] YOLO loaded ✅")

# ─────────────────────────────────────────────
# MODEL 2: MOBILENETV2 REGRESSION
# ─────────────────────────────────────────────
class MobileNetV2Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone      = models.mobilenet_v2(weights=None)
        self.features = backbone.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.head     = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.head(x)
        return x.squeeze(1)


print(f"[INFO] Loading Regression: {REGRESSION_WEIGHT_PATH}")
reg_model = MobileNetV2Regressor().to(DEVICE)
reg_model.load_state_dict(torch.load(REGRESSION_WEIGHT_PATH, map_location=DEVICE))
reg_model.eval()
print("[INFO] Regression loaded ✅")

# ─────────────────────────────────────────────
# TRANSFORM
# ─────────────────────────────────────────────
reg_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def predict_glaucoma(img: Image.Image) -> dict:
    results    = yolo_model.predict(img, verbose=False)
    result     = results[0]
    probs      = result.probs
    names      = result.names
    top_idx    = int(probs.top1)
    confidence = float(probs.top1conf)
    label      = names[top_idx]

    label_lower = label.lower()
    # Positive indicators: '+', 'pos', 'gon+', 'glaucoma'
    # Negative indicators: '-', 'neg', 'normal', 'no', 'gon-'
    # Logic: if label contains '+' → Positive, if contains '-' → Negative
    # Fallback: check known keywords
    if "+" in label_lower or any(x in label_lower for x in ["pos", "gon+"]):
        status, emoji = "Positive", "⚠️"
    else:
        status, emoji = "Negative", "✅"

    top_classes = {names[i]: round(float(probs.data[i]) * 100, 2) for i in names}

    return {
        "status"     : status,
        "emoji"      : emoji,
        "confidence" : round(confidence * 100, 2),
        "label"      : label,
        "top_classes": top_classes
    }


def predict_quality(img: Image.Image) -> dict:
    tensor = reg_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        norm_score = reg_model(tensor).item()   # [0, 1]

    # Scale back to original training range, then display over /10
    raw_score = norm_score * (SCORE_MAX - SCORE_MIN) + SCORE_MIN
    raw_score = round(max(SCORE_MIN, min(SCORE_MAX, raw_score)), 2)

    # Meter percentage based on /10 scale
    pct = (raw_score - 1.0) / (10.0 - 1.0) * 100   # always 1-10 for display

    if pct >= 75:
        quality_label, quality_color = "Excellent", "#00c896"
    elif pct >= 50:
        quality_label, quality_color = "Good",      "#7ec8e3"
    elif pct >= 25:
        quality_label, quality_color = "Fair",      "#f0a500"
    else:
        quality_label, quality_color = "Poor",      "#e74c3c"

    return {
        "raw_score"    : raw_score,
        "pct"          : round(pct, 1),
        "quality_label": quality_label,
        "quality_color": quality_color,
    }


# ─────────────────────────────────────────────
# FLASK
# ─────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def index():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "index.html")
    with open(html_path, encoding="utf-8") as f:
        return render_template_string(f.read())


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    try:
        img_bytes = file.read()
        img       = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        glaucoma_result = predict_glaucoma(img)
        quality_result  = predict_quality(img)

        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        ext     = file.filename.rsplit(".", 1)[-1].lower()
        mime    = "image/jpeg" if ext in ["jpg", "jpeg"] else f"image/{ext}"

        return jsonify({
            "success"  : True,
            "glaucoma" : glaucoma_result,
            "quality"  : quality_result,
            "image_b64": f"data:{mime};base64,{img_b64}",
            "filename" : file.filename
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  GlaucoScan → http://127.0.0.1:5000")
    print("="*55 + "\n")
    app.run(debug=True, host="127.0.0.1", port=5000)
