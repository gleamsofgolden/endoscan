"""
EndoScan AI — Inference Server (MRI Only)
==========================================
Strictly accepts only greyscale medical scans (MRI/ultrasound).
Rejects colour photos, screenshots, and non-medical images.

Run:
    python app\server.py --model models\best_model.pt
"""

import os
import io
import base64
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory

import torch
import torch.nn.functional as F
from torchvision import models, transforms


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default="models/best_model.pt")
    p.add_argument("--port",    type=int, default=5000)
    p.add_argument("--host",    default="0.0.0.0")
    p.add_argument("--app_dir", default="app")
    return p.parse_args()


def validate_mri(pil_img):
    """
    Returns (True, None) if image passes all MRI checks.
    Returns (False, error_message) if rejected.
    """
    img = pil_img.convert("RGB")
    w, h = img.size
    arr = np.array(img).astype(np.float32)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

    # 1. Minimum size
    if w < 64 or h < 64:
        return False, "Image is too small. Please upload a proper MRI scan."

    # 2. Aspect ratio — MRI slices are roughly square
    ratio = w / h
    if ratio < 0.4 or ratio > 2.5:
        return False, "Image shape doesn't match an MRI scan. Please upload a pelvic MRI image."

    # 3. Colour saturation — MRI images are greyscale or near-greyscale
    per_pixel_colour_variance = np.mean(np.std(arr, axis=2))
    if per_pixel_colour_variance > 25:
        return False, "This looks like a colour photograph, not an MRI scan. Only greyscale pelvic MRI images are accepted."

    # 4. Skin tone check
    skin_mask = (
        (r > 150) & (r < 255) &
        (g > 80)  & (g < 200) &
        (b > 50)  & (b < 160) &
        (r > g)   & (g > b)   &
        ((r - b) > 30)
    )
    if np.sum(skin_mask) / (w * h) > 0.20:
        return False, "This appears to be a photo of a person. Only pelvic MRI scans are accepted."

    # 5. Large white regions — screenshots/documents
    white_mask = (r > 245) & (g > 245) & (b > 245)
    if np.sum(white_mask) / (w * h) > 0.60:
        return False, "This looks like a document or screenshot. Only pelvic MRI scans are accepted."

    # 6. Colourful pixels — charts, logos, illustrations
    saturation = np.max(arr, axis=2) - np.min(arr, axis=2)
    if np.sum(saturation > 60) / (w * h) > 0.15:
        return False, "This image contains too many colours to be an MRI scan. Only greyscale pelvic MRI images are accepted."

    return True, None


def load_model(model_path, device):
    import torch.nn as nn
    checkpoint = torch.load(model_path, map_location=device)
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(128, 2),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    class_to_idx = checkpoint.get("class_to_idx", {"negative": 0, "positive": 1})
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    print(f"  Model loaded — Val AUC: {checkpoint.get('val_auc', 'N/A'):.4f}")
    return model, idx_to_class


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        last_conv = None
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                last_conv = m
        last_conv.register_forward_hook(
            lambda m, i, o: setattr(self, 'activations', o.detach()))
        last_conv.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))

    def generate(self, tensor, class_idx=None):
        tensor = tensor.requires_grad_(True)
        logits = self.model(tensor)
        if class_idx is None:
            class_idx = logits.argmax(1).item()
        self.model.zero_grad()
        logits[0, class_idx].backward()
        weights = self.gradients.mean([2, 3], keepdim=True)
        cam = F.relu((weights * self.activations).sum(1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam, class_idx


def make_heatmap(pil_img, cam, alpha=0.45):
    import cv2
    img_np = np.array(pil_img.convert("RGB"))
    h, w = img_np.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    colored = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = (alpha * colored + (1 - alpha) * img_np).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_image(file_bytes, filename):
    suffix = Path(filename).suffix.lower()
    if suffix == ".dcm":
        try:
            import pydicom, tempfile
            with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            ds = pydicom.dcmread(tmp_path)
            px = ds.pixel_array.astype(np.float32)
            px = (px - px.min()) / (px.max() - px.min() + 1e-8) * 255
            pil_img = Image.fromarray(px.astype(np.uint8), mode="L").convert("RGB")
            os.unlink(tmp_path)
            return pil_img
        except Exception as e:
            print(f"DICOM failed: {e}")
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


app = Flask(__name__)
model_obj = None
idx_to_class_map = {0: "negative", 1: "positive"}
device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Safe defaults so gunicorn/Railway doesn't crash on import
class _Args:
    model   = os.environ.get("MODEL_PATH", "models/best_model.pt")
    app_dir = os.environ.get("APP_DIR",    "app")
    port    = int(os.environ.get("PORT",   5000))
    host    = "0.0.0.0"

args_obj = _Args()

import threading

def _load_model_background():
    global model_obj, idx_to_class_map
    model_path = args_obj.model
    if Path(model_path).exists():
        try:
            model_obj, idx_to_class_map = load_model(model_path, device_obj)
            print("  Model loaded successfully")
        except Exception as e:
            print(f"  Model load failed: {e}")
    else:
        print(f"  WARNING: No model found at {model_path}")

threading.Thread(target=_load_model_background, daemon=True).start()

@app.route("/")
def index():
    return send_from_directory(args_obj.app_dir, "index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    file_bytes = file.read()

    try:
        pil_img = load_image(file_bytes, file.filename)
    except Exception as e:
        return jsonify({"error": f"Could not read file: {e}"}), 400

    # Strict MRI validation — reject anything that isn't a greyscale medical scan
    valid, reason = validate_mri(pil_img)
    if not valid:
        return jsonify({"error": reason}), 422

    tensor = IMG_TRANSFORM(pil_img).unsqueeze(0).to(device_obj)

    with torch.enable_grad():
        cam_gen = GradCAM(model_obj)
        cam, predicted_idx = cam_gen.generate(tensor)

    with torch.no_grad():
        logits = model_obj(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    positive_idx = next(k for k, v in idx_to_class_map.items() if v == "positive")
    positive_prob = float(probs[positive_idx])
    prediction = idx_to_class_map[predicted_idx]
    heatmap_b64 = make_heatmap(pil_img, cam)

    return jsonify({
        "prediction":           prediction,
        "positive_probability": positive_prob,
        "confidence":           float(probs[predicted_idx]),
        "probabilities":        {idx_to_class_map[i]: float(p) for i, p in enumerate(probs)},
        "heatmap_base64":       heatmap_b64,
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    args_obj = parse_args()
    device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  EndoScan AI — MRI Only Mode")
    print(f"  Device : {device_obj}")
    if Path(args_obj.model).exists():
        model_obj, idx_to_class_map = load_model(args_obj.model, device_obj)
    else:
        print(f"  WARNING: No model found at {args_obj.model} — train first!")
        idx_to_class_map = {0: "negative", 1: "positive"}
    print(f"  Running at http://localhost:{args_obj.port}\n")
    app.run(host=args_obj.host, port=args_obj.port, debug=False)