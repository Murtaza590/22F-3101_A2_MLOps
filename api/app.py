import io
import json
import os

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

API_KEY = os.environ.get("API_KEY", "")
INFO = json.load(open("model_info.json"))
THRESHOLD = INFO["threshold"]

INTERP = Interpreter(model_path="model.tflite")      # loaded once at start-up
INTERP.allocate_tensors()
IN = INTERP.get_input_details()[0]
OUT = INTERP.get_output_details()[0]
SIZE = int(IN["shape"][1])
ALLOWED = {"JPEG", "PNG", "WEBP"}


@app.get("/")
def index():
    return jsonify(service="VisionGuard model API",
                   routes=["GET /health", "POST /predict (X-API-Key header, form field 'file')"])


@app.get("/health")
def health():
    return jsonify(status="ok", model=INFO["model_version"], threshold=THRESHOLD)


@app.post("/predict")
def predict():
    if not API_KEY or request.headers.get("X-API-Key") != API_KEY:
        return jsonify(error="invalid or missing API key"), 401
    f = request.files.get("file")
    if f is None:
        return jsonify(error="send the image in a form field named 'file'"), 400
    try:
        img = Image.open(io.BytesIO(f.read()))
        fmt = img.format
        img = img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify(error="file is not a decodable image"), 400
    if fmt not in ALLOWED:
        return jsonify(error=f"unsupported format {fmt}"), 400
    x = cv2.resize(np.asarray(img, dtype="float32"), (SIZE, SIZE),
                   interpolation=cv2.INTER_LINEAR)[None].astype("float32")
    INTERP.set_tensor(IN["index"], x)
    INTERP.invoke()
    p = float(INTERP.get_tensor(OUT["index"])[0][0])
    is_ai = p >= THRESHOLD
    return jsonify(label="AI-generated" if is_ai else "Real",
                   confidence=round(p if is_ai else 1 - p, 4),
                   p_ai=round(p, 4), threshold=THRESHOLD,
                   model_version=INFO["model_version"])


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="file larger than 5 MB"), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), debug=False)