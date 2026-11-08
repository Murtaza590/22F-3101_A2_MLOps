import io
import os
import sqlite3
from datetime import datetime

import requests
from dotenv import load_dotenv
from flask import Flask, g, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

load_dotenv()
API_URL = os.environ.get("API_URL", "").rstrip("/")
API_KEY = os.environ.get("API_KEY", "")
DB_PATH = os.environ.get("DB_PATH", "history.db")
ALLOWED = {"png", "jpg", "jpeg", "webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("""CREATE TABLE IF NOT EXISTS history(
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, filename TEXT,
            label TEXT, confidence REAL, model_version TEXT)""")
    return g.db


@app.teardown_appcontext
def close_db(_):
    d = g.pop("db", None)
    if d:
        d.close()


def error(msg, code):
    return render_template("error.html", msg=msg, code=code), code


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return {"status": "ok", "api_configured": bool(API_URL)}


@app.post("/predict")
def predict():
    f = request.files.get("file")
    if not f or not f.filename:
        return error("Please choose an image file.", 400)
    name = secure_filename(f.filename)
    if name.rsplit(".", 1)[-1].lower() not in ALLOWED:
        return error("Only PNG, JPG and WEBP files are allowed.", 400)
    data = f.read()
    try:
        Image.open(io.BytesIO(data)).verify()
    except (UnidentifiedImageError, OSError):
        return error("This file looks like an image but cannot be decoded.", 400)
    try:
        r = requests.post(f"{API_URL}/predict", headers={"X-API-Key": API_KEY},
                          files={"file": (name, data)}, timeout=120)
    except requests.RequestException:
        return error("The model API is unreachable (it may be waking up, retry in a minute).", 502)
    if r.status_code != 200:
        return error(f"Model API error: {r.text[:200]}", 502)
    res = r.json()
    db().execute("INSERT INTO history(ts,filename,label,confidence,model_version) VALUES(?,?,?,?,?)",
                 (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, res["label"],
                  res["confidence"], res["model_version"]))
    db().commit()
    return render_template("result.html", r=res, name=name)


@app.get("/history")
def history():
    rows = db().execute("SELECT * FROM history ORDER BY id DESC").fetchall()
    return render_template("history.html", rows=rows)


@app.errorhandler(413)
def too_large(_):
    return error("File is larger than 5 MB.", 413)


@app.errorhandler(404)
def not_found(_):
    return error("Page not found.", 404)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)