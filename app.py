import os
import json
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB max upload


import math
from flask.json.provider import DefaultJSONProvider

class SafeJSONProvider(DefaultJSONProvider):
    """Replace NaN/Inf with null so JavaScript can parse the response."""
    def dumps(self, obj, **kwargs):
        kwargs.setdefault("default", self.default)
        return json.dumps(self._sanitize(obj), **kwargs)

    def _sanitize(self, obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._sanitize(v) for v in obj]
        return obj

app.json_provider_class = SafeJSONProvider
app.json = SafeJSONProvider(app)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
MODELS_DIR = BASE_DIR / "trained_models"
UPLOAD_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

MODEL_ID = "HikmaAI/hikmaai-mdeberta-v3-base-prompt-injection-multilingual"
LABEL_MAP = {0: "SAFE", 1: "INJECTION"}

# ── Global state ──
model_state = {
    "active": "original",  # "original" or a trained model name
    "onnx_session": None,
    "onnx_tokenizer": None,
    "pt_model": None,
    "pt_tokenizer": None,
}

training_jobs = {}  # job_id -> {status, progress, logs, ...}


def load_original_model():
    print(f"Loading ONNX model: {MODEL_ID} ...")
    model_path = hf_hub_download(
        repo_id=MODEL_ID, filename="model_quantized.onnx", subfolder="onnx/int8"
    )
    model_state["onnx_session"] = ort.InferenceSession(model_path)
    model_state["onnx_tokenizer"] = AutoTokenizer.from_pretrained(
        MODEL_ID, subfolder="onnx/int8"
    )
    model_state["active"] = "original"
    print("Original ONNX model loaded.")


def load_trained_model(model_name):
    model_dir = MODELS_DIR / model_name
    print(f"Loading trained model from {model_dir} ...")
    model_state["pt_tokenizer"] = AutoTokenizer.from_pretrained(str(model_dir))
    model_state["pt_model"] = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir)
    )
    model_state["pt_model"].eval()
    model_state["active"] = model_name
    print(f"Trained model '{model_name}' loaded.")


def softmax(x):
    # Handle NaN/Inf from broken models
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        return np.array([0.5] * len(x))
    e = np.exp(x - np.max(x))
    return e / e.sum()


def classify_text(text):
    if model_state["active"] == "original":
        tokenizer = model_state["onnx_tokenizer"]
        session = model_state["onnx_session"]
        inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=512)
        ort_inputs = {
            k: v
            for k, v in inputs.items()
            if k in [i.name for i in session.get_inputs()]
        }
        logits = session.run(None, ort_inputs)[0][0]
        probs = softmax(logits)
    else:
        tokenizer = model_state["pt_tokenizer"]
        model = model_state["pt_model"]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits[0].numpy()
        probs = softmax(logits)
    return {LABEL_MAP[i]: float(probs[i]) for i in range(len(probs))}


# ── Dataset parsing ──

def parse_dataset_file(filepath):
    """Parse CSV, JSON, or JSONL into list of {text, label} dicts.
    Accepts columns: text/prompt/input + label/class/is_injection.
    Labels: 0/1, safe/injection, benign/malicious."""
    ext = Path(filepath).suffix.lower()
    rows = []

    if ext == ".jsonl":
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    elif ext == ".json":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                rows = data
            elif isinstance(data, dict):
                # HF datasets format: {"text": [...], "label": [...]}
                keys = list(data.keys())
                length = len(data[keys[0]])
                rows = [{k: data[k][i] for k in keys} for i in range(length)]
    elif ext == ".csv":
        import csv
        with open(filepath, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv, .json, or .jsonl")

    if not rows:
        raise ValueError("Dataset is empty")

    # Detect text column
    text_col = None
    for candidate in ["text", "prompt", "input", "sentence", "content", "query"]:
        if candidate in rows[0]:
            text_col = candidate
            break
    if not text_col:
        raise ValueError(
            f"No text column found. Expected one of: text, prompt, input, sentence, content, query. "
            f"Got columns: {list(rows[0].keys())}"
        )

    # Detect label column
    label_col = None
    for candidate in ["label", "class", "is_injection", "target", "category", "classification"]:
        if candidate in rows[0]:
            label_col = candidate
            break
    if not label_col:
        raise ValueError(
            f"No label column found. Expected one of: label, class, is_injection, target, category. "
            f"Got columns: {list(rows[0].keys())}"
        )

    # Normalize labels to 0/1
    label_mapping = {
        "0": 0, "1": 1,
        "safe": 0, "injection": 1,
        "benign": 0, "malicious": 1,
        "normal": 0, "attack": 1,
        "legitimate": 0, "injected": 1,
        "false": 0, "true": 1,
        "no": 0, "yes": 1,
        "label_0": 0, "label_1": 1,
        "negative": 0, "positive": 1,
    }

    parsed = []
    for row in rows:
        text = str(row[text_col]).strip()
        raw_label = str(row[label_col]).strip().lower()
        if raw_label in label_mapping:
            label = label_mapping[raw_label]
        else:
            try:
                label = int(float(raw_label))
                if label not in (0, 1):
                    raise ValueError()
            except (ValueError, TypeError):
                raise ValueError(
                    f"Unknown label value: '{row[label_col]}'. "
                    f"Expected 0/1, safe/injection, benign/malicious, etc."
                )
        if text:
            parsed.append({"text": text, "label": label})

    return parsed


# ── Training (subprocess-based to avoid Python 3.14 pickle issues) ──

def _poll_training_subprocess(job_id, status_path, proc=None):
    """Background thread that polls the status file written by train_worker.py."""
    job = training_jobs[job_id]
    stale_count = 0
    while True:
        time.sleep(2)
        try:
            with open(status_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            job.update(data)
            stale_count = 0
            if data.get("status") in ("completed", "failed"):
                break
        except (json.JSONDecodeError, FileNotFoundError, OSError):
            stale_count += 1
            # If subprocess exited and status file never updated, read stderr log
            if proc and proc.poll() is not None and stale_count > 3:
                stderr_log = str(UPLOAD_DIR / f"train_stderr_{job_id}.log")
                err_msg = "Training subprocess crashed"
                try:
                    with open(stderr_log, "r", encoding="utf-8", errors="replace") as f:
                        err_msg = f.read().strip() or err_msg
                except OSError:
                    pass
                job["status"] = "failed"
                job["error"] = err_msg
                job["logs"].append({"msg": f"ERROR: {err_msg}"})
                break


# ── Routes ──

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/train")
def train_page():
    return render_template("train.html")


@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()
    text = data.get("text", "").strip()
    threshold = float(data.get("threshold", 0.5))

    if not text:
        return jsonify({"error": "Text is required"}), 400

    t0 = time.perf_counter()
    scores = classify_text(text)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    injection_score = scores.get("INJECTION", 0.0)
    safe_score = scores.get("SAFE", 0.0)
    is_injection = injection_score >= threshold

    return jsonify({
        "text": text,
        "threshold": threshold,
        "injection_score": round(injection_score, 6),
        "safe_score": round(safe_score, 6),
        "verdict": "INJECTION" if is_injection else "SAFE",
        "active_model": model_state["active"],
        "response_ms": round(elapsed_ms, 2),
        "raw_results": [
            {"label": label, "score": round(score, 6)}
            for label, score in scores.items()
        ],
    })


@app.route("/api/upload-dataset", methods=["POST"])
def upload_dataset():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in (".csv", ".json", ".jsonl"):
        return jsonify({"error": f"Unsupported format: {ext}. Use .csv, .json, or .jsonl"}), 400

    # Save file
    safe_name = f"{uuid.uuid4().hex[:8]}_{Path(file.filename).name}"
    filepath = UPLOAD_DIR / safe_name
    file.save(str(filepath))

    # Validate and preview
    try:
        parsed = parse_dataset_file(str(filepath))
        safe_count = sum(1 for r in parsed if r["label"] == 0)
        inj_count = sum(1 for r in parsed if r["label"] == 1)
        preview = parsed[:5]
    except Exception as e:
        filepath.unlink(missing_ok=True)
        return jsonify({"error": str(e)}), 400

    return jsonify({
        "filename": safe_name,
        "filepath": str(filepath),
        "total_samples": len(parsed),
        "safe_count": safe_count,
        "injection_count": inj_count,
        "preview": preview,
    })


@app.route("/api/start-training", methods=["POST"])
def start_training():
    import subprocess

    data = request.get_json()
    filepath = data.get("filepath")
    if not filepath or not Path(filepath).exists():
        return jsonify({"error": "Dataset file not found"}), 400

    config = {
        "dataset_path": filepath,
        "models_dir": str(MODELS_DIR),
        "model_name": data.get("model_name", f"finetuned-{int(time.time())}"),
        "base_model": data.get("base_model", "distilbert-base-multilingual-cased"),
        "epochs": int(data.get("epochs", 3)),
        "batch_size": int(data.get("batch_size", 8)),
        "learning_rate": float(data.get("learning_rate", 2e-5)),
        "weight_decay": float(data.get("weight_decay", 0.01)),
        "warmup_ratio": float(data.get("warmup_ratio", 0.1)),
        "max_length": int(data.get("max_length", 128)),
        "eval_split": float(data.get("eval_split", 0.1)),
        "logging_steps": int(data.get("logging_steps", 10)),
    }

    job_id = uuid.uuid4().hex[:12]

    # Write config file for the worker subprocess
    config_path = str(UPLOAD_DIR / f"train_config_{job_id}.json")
    status_path = str(UPLOAD_DIR / f"train_status_{job_id}.json")
    with open(config_path, "w") as f:
        json.dump(config, f)

    # Initialize status
    training_jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "progress": 0.0,
        "current_epoch": 0,
        "total_epochs": config["epochs"],
        "dataset_size": 0,
        "logs": [{"msg": "Starting training subprocess..."}],
        "eval_metrics": {},
        "error": None,
        "model_name": config["model_name"],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Write initial status file
    with open(status_path, "w") as f:
        json.dump(training_jobs[job_id], f)

    # Launch training as a completely separate Python process
    worker_script = str(BASE_DIR / "train_worker.py")
    stderr_log = str(UPLOAD_DIR / f"train_stderr_{job_id}.log")
    stderr_file = open(stderr_log, "w")
    proc = subprocess.Popen(
        ["py", worker_script, config_path, status_path],
        cwd=str(BASE_DIR),
        stdout=stderr_file,
        stderr=stderr_file,
    )

    # Start a poller thread that reads the status file
    poller = threading.Thread(
        target=_poll_training_subprocess, args=(job_id, status_path, proc), daemon=True,
    )
    poller.start()

    return jsonify({"job_id": job_id, "status": "queued"})


@app.route("/api/training-status/<job_id>")
def training_status(job_id):
    job = training_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/models")
def list_models():
    models = [{"name": "original", "description": "HikmaAI ONNX (original)"}]
    for d in sorted(MODELS_DIR.iterdir()):
        if d.is_dir() and not d.name.startswith("_cache_"):
            meta_path = d / "training_meta.json"
            meta = {}
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
            models.append({
                "name": d.name,
                "description": f"Fine-tuned ({meta.get('created_at', 'unknown')})",
                "dataset_size": meta.get("dataset_size"),
                "eval_metrics": meta.get("eval_metrics", {}),
                "config": meta.get("config", {}),
            })
    return jsonify({"models": models, "active": model_state["active"]})


@app.route("/api/switch-model", methods=["POST"])
def switch_model():
    data = request.get_json()
    name = data.get("model_name", "original")

    if name == "original":
        model_state["active"] = "original"
        return jsonify({"active": "original", "msg": "Switched to original ONNX model"})

    model_dir = MODELS_DIR / name
    if not model_dir.exists():
        return jsonify({"error": f"Model '{name}' not found"}), 404

    try:
        load_trained_model(name)
        return jsonify({"active": name, "msg": f"Switched to model '{name}'"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


load_original_model()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
