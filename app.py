from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import joblib
from PIL import Image
import io
import base64
import json
import os
from datetime import datetime

# =========================
# APP INIT
# =========================
app = FastAPI(title="Alzheimer MRI Prediction API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("svm_model_fast.pkl")

# LinearSVC inside Pipeline — extract the SVM step for decision_function
svm_step = model.named_steps["svm"]

CLASSES = {
    0: "Mild Demented",
    1: "Moderate Demented",
    2: "Non Demented",
    3: "Very Mild Demented"
}

CLASS_INFO = {
    "Mild Demented": {
        "description": "Noticeable memory problems affecting daily tasks. Patient may forget recent events, repeat questions, or have trouble with complex activities like managing finances.",
        "severity": "moderate",
        "color": "#f59e0b",
        "icon": "⚠️"
    },
    "Moderate Demented": {
        "description": "Significant cognitive decline requiring assistance with most daily activities. Increased memory loss, confusion about time/place, and personality changes are common.",
        "severity": "high",
        "color": "#ef4444",
        "icon": "🔴"
    },
    "Non Demented": {
        "description": "No signs of cognitive impairment detected in this MRI scan. Brain structure appears within normal parameters.",
        "severity": "none",
        "color": "#10b981",
        "icon": "✅"
    },
    "Very Mild Demented": {
        "description": "Very early stage cognitive changes. Slight memory lapses that may resemble normal aging. Regular monitoring recommended.",
        "severity": "low",
        "color": "#3b82f6",
        "icon": "🔵"
    }
}

IMG_SIZE = 128
LOG_FILE = "prediction_log.json"
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_MB = 10

# =========================
# VALIDATION
# =========================
def validate_upload(file: UploadFile, image_bytes: bytes):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Please upload a JPEG, PNG, WebP, or BMP image."
        )
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > MAX_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum size is {MAX_MB} MB."
        )
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is corrupted or not a valid image.")

# =========================
# PREPROCESSING
# Matches exactly how training was done:
# cv2.imread(path, 0) → grayscale, resize 128x128, flatten
# Pipeline handles StandardScaler internally
# =========================
def preprocess_image(image: Image.Image):
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    flat = resized.flatten().astype(np.float32)
    return flat.reshape(1, -1), resized  # raw flat (scaler applied by pipeline), and 2D for heatmap

# =========================
# CONFIDENCE SCORES via decision_function
# LinearSVC returns raw margin scores (not probabilities).
# We softmax them to get interpretable relative confidence.
# =========================
def get_confidence_scores(raw_flat: np.ndarray) -> tuple[dict, float]:
    """
    Uses the full pipeline to transform input (applies StandardScaler),
    then calls decision_function on the SVM step.
    Returns softmax-normalized scores and the top class confidence.
    """
    # Transform through scaler only
    scaler = model.named_steps["scaler"]
    X_scaled = scaler.transform(raw_flat)

    # Decision function on the LinearSVC
    df = svm_step.decision_function(X_scaled)[0]  # shape: (n_classes,)

    # Softmax for interpretable percentages
    df_shifted = df - df.max()
    exp_df = np.exp(df_shifted)
    softmax = exp_df / exp_df.sum()

    scores = {}
    for idx, prob in enumerate(softmax):
        scores[CLASSES[idx]] = round(float(prob) * 100, 2)

    predicted_idx = int(np.argmax(softmax))
    top_confidence = round(float(softmax[predicted_idx]) * 100, 2)
    return scores, top_confidence

# =========================
# SALIENCY HEATMAP (Patch Occlusion)
# Works with LinearSVC — no backprop needed.
# Measures how much occluding each region changes the decision score.
# =========================
def generate_heatmap(gray_2d: np.ndarray, predicted_idx: int) -> str | None:
    try:
        scaler = model.named_steps["scaler"]
        patch = 16
        steps = IMG_SIZE // patch
        sensitivity = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

        # Baseline decision score for predicted class
        baseline_flat = gray_2d.flatten().astype(np.float32).reshape(1, -1)
        baseline_scaled = scaler.transform(baseline_flat)
        baseline_score = svm_step.decision_function(baseline_scaled)[0][predicted_idx]

        for i in range(steps):
            for j in range(steps):
                perturbed = gray_2d.copy().astype(np.float32)
                patch_mean = perturbed[i*patch:(i+1)*patch, j*patch:(j+1)*patch].mean()
                perturbed[i*patch:(i+1)*patch, j*patch:(j+1)*patch] = patch_mean

                p_flat = perturbed.flatten().reshape(1, -1)
                p_scaled = scaler.transform(p_flat)
                p_score = svm_step.decision_function(p_scaled)[0][predicted_idx]

                drop = baseline_score - p_score
                sensitivity[i*patch:(i+1)*patch, j*patch:(j+1)*patch] = max(drop, 0)

        # Normalize + colorize
        if sensitivity.max() > 0:
            sensitivity = (sensitivity / sensitivity.max() * 255).astype(np.uint8)
        else:
            sensitivity = np.zeros_like(sensitivity, dtype=np.uint8)

        heatmap = cv2.applyColorMap(sensitivity, cv2.COLORMAP_JET)
        heatmap = cv2.GaussianBlur(heatmap, (11, 11), 0)
        original_bgr = cv2.cvtColor(gray_2d, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(original_bgr, 0.5, heatmap, 0.5, 0)

        _, buffer = cv2.imencode(".png", overlay)
        return base64.b64encode(buffer).decode("utf-8")

    except Exception as e:
        print(f"[Heatmap error] {e}")
        return None

# =========================
# LOGGING
# =========================
def log_prediction(filename: str, prediction: str, confidence: float, scores: dict):
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "filename": filename,
        "prediction": prediction,
        "confidence_percent": confidence,
        "all_scores": scores
    }
    log = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE) as f:
                log = json.load(f)
        except Exception:
            log = []
    log.append(record)
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)

# =========================
# ENDPOINTS
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    validate_upload(file, image_bytes)

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    raw_flat, gray_2d = preprocess_image(image)

    # Predict using full pipeline (scaler + LinearSVC)
    prediction_idx = int(model.predict(raw_flat)[0])
    prediction_label = CLASSES[prediction_idx]

    # Confidence scores
    scores, top_confidence = get_confidence_scores(raw_flat)

    # Heatmap
    heatmap_b64 = generate_heatmap(gray_2d, prediction_idx)

    # Log
    log_prediction(file.filename or "unknown", prediction_label, top_confidence, scores)

    info = CLASS_INFO[prediction_label]

    return JSONResponse(content={
        "prediction_class": prediction_label,
        "confidence_percent": top_confidence,
        "all_class_scores": scores,
        "description": info["description"],
        "severity": info["severity"],
        "color": info["color"],
        "icon": info["icon"],
        "heatmap_base64": heatmap_b64,
        "note": "LinearSVC confidence scores are softmax-normalized decision margins, not true probabilities."
    })


@app.get("/history")
async def get_history(limit: int = 20):
    if not os.path.exists(LOG_FILE):
        return {"history": [], "total": 0}
    try:
        with open(LOG_FILE) as f:
            log = json.load(f)
    except Exception:
        return {"history": [], "total": 0}
    return {"history": log[-limit:][::-1], "total": len(log)}


@app.delete("/history")
async def clear_history():
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    return {"message": "History cleared."}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_type": type(svm_step).__name__,
        "supports_proba": hasattr(svm_step, "predict_proba"),
        "classes": list(CLASSES.values())
    }


# =========================
# SPEECH / LINGUISTIC ANALYSIS
# =========================
import tempfile
from speech_features import (
    extract_acoustic_features,
    extract_linguistic_features,
    extract_wav2vec_embeddings,
    estimate_mmse_proxy,
    _check_wav2vec,
)

AUDIO_ALLOWED = {"audio/wav", "audio/wave", "audio/mpeg", "audio/mp4",
                 "audio/x-m4a", "audio/ogg", "audio/webm", "audio/flac",
                 "application/octet-stream"}


def calculate_linguistic_risk(ling: dict, acou: dict) -> float:
    """
    Rule-based risk score 0–100 from validated AD speech markers.
    Calibrated against ADReSS 2021 feature importance rankings.
    """
    score = 0.0

    ttr = ling.get("type_token_ratio", 1.0)
    if ttr < 0.35:   score += 30
    elif ttr < 0.55: score += 15

    filler = ling.get("filler_ratio", 0.0)
    if filler > 0.10:   score += 20
    elif filler > 0.05: score += 10

    rep = ling.get("repetition_ratio", 0.0)
    if rep > 0.30:   score += 20
    elif rep > 0.15: score += 10

    avg_len = ling.get("avg_sentence_length", 10.0)
    if avg_len < 4:  score += 15
    elif avg_len < 7: score += 7

    pauses = acou.get("pause_count", 0)
    if pauses > 10:  score += 15
    elif pauses > 5: score += 7

    return min(round(float(score), 2), 100.0)


def get_risk_level(score: float) -> str:
    if score >= 60: return "High Linguistic Risk"
    if score >= 30: return "Moderate Linguistic Risk"
    return "Low Linguistic Risk"


def get_risk_severity(score: float) -> str:
    if score >= 60: return "high"
    if score >= 30: return "moderate"
    return "low"


@app.post("/analyze/speech")
async def analyze_speech(file: UploadFile = File(...)):
    """
    Accepts an audio file (WAV / MP3 / M4A / OGG / WEBM).

    Returns:
      - transcript              (Whisper ASR)
      - acoustic_features       (librosa signal processing)
      - linguistic_features     (NLTK / rule-based)
      - wav2vec_available        (bool — whether deep embeddings ran)
      - linguistic_risk_score   (0–100 rule-based)
      - risk_level / risk_severity
      - mmse_proxy              (MMSE-like score 0–30, research-grade)
    """
    audio_bytes = await file.read()

    if len(audio_bytes) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio file too large. Max 50 MB.")

    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # ── Core feature extraction ──────────────────────────────────────
        acoustic   = extract_acoustic_features(tmp_path)
        linguistic = extract_linguistic_features(tmp_path)

        # ── wav2vec2 deep embeddings (optional — needs transformers+torch) ─
        wav2vec_ok = _check_wav2vec()
        wav2vec_summary = {}
        if wav2vec_ok:
            embeddings = extract_wav2vec_embeddings(tmp_path)
            # Summarise the 768-dim vector: mean, std, L2-norm
            wav2vec_summary = {
                "embedding_dim":  int(embeddings.shape[0]),
                "embedding_mean": round(float(embeddings.mean()), 6),
                "embedding_std":  round(float(embeddings.std()),  6),
                "embedding_norm": round(float(np.linalg.norm(embeddings)), 4),
                "note": (
                    "facebook/wav2vec2-base hidden states (mean-pooled). "
                    "Used in ADReSS 2021 winning submissions."
                ),
            }

        # ── MMSE proxy score ─────────────────────────────────────────────
        mmse = estimate_mmse_proxy(linguistic, acoustic)

        # ── Rule-based risk score ────────────────────────────────────────
        risk_score    = calculate_linguistic_risk(linguistic, acoustic)
        risk_level    = get_risk_level(risk_score)
        risk_severity = get_risk_severity(risk_score)

        # ── Build response ───────────────────────────────────────────────
        acoustic_summary = {k: v for k, v in acoustic.items()
                            if k not in ("mfcc_means", "mfcc_stds")}
        acoustic_summary["mfcc_sample"] = acoustic.get("mfcc_means", [])[:5]

        response = {
            "transcript": linguistic.get("transcript", ""),

            "acoustic_features": acoustic_summary,

            "linguistic_features": {
                "word_count":          linguistic.get("word_count",          0),
                "unique_word_count":   linguistic.get("unique_word_count",   0),
                "sentence_count":      linguistic.get("sentence_count",      0),
                "avg_sentence_length": linguistic.get("avg_sentence_length", 0),
                "type_token_ratio":    linguistic.get("type_token_ratio",    0),
                "filler_ratio":        linguistic.get("filler_ratio",        0),
                "repetition_ratio":    linguistic.get("repetition_ratio",    0),
                "noun_verb_ratio":     linguistic.get("noun_verb_ratio",     0),
                "pronoun_ratio":       linguistic.get("pronoun_ratio",       0),
            },

            # wav2vec2 deep embeddings summary
            "wav2vec_available": wav2vec_ok,
            "wav2vec_embeddings": wav2vec_summary if wav2vec_ok else {
                "note": (
                    "Install transformers + torch to enable wav2vec2 embeddings: "
                    "pip install transformers torch"
                )
            },

            # MMSE proxy (research-grade, not clinical)
            "mmse_proxy": mmse,

            # Rule-based risk
            "linguistic_risk_score": risk_score,
            "risk_level":            risk_level,
            "risk_severity":         risk_severity,
        }

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speech analysis failed: {str(e)}")
    finally:
        os.unlink(tmp_path)
