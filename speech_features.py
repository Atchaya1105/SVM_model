"""
speech_features.py
==================
Extracts acoustic + linguistic + wav2vec2 deep embeddings from audio.

Feature groups:
  1. Acoustic (8)       — librosa signal processing
  2. MFCC (26)          — 13 means + 13 stds
  3. Linguistic (7)     — NLTK / rule-based from Whisper transcript
  4. wav2vec2 (768)     — deep speech representations (facebook/wav2vec2-base)
                          Used in ADReSS 2021 winning submissions

Total feature vector: 8 + 26 + 7 + 768 = 809 dimensions
"""

import librosa
import numpy as np
import whisper
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import re
import os

nltk.download('punkt',                      quiet=True)
nltk.download('punkt_tab',                  quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# ─── Lazy-loaded models ────────────────────────────────────────────────────
_whisper_model  = None
_wav2vec_model  = None
_wav2vec_proc   = None
_wav2vec_available = None   # None = not yet checked


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def _check_wav2vec():
    """Return True if transformers + torch are installed."""
    global _wav2vec_available
    if _wav2vec_available is not None:
        return _wav2vec_available
    try:
        import torch                                    # noqa: F401
        from transformers import Wav2Vec2Processor, Wav2Vec2Model  # noqa: F401
        _wav2vec_available = True
    except ImportError:
        _wav2vec_available = False
        print("[speech_features] wav2vec2 not available — "
              "install: pip install transformers torch")
    return _wav2vec_available


def get_wav2vec():
    """
    Lazy-load facebook/wav2vec2-base.
    Downloads ~360 MB on first call, cached by HuggingFace.
    """
    global _wav2vec_model, _wav2vec_proc
    if _wav2vec_model is None:
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
        import torch
        model_id = "facebook/wav2vec2-base"
        print(f"[speech_features] Loading {model_id} …")
        _wav2vec_proc  = Wav2Vec2Processor.from_pretrained(model_id)
        _wav2vec_model = Wav2Vec2Model.from_pretrained(model_id)
        _wav2vec_model.eval()
        print("[speech_features] wav2vec2 ready.")
    return _wav2vec_proc, _wav2vec_model


# ═══════════════════════════════════════════════════════════════════════════
# 1. ACOUSTIC FEATURES  (8 scalars)
# ═══════════════════════════════════════════════════════════════════════════
def extract_acoustic_features(audio_path: str) -> dict:
    """
    Signal-processing features that correlate with cognitive decline:
    speech ratio, pause statistics, pitch, spectral shape, ZCR.
    """
    y, sr = librosa.load(audio_path, sr=16000)

    # Speech / silence segmentation
    intervals = librosa.effects.split(y, top_db=25)
    speech_duration = sum((e - s) for s, e in intervals) / sr
    total_duration  = len(y) / sr
    speech_ratio    = speech_duration / total_duration if total_duration > 0 else 0

    # Pause analysis (gaps > 0.3 s)
    pause_count, pause_durations = 0, []
    for i in range(1, len(intervals)):
        gap = (intervals[i][0] - intervals[i-1][1]) / sr
        if gap > 0.3:
            pause_count += 1
            pause_durations.append(gap)
    avg_pause = float(np.mean(pause_durations)) if pause_durations else 0.0

    # MFCCs
    mfccs      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfccs, axis=1)
    mfcc_stds  = np.std(mfccs,  axis=1)

    # Pitch
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pv         = pitches[pitches > 0]
    pitch_mean = float(np.mean(pv)) if len(pv) > 0 else 0.0
    pitch_std  = float(np.std(pv))  if len(pv) > 0 else 0.0

    # Spectral
    spec_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spec_rolloff  = float(np.mean(librosa.feature.spectral_rolloff(y=y,  sr=sr)))
    zcr           = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    return {
        "speech_ratio":       round(speech_ratio, 4),
        "pause_count":        pause_count,
        "avg_pause_duration": round(avg_pause, 4),
        "pitch_mean":         round(pitch_mean, 4),
        "pitch_std":          round(pitch_std, 4),
        "spectral_centroid":  round(spec_centroid, 4),
        "spectral_rolloff":   round(spec_rolloff, 4),
        "zero_crossing_rate": round(zcr, 4),
        "mfcc_means":         mfcc_means.tolist(),
        "mfcc_stds":          mfcc_stds.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. LINGUISTIC FEATURES  (7 scalars + transcript)
# ═══════════════════════════════════════════════════════════════════════════
def extract_linguistic_features(audio_path: str) -> dict:
    """
    Transcribe with Whisper then compute NLP markers validated in
    DementiaBank / ADReSS research.
    """
    result = get_whisper_model().transcribe(audio_path, language="en")
    text   = result["text"].strip()

    if not text:
        return {"error": "No speech detected", "transcript": ""}

    words          = word_tokenize(text.lower())
    sentences      = sent_tokenize(text)
    word_count     = len(words)
    sentence_count = len(sentences)

    # Vocabulary richness (Type-Token Ratio)
    unique_words = set(words)
    ttr = len(unique_words) / word_count if word_count > 0 else 0

    # Sentence complexity
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

    # Filler words
    fillers = ['um', 'uh', 'like', 'you know', 'sort of',
               'kind of', 'basically', 'i mean']
    filler_count = sum(text.lower().count(f) for f in fillers)
    filler_ratio = filler_count / word_count if word_count > 0 else 0

    # Repetition
    word_freq     = Counter(words)
    repeated      = sum(1 for w, c in word_freq.items() if c > 2 and len(w) > 3)
    repetition_ratio = repeated / len(unique_words) if unique_words else 0

    # POS features
    pos_tags = nltk.pos_tag(words)
    nouns    = sum(1 for _, t in pos_tags if t.startswith('NN'))
    verbs    = sum(1 for _, t in pos_tags if t.startswith('VB'))
    pronouns = sum(1 for _, t in pos_tags if t in ('PRP', 'PRP$', 'WP', 'WP$'))
    noun_verb_ratio = nouns / verbs if verbs > 0 else 0
    pronoun_ratio   = pronouns / word_count if word_count > 0 else 0

    return {
        "transcript":          text,
        "word_count":          word_count,
        "sentence_count":      sentence_count,
        "avg_sentence_length": round(avg_sentence_length, 4),
        "type_token_ratio":    round(ttr, 4),
        "filler_ratio":        round(filler_ratio, 4),
        "repetition_ratio":    round(repetition_ratio, 4),
        "noun_verb_ratio":     round(noun_verb_ratio, 4),
        "pronoun_ratio":       round(pronoun_ratio, 4),
        "unique_word_count":   len(unique_words),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. WAV2VEC 2.0 DEEP EMBEDDINGS  (768-dim)
# ═══════════════════════════════════════════════════════════════════════════
def extract_wav2vec_embeddings(audio_path: str) -> np.ndarray:
    """
    Extract 768-dimensional mean-pooled hidden states from
    facebook/wav2vec2-base.

    These deep representations capture:
      - Articulatory precision
      - Prosodic patterns
      - Voice quality changes
      - Temporal speech dynamics

    All of which degrade measurably in Alzheimer's disease.
    Used in ADReSS 2020/2021 winning systems (Interspeech).

    Returns zeros (768,) if transformers/torch not installed.
    """
    if not _check_wav2vec():
        return np.zeros(768, dtype=np.float32)

    import torch

    proc, wav2vec = get_wav2vec()

    # Load audio at 16 kHz (wav2vec2 requirement)
    y, sr = librosa.load(audio_path, sr=16000)

    # Chunk into 30-second segments to avoid OOM on long recordings
    chunk_size = 16000 * 30
    chunks     = [y[i:i + chunk_size] for i in range(0, len(y), chunk_size)]
    all_hidden = []

    with torch.no_grad():
        for chunk in chunks:
            if len(chunk) < 400:          # skip very short trailing chunks
                continue
            inputs = proc(
                chunk,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )
            outputs = wav2vec(**inputs)
            # Mean-pool over time → (1, 768)
            hidden = outputs.last_hidden_state.mean(dim=1).squeeze(0)
            all_hidden.append(hidden.numpy())

    if not all_hidden:
        return np.zeros(768, dtype=np.float32)

    # Average across chunks
    embedding = np.mean(np.stack(all_hidden, axis=0), axis=0)
    return embedding.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# 4. MMSE PROXY SCORE  (Mini-Mental State Examination estimate)
# ═══════════════════════════════════════════════════════════════════════════
def estimate_mmse_proxy(ling: dict, acou: dict) -> dict:
    """
    Estimates a proxy MMSE-like score (0–30) from speech features.

    The real MMSE is a 30-point clinical test. This proxy uses the
    same linguistic markers validated in:
      - Fraser et al. (2016) — Linguistic Features Identify Alzheimer's
      - Luz et al. (2021)    — ADReSS 2021 Challenge

    NOT a replacement for clinical MMSE — a research-grade approximation.
    """
    score = 30.0   # start at max, deduct for each marker

    ttr        = ling.get("type_token_ratio",    1.0)
    filler     = ling.get("filler_ratio",        0.0)
    rep        = ling.get("repetition_ratio",    0.0)
    avg_len    = ling.get("avg_sentence_length", 10.0)
    word_count = ling.get("word_count",          50)
    pauses     = acou.get("pause_count",         0)
    speech_r   = acou.get("speech_ratio",        1.0)

    # Vocabulary richness (max deduction: 8)
    if ttr < 0.30:   score -= 8
    elif ttr < 0.45: score -= 5
    elif ttr < 0.60: score -= 2

    # Filler words (max deduction: 5)
    if filler > 0.15:  score -= 5
    elif filler > 0.08: score -= 3
    elif filler > 0.04: score -= 1

    # Repetition (max deduction: 5)
    if rep > 0.35:   score -= 5
    elif rep > 0.20: score -= 3
    elif rep > 0.10: score -= 1

    # Sentence length (max deduction: 4)
    if avg_len < 3:  score -= 4
    elif avg_len < 5: score -= 2
    elif avg_len < 7: score -= 1

    # Pause count (max deduction: 4)
    if pauses > 15:  score -= 4
    elif pauses > 8: score -= 2
    elif pauses > 4: score -= 1

    # Speech ratio — very low = mostly silence (max deduction: 4)
    if speech_r < 0.30:  score -= 4
    elif speech_r < 0.50: score -= 2
    elif speech_r < 0.65: score -= 1

    score = max(0.0, min(30.0, round(score, 1)))

    # Interpret
    if score >= 24:
        interpretation = "Normal cognition (≥24)"
        severity       = "normal"
    elif score >= 18:
        interpretation = "Mild cognitive impairment (18–23)"
        severity       = "mild"
    elif score >= 10:
        interpretation = "Moderate dementia (10–17)"
        severity       = "moderate"
    else:
        interpretation = "Severe dementia (<10)"
        severity       = "severe"

    return {
        "mmse_proxy_score":   score,
        "mmse_interpretation": interpretation,
        "mmse_severity":       severity,
        "mmse_note": (
            "Proxy estimate from speech features. "
            "Not a clinical diagnosis. "
            "Based on Fraser et al. (2016) & ADReSS 2021 methodology."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. COMBINED FEATURE VECTOR  (for downstream ML)
# ═══════════════════════════════════════════════════════════════════════════
def get_speech_feature_vector(audio_path: str) -> np.ndarray:
    """
    Returns a single float32 array:
      [acoustic(8), mfcc_means(13), mfcc_stds(13), linguistic(7), wav2vec(768)]
    Total: 809 dimensions.

    Use this to train a classifier on DementiaBank labels.
    """
    acoustic   = extract_acoustic_features(audio_path)
    linguistic = extract_linguistic_features(audio_path)
    embeddings = extract_wav2vec_embeddings(audio_path)

    handcrafted = np.array([
        acoustic["speech_ratio"],
        acoustic["pause_count"],
        acoustic["avg_pause_duration"],
        acoustic["pitch_mean"],
        acoustic["pitch_std"],
        acoustic["spectral_centroid"],
        acoustic["spectral_rolloff"],
        acoustic["zero_crossing_rate"],
        *acoustic["mfcc_means"],                        # 13
        *acoustic["mfcc_stds"],                         # 13
        linguistic.get("word_count",          0),
        linguistic.get("avg_sentence_length", 0),
        linguistic.get("type_token_ratio",    0),
        linguistic.get("filler_ratio",        0),
        linguistic.get("repetition_ratio",    0),
        linguistic.get("noun_verb_ratio",     0),
        linguistic.get("pronoun_ratio",       0),
    ], dtype=np.float32)

    return np.concatenate([handcrafted, embeddings])
