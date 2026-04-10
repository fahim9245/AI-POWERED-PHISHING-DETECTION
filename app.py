"""
app.py — PhishGuard AI Flask API Server
========================================
Serves predictions from the trained ML model via REST API.
The frontend (app.js) calls /predict to get ML-based scores.

Endpoints:
    GET  /health          — Health check + model info
    POST /predict         — Predict single URL
    POST /predict/batch   — Predict multiple URLs
    GET  /features?url=…  — Extract features only (debug)

Usage:
    python train.py        # Train model first (one time)
    python app.py          # Start server on http://localhost:5000
"""

import os
import time
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

from features import extract_features
from model import load_model, predict, MODEL_PATH


# ── App Setup ─────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)   # Allow requests from the frontend (different origin / port)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Load Model (once at startup) ──────────────────────────────────────────────

MODEL_PAYLOAD = None

def load_model_once():
    global MODEL_PAYLOAD
    if MODEL_PAYLOAD is not None:
        return MODEL_PAYLOAD
    try:
        MODEL_PAYLOAD = load_model(MODEL_PATH)
        log.info(f"✓ Model loaded: {MODEL_PAYLOAD['model_name']} v{MODEL_PAYLOAD.get('version','?')}")
        return MODEL_PAYLOAD
    except FileNotFoundError as e:
        log.error(str(e))
        return None


# ── Request Validation ────────────────────────────────────────────────────────

def validate_url(url) -> tuple[str | None, str | None]:
    """Returns (cleaned_url, error_message)"""
    if not url:
        return None, "Missing 'url' field in request body"
    url = url.strip()
    if len(url) > 2048:
        return None, "URL exceeds maximum length of 2048 characters"
    if not url.startswith(("http://", "https://", "ftp://")) and "." not in url:
        return None, "URL appears invalid — must contain a domain"
    return url, None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check — confirms server and model are online."""
    payload = load_model_once()
    if payload is None:
        return jsonify({
            "status":  "degraded",
            "error":   "Model not loaded. Run: python train.py",
            "model":   None,
        }), 503

    return jsonify({
        "status":       "ok",
        "model":        payload["model_name"],
        "version":      payload.get("version", "unknown"),
        "features":     len(payload["feature_names"]),
        "server":       "PhishGuard AI v2.0",
    })


@app.route("/predict", methods=["POST"])
def predict_url():
    """
    Predict phishing probability for a single URL.

    Request:
        POST /predict
        Content-Type: application/json
        { "url": "https://suspicious-site.tk/login" }

    Response:
        {
            "score":       0.87,        // phishing probability 0–1
            "score_pct":   87,          // as percentage
            "verdict":     "PHISHING",  // PHISHING / SUSPICIOUS / LOW RISK / SAFE
            "probability": 0.87,
            "confidence":  0.91,
            "features":    { ... },     // all extracted features
            "importances": { ... },     // feature importances from model
            "reasons":     [ ... ],     // human-readable indicators
            "model_name":  "RandomForestClassifier",
            "elapsed_ms":  12.3
        }
    """
    payload = load_model_once()
    if payload is None:
        return jsonify({"error": "Model not loaded. Run: python train.py"}), 503

    data = request.get_json(silent=True) or {}
    url, err = validate_url(data.get("url"))
    if err:
        return jsonify({"error": err}), 400

    t0     = time.perf_counter()
    result = predict(url, payload)
    elapsed = round((time.perf_counter() - t0) * 1000, 2)

    result["url"]        = url
    result["elapsed_ms"] = elapsed

    log.info(f"PREDICT {result['verdict']:12s} score={result['score_pct']:3d}%  {url[:80]}")
    return jsonify(result)


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Predict phishing probability for multiple URLs.

    Request:
        POST /predict/batch
        Content-Type: application/json
        { "urls": ["https://url1.com", "http://phish.tk/login", ...] }

    Response:
        { "results": [ { ...same as /predict... }, ... ] }
    """
    payload = load_model_once()
    if payload is None:
        return jsonify({"error": "Model not loaded. Run: python train.py"}), 503

    data = request.get_json(silent=True) or {}
    urls = data.get("urls", [])

    if not isinstance(urls, list):
        return jsonify({"error": "'urls' must be an array"}), 400
    if len(urls) > 100:
        return jsonify({"error": "Batch limit is 100 URLs per request"}), 400

    results = []
    for raw_url in urls:
        url, err = validate_url(raw_url)
        if err:
            results.append({"url": raw_url, "error": err})
            continue

        result      = predict(url, payload)
        result["url"] = url
        results.append(result)

    phish_count = sum(1 for r in results if r.get("verdict") == "PHISHING")
    log.info(f"BATCH {len(results)} URLs — {phish_count} phishing detected")

    return jsonify({
        "results":      results,
        "total":        len(results),
        "phishing":     phish_count,
        "suspicious":   sum(1 for r in results if r.get("verdict") == "SUSPICIOUS"),
        "safe":         sum(1 for r in results if r.get("score_pct", 100) < 40),
    })


@app.route("/features", methods=["GET"])
def get_features():
    """
    Debug endpoint — extract and return all features for a URL.

    Usage: GET /features?url=https://example.com
    """
    url = request.args.get("url", "").strip()
    if not url:
        return jsonify({"error": "Missing ?url= parameter"}), 400

    features = extract_features(url)
    return jsonify({
        "url":      url,
        "features": features,
        "count":    len(features),
    })


@app.route("/", methods=["GET"])
def index():
    """Root endpoint — API info."""
    return jsonify({
        "name":     "PhishGuard AI API",
        "version":  "2.0.0",
        "endpoints": {
            "GET  /health":           "Health check",
            "POST /predict":          "Predict single URL",
            "POST /predict/batch":    "Predict multiple URLs",
            "GET  /features?url=...": "Extract features (debug)",
        },
        "docs": "Open index.html in your browser for the full UI",
    })


# ── Error Handlers ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found", "available": [
        "GET /health", "POST /predict", "POST /predict/batch",
    ]}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(500)
def internal_error(e):
    log.exception("Internal server error")
    return jsonify({"error": "Internal server error"}), 500


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  PhishGuard AI — API Server")
    print("="*60)

    # Pre-load model
    p = load_model_once()
    if p is None:
        print("\n⚠  Model not found! Run this first:")
        print("     python train.py\n")
    else:
        print(f"\n✓ Model: {p['model_name']} ({len(p['feature_names'])} features)")
        print("✓ CORS enabled — frontend can connect\n")

    print("Starting server on http://localhost:5000")
    print("Press Ctrl+C to stop\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,     # Set True for development auto-reload
        threaded=True,   # Handle concurrent requests
    )
