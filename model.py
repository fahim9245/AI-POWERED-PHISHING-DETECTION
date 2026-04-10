"""
model.py — PhishGuard AI Machine Learning Model
================================================
Trains a Random Forest classifier on URL features.
Saves model + feature importances to disk as phishguard_model.pkl.

Usage:
    python model.py          # Train and save model
    python model.py --eval   # Train, evaluate, and show metrics
"""

import os
import pickle
import argparse
import numpy as np
from features import extract_features, features_to_vector, get_feature_names

# scikit-learn imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ── Model Config ─────────────────────────────────────────────────────────────

MODEL_PATH   = "phishguard_model.pkl"
MODEL_NAME   = "RandomForestClassifier"

RF_PARAMS = {
    "n_estimators":      200,
    "max_depth":         12,
    "min_samples_split": 4,
    "min_samples_leaf":  2,
    "max_features":      "sqrt",
    "random_state":      42,
    "n_jobs":            -1,
    "class_weight":      "balanced",
}


# ── Dataset ───────────────────────────────────────────────────────────────────

# Labeled URL dataset  (label: 1 = phishing, 0 = legitimate)
LABELED_URLS = [
    # ── Phishing (label=1) ──────────────────────────────────────────────────
    ("http://paypa1-secure-login.tk/account/verify",                1),
    ("http://192.168.50.1/login?ref=paypal",                        1),
    ("https://amazon-account-suspended.ml/confirm-identity",        1),
    ("http://secure.bank0famerica.update-credentials.xyz/verify",   1),
    ("http://appleid.apple.com.phish-site.tk/login",                1),
    ("https://microsoft-account.update-required.ml/signin",         1),
    ("http://paypal.secure-update.gq/confirm?user=1234",            1),
    ("https://netflix-billing.update-now.cf/payment",               1),
    ("http://192.0.2.1/banking/login.php",                          1),
    ("http://secure-bankofamerica.fake-site.ml/signin",             1),
    ("https://google.com.phishing-domain.tk/auth",                  1),
    ("http://ebay-support.account-verify.xyz/login",                1),
    ("https://amazon.account-security.update.ml/verify",            1),
    ("http://login.chase.bank-secure-update.gq/auth",               1),
    ("http://icloud-id.apple.com.verify-now.tk/account",            1),
    ("http://confirm-identity.wellsfargo.phish.ml/secure",          1),
    ("https://paypal-alerts.suspicious-domain.cf/update",           1),
    ("http://www.secure-login-microsoft365.xyz/portal",             1),
    ("http://amazon-winner-prize.click/claim?ref=2024",             1),
    ("http://bit.ly/free-bitcoin-claim-2024",                       1),
    ("http://dropbox.com.account.update-now.ml/signin",             1),
    ("https://instagram.com.verify-account.gq/security",            1),
    ("http://login.facebook.com.phish.tk/auth",                     1),
    ("http://citibank.account-suspended.xyz/login",                 1),
    ("http://secure.update.paypal.com.phish.cf/billing",            1),
    ("http://account-suspended.netflix.ml/reactivate",              1),
    ("https://apple.com.id-verify.phishing.top/icloud",             1),
    ("http://192.168.1.1:8080/bank/login",                          1),
    ("http://free-crypto-winner.tk/claim?wallet=xxx",               1),
    ("http://urgently-verify-your-amazon-account.xyz/confirm",      1),

    # ── Legitimate (label=0) ────────────────────────────────────────────────
    ("https://google.com",                                          0),
    ("https://www.amazon.com/products",                             0),
    ("https://github.com/user/repo",                                0),
    ("https://stackoverflow.com/questions/12345",                   0),
    ("https://wikipedia.org/wiki/Python",                           0),
    ("https://linkedin.com/in/username",                            0),
    ("https://youtube.com/watch?v=abc123",                          0),
    ("https://reddit.com/r/programming",                            0),
    ("https://twitter.com/user/status/123",                         0),
    ("https://microsoft.com/en-us/windows",                         0),
    ("https://apple.com/iphone",                                    0),
    ("https://paypal.com/signin",                                   0),
    ("https://netflix.com/browse",                                   0),
    ("https://facebook.com/login",                                   0),
    ("https://instagram.com/explore",                               0),
    ("https://docs.python.org/3/library/os.html",                   0),
    ("https://aws.amazon.com/s3/",                                   0),
    ("https://developer.mozilla.org/en-US/docs/Web",                0),
    ("https://npmjs.com/package/express",                            0),
    ("https://pypi.org/project/flask/",                              0),
    ("https://cloudflare.com/products/cdn/",                        0),
    ("https://digitalocean.com/pricing",                            0),
    ("https://stripe.com/docs/api",                                  0),
    ("https://twilio.com/docs/sms",                                  0),
    ("https://medium.com/@author/article",                          0),
    ("https://news.ycombinator.com/",                               0),
    ("https://coursera.org/learn/machine-learning",                  0),
    ("https://udemy.com/course/python-bootcamp",                    0),
    ("https://kaggle.com/competitions",                              0),
    ("https://huggingface.co/models",                               0),
]


def build_dataset():
    """Convert labeled URLs to feature vectors and labels."""
    X, y = [], []
    for url, label in LABELED_URLS:
        feats = extract_features(url)
        vec   = features_to_vector(feats)
        X.append(vec)
        y.append(label)
    return np.array(X, dtype=float), np.array(y)


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(X, y, evaluate: bool = False):
    """
    Train a Random Forest model and return the fitted model.
    Optionally prints evaluation metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)

    if evaluate:
        _evaluate_model(model, X_train, X_test, y_train, y_test)

    return model


def _evaluate_model(model, X_train, X_test, y_train, y_test):
    """Print detailed evaluation metrics."""
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]
    train_score = model.score(X_train, y_train)
    test_score  = accuracy_score(y_test, y_pred)
    auc         = roc_auc_score(y_test, y_prob)

    print("\n" + "="*60)
    print(" PhishGuard AI — Model Evaluation")
    print("="*60)
    print(f"  Training Accuracy : {train_score:.4f}")
    print(f"  Test Accuracy     : {test_score:.4f}")
    print(f"  ROC-AUC Score     : {auc:.4f}")
    print("\n── Classification Report ──────────────────────────")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))
    print("── Confusion Matrix ───────────────────────────────")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
    print("\n── Top 10 Feature Importances ─────────────────────")
    names       = get_feature_names()
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:10]
    for i, idx in enumerate(top_indices):
        print(f"  {i+1:2d}. {names[idx]:<30s} {importances[idx]:.4f}")
    print("="*60 + "\n")


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_model(model, path: str = MODEL_PATH):
    """Serialize model + metadata to disk."""
    names       = get_feature_names()
    importances = dict(zip(names, model.feature_importances_.tolist()))

    payload = {
        "model":        model,
        "model_name":   MODEL_NAME,
        "feature_names": names,
        "importances":  importances,
        "version":      "2.0.0",
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"✓ Model saved → {path}")


def load_model(path: str = MODEL_PATH):
    """Load model payload from disk. Returns dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at '{path}'.\n"
            "Run: python train.py  (or python model.py --train)"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(url: str, model_payload: dict) -> dict:
    """
    Predict phishing probability for a URL.

    Returns:
        {
            score (float 0–1): phishing probability,
            verdict (str): PHISHING / SUSPICIOUS / LOW RISK / SAFE,
            confidence (float 0–1),
            features (dict),
            importances (dict),
            model_name (str),
            reasons (list[str]),
        }
    """
    model  = model_payload["model"]
    names  = model_payload["feature_names"]
    imps   = model_payload.get("importances", {})

    features = extract_features(url)
    vec      = features_to_vector(features)
    X        = np.array([vec])

    proba    = model.predict_proba(X)[0]   # [P(legit), P(phish)]
    phish_p  = float(proba[1])
    score    = round(phish_p, 4)

    # Determine verdict
    score_pct = score * 100
    if score_pct >= 70:
        verdict = "PHISHING"
    elif score_pct >= 40:
        verdict = "SUSPICIOUS"
    elif score_pct >= 20:
        verdict = "LOW RISK"
    else:
        verdict = "SAFE"

    # Generate reasons from high-weight features
    reasons = _build_reasons(features, imps)

    return {
        "score":       score,
        "score_pct":   int(score_pct),
        "verdict":     verdict,
        "probability": phish_p,
        "confidence":  round(max(proba), 4),
        "features":    features,
        "importances": imps,
        "model_name":  model_payload.get("model_name", "RandomForest"),
        "reasons":     reasons,
        "version":     model_payload.get("version", "2.0.0"),
    }


def _build_reasons(features: dict, importances: dict) -> list:
    """Generate human-readable reasons from feature values."""
    reasons = []
    f = features

    if f.get("has_ip"):           reasons.append("IP address used instead of domain name")
    if f.get("has_at_symbol"):    reasons.append("Contains @ symbol — URL redirection trick")
    if f.get("brand_in_subdomain"): reasons.append("Trusted brand name spoofed in subdomain")
    if f.get("brand_in_domain"):  reasons.append("Brand name spoofed in domain")
    if f.get("has_suspicious_tld"): reasons.append("Suspicious top-level domain detected")
    if f.get("has_hex_encoding"): reasons.append("Hexadecimal encoding in URL")
    if f.get("has_double_slash"): reasons.append("Double slash redirect trick detected")
    if f.get("has_port"):         reasons.append("Non-standard port detected")
    if f.get("is_url_shortened"): reasons.append("URL shortening service used")
    if f.get("has_exe_extension"): reasons.append("Executable file extension in path")

    kw = f.get("keywords_in_url", 0)
    if kw >= 3:
        reasons.append(f"{kw} phishing keywords found in URL")
    elif kw > 0:
        reasons.append(f"Phishing keyword(s) detected ({kw})")

    ent = f.get("domain_entropy", 0)
    if ent > 3.8:
        reasons.append(f"High domain entropy ({ent:.2f}) — randomized characters")

    if f.get("url_length", 0) > 75:
        reasons.append(f"Unusually long URL ({f['url_length']} chars)")

    if not f.get("has_https"):
        reasons.append("No HTTPS — unencrypted connection")

    if f.get("subdomain_depth", 0) > 3:
        reasons.append(f"Deep subdomain nesting ({f['subdomain_depth']} levels)")

    if f.get("is_trusted_domain") and not reasons:
        reasons.append("Domain matches trusted whitelist")

    return reasons


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PhishGuard AI Model Trainer")
    parser.add_argument("--train",  action="store_true", help="Train and save model")
    parser.add_argument("--eval",   action="store_true", help="Train and evaluate")
    parser.add_argument("--predict", type=str,           help="Predict a single URL")
    args = parser.parse_args()

    if args.predict:
        payload = load_model()
        result  = predict(args.predict, payload)
        print(f"\nURL     : {args.predict}")
        print(f"Verdict : {result['verdict']}")
        print(f"Score   : {result['score_pct']}%")
        print(f"Reasons : {result['reasons']}")

    else:
        print("Building dataset…")
        X, y = build_dataset()
        print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
        print(f"  Phishing   : {y.sum()}")
        print(f"  Legitimate : {(y==0).sum()}")

        evaluate = args.eval or True  # always evaluate when run directly
        print("\nTraining model…")
        model = train_model(X, y, evaluate=evaluate)
        save_model(model)
