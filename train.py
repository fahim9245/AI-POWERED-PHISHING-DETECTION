"""
train.py — PhishGuard AI Training Pipeline
==========================================
Generates a synthetic + real dataset, trains the model,
evaluates it, and saves to disk.

Usage:
    python train.py           # Train with built-in dataset
    python train.py --augment # Augment with synthetic samples
    python train.py --csv path/to/urls.csv  # Use custom CSV dataset

CSV format expected:
    url,label
    https://google.com,0
    http://phish.tk/login,1
"""

import os
import sys
import random
import argparse
import numpy as np

from features import extract_features, features_to_vector, get_feature_names
from model import train_model, save_model, LABELED_URLS, MODEL_PATH


# ── Synthetic Data Generator ─────────────────────────────────────────────────

PHISHING_PATTERNS = [
    "http://{brand}-secure-login.{tld}/account/verify",
    "http://192.168.{r}.{r}/login?ref={brand}",
    "https://{brand}-account-suspended.{tld}/confirm-identity",
    "http://secure.{brand}update-credentials.xyz/verify",
    "http://{brand}.com.phish-site.{tld}/login",
    "https://{brand}-billing.update-now.{tld}/payment",
    "http://login.{brand}.bank-secure-update.{tld}/auth",
    "http://confirm-identity.{brand}.phish.{tld}/secure",
    "http://urgently-verify-your-{brand}-account.xyz/confirm",
]

LEGIT_PATTERNS = [
    "https://www.{brand}.com/products/{product}",
    "https://{brand}.com/login",
    "https://docs.{brand}.com/en/api",
    "https://support.{brand}.com/help?id={r}",
    "https://blog.{brand}.com/{year}/post",
    "https://{brand}.com/account/settings",
]

BRANDS = ["paypal", "amazon", "google", "microsoft", "apple", "netflix", "ebay", "facebook"]
PHISH_TLDS = ["tk", "ml", "ga", "cf", "gq", "xyz", "top"]
LEGIT_BRANDS = ["stripe", "shopify", "notion", "figma", "vercel", "railway", "supabase"]
PRODUCTS = ["shoes", "electronics", "books", "clothing", "software", "gaming"]


def _gen(pattern: str) -> str:
    brand   = random.choice(BRANDS)
    tld     = random.choice(PHISH_TLDS)
    product = random.choice(PRODUCTS)
    r       = random.randint(1, 254)
    year    = random.randint(2020, 2024)
    return pattern.format(brand=brand, tld=tld, product=product, r=r, year=year)


def generate_synthetic(n_phish: int = 100, n_legit: int = 100):
    """Generate synthetic labeled URL samples."""
    samples = []

    for _ in range(n_phish):
        pattern = random.choice(PHISHING_PATTERNS)
        samples.append((_gen(pattern), 1))

    for _ in range(n_legit):
        pattern = random.choice(LEGIT_PATTERNS)
        brand   = random.choice(LEGIT_BRANDS + BRANDS)
        product = random.choice(PRODUCTS)
        r       = random.randint(1000, 9999)
        year    = random.randint(2020, 2024)
        url     = pattern.format(brand=brand, product=product, r=r, year=year, tld="com")
        samples.append((url, 0))

    return samples


def load_csv_dataset(csv_path: str):
    """Load dataset from CSV file with columns: url, label"""
    import csv
    samples = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url   = row.get("url", "").strip()
            label = int(row.get("label", 0))
            if url:
                samples.append((url, label))
    print(f"  Loaded {len(samples)} samples from {csv_path}")
    return samples


def build_dataset_from_samples(samples):
    """Convert labeled (url, label) pairs to feature matrix."""
    X, y, skipped = [], [], 0
    for url, label in samples:
        try:
            feats = extract_features(url)
            vec   = features_to_vector(feats)
            X.append(vec)
            y.append(label)
        except Exception as e:
            skipped += 1

    if skipped:
        print(f"  ⚠ Skipped {skipped} URLs due to extraction errors")

    return np.array(X, dtype=float), np.array(y)


# ── Main Training Pipeline ────────────────────────────────────────────────────

def run_training(augment: bool = False, csv_path: str = None):
    print("\n" + "="*60)
    print("  PhishGuard AI — Training Pipeline")
    print("="*60)

    # Collect dataset
    all_samples = list(LABELED_URLS)
    print(f"\n[1/4] Base dataset: {len(all_samples)} samples")

    if csv_path:
        csv_samples = load_csv_dataset(csv_path)
        all_samples.extend(csv_samples)
        print(f"      After CSV merge: {len(all_samples)} samples")

    if augment:
        print("[2/4] Generating synthetic augmentation data…")
        synthetic = generate_synthetic(n_phish=150, n_legit=150)
        all_samples.extend(synthetic)
        print(f"      After augmentation: {len(all_samples)} samples")
    else:
        print("[2/4] Skipping augmentation (use --augment to enable)")

    # Feature extraction
    print("[3/4] Extracting features…")
    X, y = build_dataset_from_samples(all_samples)
    n_features = X.shape[1]
    n_phish    = int(y.sum())
    n_legit    = int((y == 0).sum())

    print(f"      Samples  : {len(X)}")
    print(f"      Features : {n_features}")
    print(f"      Phishing : {n_phish}")
    print(f"      Legit    : {n_legit}")
    print(f"      Class balance: {n_phish/len(X)*100:.1f}% phishing")

    # Train
    print("[4/4] Training Random Forest…")
    model = train_model(X, y, evaluate=True)
    save_model(model, MODEL_PATH)

    print("\n✅ Training complete!")
    print(f"   Model saved to: {MODEL_PATH}")
    print(f"   Run the server: python app.py\n")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PhishGuard AI Training Script")
    parser.add_argument("--augment",  action="store_true", help="Add synthetic training data")
    parser.add_argument("--csv",      type=str, default=None, help="Path to custom CSV dataset")
    args = parser.parse_args()

    run_training(augment=args.augment, csv_path=args.csv)
