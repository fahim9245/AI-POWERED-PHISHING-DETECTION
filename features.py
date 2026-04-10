"""
features.py — PhishGuard AI Feature Extraction Engine
======================================================
Extracts 30+ numerical features from a URL for ML classification.
Used by both model.py (training) and app.py (inference).
"""

import re
import math
import urllib.parse
from typing import Dict, Any


# ── Constants ────────────────────────────────────────────────────────────────

SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "account", "secure", "update", "confirm",
    "banking", "paypal", "amazon", "google", "microsoft", "apple", "netflix",
    "password", "credential", "urgent", "suspended", "limited", "click",
    "free", "winner", "prize", "bitcoin", "wallet", "recover", "validation",
]

TRUSTED_DOMAINS = [
    "google.com", "microsoft.com", "apple.com", "amazon.com", "paypal.com",
    "facebook.com", "twitter.com", "instagram.com", "linkedin.com", "github.com",
    "youtube.com", "wikipedia.org", "stackoverflow.com", "reddit.com",
]

SUSPICIOUS_TLDS = [
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top",
    ".work", ".click", ".link", ".download", ".zip", ".review",
    ".pw", ".cc", ".su", ".ws",
]

BRAND_NAMES = [
    "paypal", "apple", "google", "microsoft", "amazon", "netflix",
    "bankofamerica", "citibank", "wellsfargo", "ebay", "dropbox",
]

SHORTENING_SERVICES = [
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly",
    "is.gd", "buff.ly", "adf.ly", "cutt.ly",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def shannon_entropy(s: str) -> float:
    """Calculate Shannon entropy of a string."""
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    total = len(s)
    return -sum((n / total) * math.log2(n / total) for n in freq.values())


def safe_parse(url: str) -> urllib.parse.ParseResult | None:
    """Safely parse a URL, adding scheme if missing."""
    try:
        if "://" not in url:
            url = "http://" + url
        return urllib.parse.urlparse(url)
    except Exception:
        return None


def get_domain(parsed: urllib.parse.ParseResult) -> str:
    """Return cleaned domain without www prefix."""
    return re.sub(r"^www\.", "", parsed.netloc.lower()).split(":")[0]


# ── Feature Extraction ────────────────────────────────────────────────────────

def extract_features(url: str) -> Dict[str, Any]:
    """
    Extract 30+ features from a URL.

    Returns a dict with all feature values.
    Feature names match those used during model training.
    """
    parsed = safe_parse(url)
    if parsed is None:
        return _empty_features(url)

    domain = get_domain(parsed)
    full   = url.lower()
    path   = parsed.path.lower()
    query  = parsed.query.lower()

    # Domain parts
    parts      = domain.split(".")
    subdomain  = ".".join(parts[:-2]) if len(parts) > 2 else ""
    tld        = "." + parts[-1] if parts else ""
    sld        = parts[-2] if len(parts) >= 2 else domain   # second-level domain

    # Keyword counts
    kw_in_url    = sum(1 for k in SUSPICIOUS_KEYWORDS if k in full)
    kw_in_domain = sum(1 for k in SUSPICIOUS_KEYWORDS if k in domain)
    kw_in_path   = sum(1 for k in SUSPICIOUS_KEYWORDS if k in path)

    # Brand spoofing
    brand_in_subdomain = any(b in subdomain for b in BRAND_NAMES) and \
                         not any(domain == t or domain.endswith("." + t) for t in TRUSTED_DOMAINS)
    brand_in_domain    = any(b in sld for b in BRAND_NAMES) and \
                         not any(domain == t or domain.endswith("." + t) for t in TRUSTED_DOMAINS)

    # Query params
    query_params = urllib.parse.parse_qs(parsed.query)
    num_params   = len(query_params)

    # Counts in domain
    digits_in_domain = sum(1 for c in domain if c.isdigit())
    hyphens_in_domain = domain.count("-")
    dots_in_domain    = domain.count(".")

    features: Dict[str, Any] = {
        # Length features
        "url_length":        len(url),
        "domain_length":     len(domain),
        "path_length":       len(parsed.path),
        "query_length":      len(parsed.query),
        "num_query_params":  num_params,

        # Protocol features
        "has_https":         int(parsed.scheme == "https"),
        "has_http":          int(parsed.scheme == "http"),

        # Domain structure
        "subdomain_depth":   max(0, len(parts) - 2),
        "domain_entropy":    round(shannon_entropy(domain), 4),
        "sld_entropy":       round(shannon_entropy(sld), 4),
        "digits_in_domain":  digits_in_domain,
        "hyphens_in_domain": hyphens_in_domain,
        "dots_in_domain":    dots_in_domain,

        # IP & Port
        "has_ip":            int(bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", parsed.netloc.split(":")[0]))),
        "has_port":          int(bool(parsed.port)),

        # Suspicious indicators
        "has_at_symbol":     int("@" in url),
        "has_double_slash":  int("//" in url[8:]),
        "has_hex_encoding":  int(bool(re.search(r"%[0-9a-fA-F]{2}", url))),
        "has_suspicious_tld": int(any(domain.endswith(t) for t in SUSPICIOUS_TLDS)),
        "is_url_shortened":  int(any(domain == s for s in SHORTENING_SERVICES)),
        "has_redirect_param": int(any(k in query for k in ["redirect", "url=", "goto=", "return="])),

        # Keyword features
        "keywords_in_url":    kw_in_url,
        "keywords_in_domain": kw_in_domain,
        "keywords_in_path":   kw_in_path,

        # Brand spoofing
        "brand_in_subdomain": int(brand_in_subdomain),
        "brand_in_domain":    int(brand_in_domain),

        # Trust signal
        "is_trusted_domain":  int(any(domain == t or domain.endswith("." + t) for t in TRUSTED_DOMAINS)),

        # Path features
        "path_depth":         len([p for p in parsed.path.split("/") if p]),
        "has_exe_extension":  int(bool(re.search(r"\.(exe|zip|rar|bat|cmd|sh|php|asp)$", path))),

        # Miscellaneous
        "num_special_chars":  len(re.findall(r"[^a-zA-Z0-9.\-_/:]", url)),
        "url_entropy":        round(shannon_entropy(url), 4),
    }

    return features


def _empty_features(url: str) -> Dict[str, Any]:
    """Return zero-filled features for unparseable URLs."""
    keys = [
        "url_length", "domain_length", "path_length", "query_length",
        "num_query_params", "has_https", "has_http", "subdomain_depth",
        "domain_entropy", "sld_entropy", "digits_in_domain", "hyphens_in_domain",
        "dots_in_domain", "has_ip", "has_port", "has_at_symbol", "has_double_slash",
        "has_hex_encoding", "has_suspicious_tld", "is_url_shortened",
        "has_redirect_param", "keywords_in_url", "keywords_in_domain",
        "keywords_in_path", "brand_in_subdomain", "brand_in_domain",
        "is_trusted_domain", "path_depth", "has_exe_extension",
        "num_special_chars", "url_entropy",
    ]
    return {k: 0 for k in keys}


def features_to_vector(features: Dict[str, Any]) -> list:
    """Convert features dict to ordered list for ML model input."""
    ordered_keys = [
        "url_length", "domain_length", "path_length", "query_length",
        "num_query_params", "has_https", "has_http", "subdomain_depth",
        "domain_entropy", "sld_entropy", "digits_in_domain", "hyphens_in_domain",
        "dots_in_domain", "has_ip", "has_port", "has_at_symbol", "has_double_slash",
        "has_hex_encoding", "has_suspicious_tld", "is_url_shortened",
        "has_redirect_param", "keywords_in_url", "keywords_in_domain",
        "keywords_in_path", "brand_in_subdomain", "brand_in_domain",
        "is_trusted_domain", "path_depth", "has_exe_extension",
        "num_special_chars", "url_entropy",
    ]
    return [features.get(k, 0) for k in ordered_keys]


def get_feature_names() -> list:
    """Return ordered list of feature names (must match features_to_vector)."""
    return [
        "url_length", "domain_length", "path_length", "query_length",
        "num_query_params", "has_https", "has_http", "subdomain_depth",
        "domain_entropy", "sld_entropy", "digits_in_domain", "hyphens_in_domain",
        "dots_in_domain", "has_ip", "has_port", "has_at_symbol", "has_double_slash",
        "has_hex_encoding", "has_suspicious_tld", "is_url_shortened",
        "has_redirect_param", "keywords_in_url", "keywords_in_domain",
        "keywords_in_path", "brand_in_subdomain", "brand_in_domain",
        "is_trusted_domain", "path_depth", "has_exe_extension",
        "num_special_chars", "url_entropy",
    ]


# ── CLI Testing ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_urls = [
        "https://google.com",
        "http://paypa1-secure-login.tk/account/verify",
        "http://192.168.0.1/login?ref=paypal",
        "https://amazon-account-suspended.ml/confirm-identity",
        "http://secure.bank0famerica.update-credentials.xyz/verify",
    ]

    for url in test_urls:
        feats = extract_features(url)
        vec   = features_to_vector(feats)
        print(f"\nURL: {url}")
        print(f"  Features: {len(feats)}")
        print(f"  Suspicious TLD: {feats['has_suspicious_tld']}")
        print(f"  Brand in subdomain: {feats['brand_in_subdomain']}")
        print(f"  Domain entropy: {feats['domain_entropy']}")
        print(f"  Keywords in URL: {feats['keywords_in_url']}")
        print(f"  Is trusted: {feats['is_trusted_domain']}")
