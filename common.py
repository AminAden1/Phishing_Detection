# common.py

import os
import hashlib
import random
import pickle
import re

DATA_DIR = "data"
HTML_DIR = os.path.join(DATA_DIR, "html")
SCREENSHOT_DIR = os.path.join(DATA_DIR, "screens")

os.makedirs(HTML_DIR, exist_ok=True)
os.makedirs(SCREENSHOT_DIR, exist_ok=True)


def log(msg: str):
    print(f"[+] {msg}")


def safe_filename(url: str) -> str:
    """
    Stable hash-based filename for a URL.
    This MUST match everywhere (techniques + training).
    """
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return h


def save_html(url: str, html: str, suffix: str = "") -> str:
    """
    Save HTML content for a URL.
    suffix: "", "_pert", "_t2", etc.
    """
    base = safe_filename(url)
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix
    path = os.path.join(HTML_DIR, f"{base}{suffix}.html")
    with open(path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(html)
    return path


MODEL_PATH = "model.pkl"
_cached_model = None


def _load_model():
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    if not os.path.exists(MODEL_PATH):
        log("Model not found: using RANDOM predictions until trained.")
        _cached_model = None
        return None

    with open(MODEL_PATH, "rb") as f:
        _cached_model = pickle.load(f)  # {"model": clf, "vectorizer": vec}
    return _cached_model


def _clean_html_text(raw_html: str) -> str:
    """
    Very lightweight cleaning used BOTH in training and inference.
    No BeautifulSoup needed; regex-based to keep it simple & fast.
    """
    # strip script/style
    raw_html = re.sub(r"<script.*?</script>", " ", raw_html, flags=re.DOTALL | re.IGNORECASE)
    raw_html = re.sub(r"<style.*?</style>", " ", raw_html, flags=re.DOTALL | re.IGNORECASE)
    # remove tags
    raw_html = re.sub(r"<[^>]+>", " ", raw_html)
    # collapse spaces
    raw_html = re.sub(r"\s+", " ", raw_html)
    return raw_html.strip()


def knowphish_predict(html_path: str, screenshot_path: str = None) -> float:
    """
    Predict phishing probability from an HTML file.
    If model is missing, returns RANDOM probability in [0,1].
    """
    mdl = _load_model()
    if mdl is None:
        return random.random()  # random baseline until trained

    try:
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
    except:
        return 0.5

    text = _clean_html_text(raw)
    if not text:
        return 0.5

    vec = mdl["vectorizer"]
    clf = mdl["model"]

    X = vec.transform([text])
    proba = clf.predict_proba(X)[0][1]  # phishing probability
    return float(proba)
