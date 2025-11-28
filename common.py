# common.py

import os
import hashlib
import pickle
import random

HTML_DIR = "data/html"
SCREENSHOT_DIR = "data/screenshots"

os.makedirs(HTML_DIR, exist_ok=True)
os.makedirs(SCREENSHOT_DIR, exist_ok=True)


def log(msg: str):
    print(f"[+] {msg}")


def url_hash(url: str) -> str:
    """
    Creates a short, consistent hash filename from a URL.
    This matches what is already in your data/html folder.
    """
    return hashlib.md5(url.encode("utf-8")).hexdigest()


def save_html(url: str, html: str) -> str:
    """
    Save HTML content to hashed filename.
    """
    filename = url_hash(url) + ".html"
    path = os.path.join(HTML_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def safe_filename(url: str) -> str:
    """
    Alias to remain compatible — everything uses the md5 hash.
    """
    return url_hash(url)


def knowphish_predict(html_path: str, screenshot_path: str) -> float:
    """
    Loads trained model.pkl if available.
    Otherwise returns RANDOM scores.
    """
    model_path = "model.pkl"

    if not os.path.exists(model_path):
        log("Model not found: using RANDOM predictions until model is trained.")
        return random.random()

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except:
        log("Failed to load model.pkl — using random predictions.")
        return random.random()

    # Load HTML text as features
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
    except:
        return random.random()

    text_len = len(html)
    num_tags = html.count("<")

    features = [[text_len, num_tags]]
    prob = model.predict_proba(features)[0][1]
    return float(prob)
