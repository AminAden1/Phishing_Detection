# train_knowphish_model.py

import os
import re
import pickle
import hashlib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from common import HTML_DIR, SCREENSHOT_DIR, log, url_hash


def extract_features(html_text: str):
    if html_text is None:
        return [0, 0]
    text_len = len(html_text)
    num_tags = html_text.count("<")
    return [text_len, num_tags]


def train_model(urls_csv="urls.csv"):
    df = pd.read_csv(urls_csv)

    X = []
    y = []

    log("[+] Loading training data...")

    for _, row in df.iterrows():
        url = row["url"]
        label = 1 if row["label"] == "phish" else 0

        h = url_hash(url)

        html_path = os.path.join(HTML_DIR, h + ".html")
        if not os.path.exists(html_path):
            continue

        try:
            with open(html_path, "r", encoding="utf-8") as f:
                html = f.read()
        except:
            continue

        feats = extract_features(html)
        X.append(feats)
        y.append(label)

    if len(X) == 0:
        print("[-] No matching HTML/screenshots found!")
        return

    log("[+] Training RandomForest classifier...")
    model = RandomForestClassifier(n_estimators=150)
    model.fit(X, y)

    preds = model.predict(X)
    print("\n=== MODEL PERFORMANCE ===")
    print("Accuracy:", accuracy_score(y, preds))
    print("F1 Score:", f1_score(y, preds))

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    log("[+] Saved model.pkl")


if __name__ == "__main__":
    train_model()
