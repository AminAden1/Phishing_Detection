# train_knowphish_model.py

import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from common import HTML_DIR, safe_filename, log, _clean_html_text

MIN_TEXT_LEN = 80  # must have some content


print("[+] [+] Loading URL labels...")
df = pd.read_csv("urls.csv")

texts = []
labels = []

print("[+] [+] Collecting HTML/text for each URL...")

for _, row in df.iterrows():
    url = row["url"]
    label = 1 if row["label"] == "phish" else 0

    base = safe_filename(url)

    candidate_files = [
        os.path.join(HTML_DIR, f"{base}.html"),
        os.path.join(HTML_DIR, f"{base}_pert.html"),
        os.path.join(HTML_DIR, f"{base}_t2.html"),
    ]

    for path in candidate_files:
        if not os.path.exists(path):
            continue

        try:
            raw = open(path, "r", encoding="utf-8", errors="ignore").read()
        except:
            continue

        text = _clean_html_text(raw)
        if not text or len(text) < MIN_TEXT_LEN:
            continue

        texts.append(text)
        labels.append(label)

print(f"[+] Total usable HTML pages: {len(texts)}")

if len(texts) < 20:
    print("[-] Not enough training data. Run technique1/2 on more URLs.")
    raise SystemExit

print("[+] [+] Vectorizing text...")
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(texts)

print("[+] [+] Training RandomForest classifier...")
clf = RandomForestClassifier(n_estimators=250, random_state=42)
clf.fit(X, labels)

pred = clf.predict(X)

print("\n=== MODEL PERFORMANCE (on training set) ===")
print("Accuracy:", accuracy_score(labels, pred))
print("F1 Score:", f1_score(labels, pred))

with open("model.pkl", "wb") as f:
    pickle.dump({"model": clf, "vectorizer": vectorizer}, f)

print("[+] [+] Saved model.pkl")
