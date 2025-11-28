import os
import pandas as pd
import random
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from sklearn.metrics import f1_score
from playwright.sync_api import sync_playwright

from common import (
    log, save_html, safe_filename, knowphish_predict
)

HTML_DIR = "data/html"
SCREENSHOT_DIR = "data/screenshots"

os.makedirs(HTML_DIR, exist_ok=True)
os.makedirs(SCREENSHOT_DIR, exist_ok=True)


def random_capitalization(text, p=0.15):
    out = []
    for c in text:
        if c.isalpha() and random.random() < p:
            out.append(c.upper())
        else:
            out.append(c)
    return "".join(out)


def inject_stopwords(text):
    sw = ["just", "only", "really", "very", "kind of"]
    words = text.split()
    for i in range(0, len(words), 7):
        words.insert(i, random.choice(sw))
    return " ".join(words)


def benign_paraphrase(text):
    repl = {
        "log in": "sign in",
        "verify": "confirm",
        "account": "user account",
    }
    for a, b in repl.items():
        text = text.replace(a, b).replace(a.capitalize(), b.capitalize())
    return text


def perturb_html(html):
    soup = BeautifulSoup(html, "html.parser")

    for node in soup.find_all(string=True):
        t = str(node)
        if not t.strip():
            continue

        t = random_capitalization(t)
        t = inject_stopwords(t)
        t = benign_paraphrase(t)

        node.replace_with(t)

    head = soup.find("head")
    if head:
        meta = soup.new_tag("meta")
        meta.attrs["name"] = "x-benign-variant"
        meta.attrs["content"] = str(random.randint(0, 10000))
        head.append(meta)

    return str(soup)


def run_technique1(urls_csv="urls.csv", n_samples=20):
    df = pd.read_csv(urls_csv)
    df = df.sample(min(n_samples, len(df)), random_state=42)

    results = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()

        for _, row in df.iterrows():
            url = row["url"]
            label = 1 if row["label"] == "phish" else 0

            log(f"[T1] Fetching: {url}")

            try:
                page.goto(url, timeout=15000, wait_until="domcontentloaded")
                page.wait_for_timeout(4000)
                html = page.content()
                final_url = page.url

                html_path = os.path.join(
                    HTML_DIR, safe_filename(final_url) + ".html"
                )
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html)

                screenshot_path = os.path.join(
                    SCREENSHOT_DIR, safe_filename(final_url) + "_t1.png"
                )
                page.screenshot(path=screenshot_path, full_page=True)

                p_base = knowphish_predict(html_path, screenshot_path)
                y_base = 1 if p_base >= 0.5 else 0

                perturbed_html = perturb_html(html)
                pert_path = os.path.join(
                    HTML_DIR, safe_filename(final_url) + "_pert.html"
                )
                with open(pert_path, "w", encoding="utf-8") as f:
                    f.write(perturbed_html)

                p_pert = knowphish_predict(pert_path, screenshot_path)
                y_pert = 1 if p_pert >= 0.5 else 0

                results.append({
                    "url": url,
                    "label": label,
                    "p_base": p_base,
                    "y_base": y_base,
                    "p_pert": p_pert,
                    "y_pert": y_pert
                })

            except Exception as e:
                log(f"ERROR: {e}")
                continue

        browser.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv("technique1_results.csv", index=False)
    log("Saved technique1_results.csv")

    evaluate_technique1(results_df)


def evaluate_technique1(df: pd.DataFrame):
    if df.empty:
        print("No results.")
        return

    y = df["label"]
    base = df["y_base"]
    pert = df["y_pert"]

    print("\n=== TECHNIQUE 1 METRICS ===")
    print("Baseline F1:", f1_score(y, base))
    print("Perturbed F1:", f1_score(y, pert))

    plt.bar(["Base", "Perturbed"],
            [f1_score(y, base), f1_score(y, pert)])
    plt.title("F1 Score Drop After Perturbation")
    plt.savefig("t1_f1_drop.png")
    plt.close()
    print("Saved visualization t1_f1_drop.png")


if __name__ == "__main__":
    run_technique1()
