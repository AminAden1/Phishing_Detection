# technique2.py

import os
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from playwright.sync_api import sync_playwright

from common import log, save_html, safe_filename, knowphish_predict, HTML_DIR, SCREENSHOT_DIR


def run_technique2(urls_csv="urls.csv", n_samples=20):
    df = pd.read_csv(urls_csv)
    df = df.sample(min(n_samples, len(df)), random_state=42)

    results = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()

        for _, row in df.iterrows():
            url = row["url"]
            label = 1 if row["label"] == "phish" else 0

            log(f"[T2] {url}")

            try:
                page.goto(url, timeout=20000, wait_until="domcontentloaded")
                page.wait_for_timeout(2000)
                html = page.content()
                final_url = page.url

                # Save html
                html_path = save_html(final_url + "_t2", html)

                # Save screenshot
                screenshot_path = os.path.join(
                    SCREENSHOT_DIR,
                    safe_filename(final_url) + "_t2.png"
                )
                page.screenshot(path=screenshot_path, full_page=True)

                # Predict
                prob = knowphish_predict(html_path, screenshot_path)
                y_pred = 1 if prob >= 0.5 else 0

                results.append({
                    "url": url,
                    "label": label,
                    "y_pred": y_pred,
                    "score": prob
                })

            except Exception as e:
                log(f"Client fetch failed: {e}")
                continue

        browser.close()

    df_out = pd.DataFrame(results)
    df_out.to_csv("technique2_results.csv", index=False)
    log("Saved technique2_results.csv")

    # Visualization (score distribution)
    plt.hist(df_out["score"], bins=15)
    plt.title("Technique 2 Score Histogram")
    plt.savefig("t2_similarity_hist.png")
    plt.close()
    print("Saved visualization t2_similarity_hist.png")
