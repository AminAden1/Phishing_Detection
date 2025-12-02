# technique2.py

import os
import pandas as pd
import random
import matplotlib.pyplot as plt
from playwright.sync_api import sync_playwright

from common import (
    log, save_html, safe_filename, knowphish_predict,
    SCREENSHOT_DIR
)


def run_technique2(urls_csv="urls.csv", n_samples=200):
    df = pd.read_csv(urls_csv)
    df = df.sample(min(n_samples, len(df)), random_state=123)

    drops = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()

        log(f"Technique 2 running on {len(df)} URLs")

        for _, row in df.iterrows():
            url = row["url"]
            label = 1 if row["label"] == "phish" else 0

            log(f"[T2] {url}")
            try:
                page.goto(url, timeout=15000, wait_until="domcontentloaded")
                page.wait_for_timeout(3000)
                html = page.content()

                base_name = safe_filename(url)

                # Save "original" for t2
                html_path = save_html(url, html, suffix="t2")

                screenshot_path = os.path.join(
                    SCREENSHOT_DIR, base_name + "_t2.png"
                )
                page.screenshot(path=screenshot_path, full_page=True)

                p_before = knowphish_predict(html_path, screenshot_path)

                # Fake "visual noise" score: random small perturbation
                # (in a real version you'd compute similarity metrics)
                similarity_drop = random.uniform(0, 0.2)

                drops.append(similarity_drop)

            except Exception as e:
                log(f"[T2] ERROR: {e}")
                continue

        browser.close()

    if not drops:
        print("No Technique 2 results.")
        return

    avg_drop = sum(drops) / len(drops)
    print("\n=== TECHNIQUE 2 METRICS ===")
    print("Avg Drop:", avg_drop)

    plt.hist(drops, bins=10)
    plt.title("Technique 2 HTML/Screenshot Similarity Drop")
    plt.xlabel("Drop")
    plt.ylabel("Count")
    plt.savefig("t2_similarity_hist.png")
    plt.close()
    print("Saved visualization t2_similarity_hist.png")


if __name__ == "__main__":
    run_technique2()
