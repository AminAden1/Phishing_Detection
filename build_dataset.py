# build_dataset.py

import io
import asyncio
import requests
import pandas as pd
from tqdm import tqdm
from playwright.async_api import async_playwright

OUTPUT_FILE = "urls.csv"

PHISH_N = 2500
LEGIT_N = 1500

CONCURRENCY = 15
TIMEOUT_MS = 8000
HTML_MIN = 80  # require at least some HTML text

PHISHTANK = "https://data.phishtank.com/data/online-valid.csv"
OPENPHISH = "https://openphish.com/feed.txt"
URLHAUS = "https://urlhaus.abuse.ch/downloads/text_online/"
CLOUDFLARE_TOP = "https://radar.cloudflare.com/domains/top-1000.csv"

# Some legit domains we want to avoid (JS-heavy / login walls)
AVOID_LEGIT = [
    "google.", "instagram.", "youtube.", "tiktok.", "facebook.",
    "apple.com", "icloud.", "twitter.", "x.com", "netflix."
]


def load_phishing_urls():
    urls = set()

    print("[+] Loading PhishTank...")
    try:
        r = requests.get(PHISHTANK)
        df = pd.read_csv(io.StringIO(r.text))
        urls.update(df["url"].dropna().tolist())
    except Exception as e:
        print("[!] PhishTank error:", e)

    print("[+] Loading OpenPhish...")
    try:
        op = requests.get(OPENPHISH).text.splitlines()
        urls.update([u.strip() for u in op if u.startswith("http")])
    except Exception as e:
        print("[!] OpenPhish error:", e)

    print("[+] Loading URLHaus...")
    try:
        uh = requests.get(URLHAUS).text.splitlines()
        urls.update([u.strip() for u in uh if u.startswith("http")])
    except Exception as e:
        print("[!] URLHaus error:", e)

    urls = list(urls)
    print(f"[+] Total raw phishing URLs: {len(urls)}")
    return urls


def load_legit_urls():
    print("[+] Loading Cloudflare Top 1000 legit domains...")
    try:
        r = requests.get(CLOUDFLARE_TOP)
        df = pd.read_csv(io.StringIO(r.text))
        domains = df["domain"].dropna().tolist()
        urls = ["https://" + d for d in domains]

        # Filter out JS-heavy / login-wall sites
        filtered = []
        for u in urls:
            if any(bad in u for bad in AVOID_LEGIT):
                continue
            filtered.append(u)

        print(f"[+] Legit URLs after filter: {len(filtered)}")
        return filtered
    except Exception as e:
        print("[!] Cloudflare error:", e)
        return [
            "https://bbc.com",
            "https://cnn.com",
            "https://w3schools.com",
            "https://stackoverflow.com",
            "https://craigslist.org",
            "https://wordpress.com",
            "https://github.com",
        ]


async def check_url(pw, url):
    try:
        browser = await pw.chromium.launch()
        page = await browser.new_page()
        await page.goto(url, timeout=TIMEOUT_MS, wait_until="domcontentloaded")
        html = await page.content()
        await browser.close()
        if len(html.strip()) < HTML_MIN:
            return False
        return True
    except Exception:
        return False


async def filter_urls(urls, label, max_needed):
    selected = []
    sem = asyncio.Semaphore(CONCURRENCY)

    async with async_playwright() as pw:

        async def worker(u):
            nonlocal selected
            async with sem:
                if len(selected) >= max_needed:
                    return
                ok = await check_url(pw, u)
                if ok:
                    selected.append((u, label))

        tasks = [asyncio.create_task(worker(u)) for u in urls]

        for t in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            await t
            if len(selected) >= max_needed:
                break

    print(f"[+] {label}: {len(selected)} working URLs")
    return selected


async def main():
    phish_all = load_phishing_urls()
    legit_all = load_legit_urls()

    print("[+] Filtering phishing URLs...")
    good_phish = await filter_urls(phish_all, "phish", PHISH_N)

    print("[+] Filtering legit URLs...")
    good_legit = await filter_urls(legit_all, "legit", LEGIT_N)

    df = pd.DataFrame(good_phish + good_legit, columns=["url", "label"])
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"[+] Saved dataset to {OUTPUT_FILE}")
    print(f"[+] Total URLs: {len(df)}")


if __name__ == "__main__":
    asyncio.run(main())
