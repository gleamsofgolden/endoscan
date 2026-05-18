"""
Download random non-MRI images for gatekeeper training.
Uses Unsplash's free API (no key needed for small usage).

Run:
    python training\download_not_mri.py
"""

import os
import urllib.request
import time
from pathlib import Path

# Categories of clearly non-medical images
QUERIES = [
    "cat", "dog", "car", "food", "city",
    "nature", "people", "building", "flower", "laptop"
]

SAVE_DIR = Path("gatekeeper_data/train/not_mri")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

VAL_DIR = Path("gatekeeper_data/val/not_mri")
VAL_DIR.mkdir(parents=True, exist_ok=True)

COUNT = 0
TARGET = 200  # total images to download

print(f"Downloading {TARGET} non-MRI images...\n")

for query in QUERIES:
    per_query = TARGET // len(QUERIES)
    for i in range(per_query):
        # Unsplash source — returns a random image for the query
        url = f"https://source.unsplash.com/224x224/?{query}&sig={COUNT}"
        filename = f"{query}_{i:03d}.jpg"

        # 80% train, 20% val
        save_path = SAVE_DIR / filename if COUNT % 5 != 0 else VAL_DIR / filename

        try:
            urllib.request.urlretrieve(url, save_path)
            print(f"  Downloaded {filename}")
            COUNT += 1
            time.sleep(0.3)  # be polite to the server
        except Exception as e:
            print(f"  Skipped {filename}: {e}")

print(f"\nDone! Downloaded {COUNT} images.")
print(f"  Train: {len(list(SAVE_DIR.iterdir()))} images → {SAVE_DIR}")
print(f"  Val  : {len(list(VAL_DIR.iterdir()))} images → {VAL_DIR}")
