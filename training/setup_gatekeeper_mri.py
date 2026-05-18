"""
Copy your existing MRI images into the gatekeeper data folders.
Run this from inside the endometriosis-ai folder.

Run:
    python training\setup_gatekeeper_mri.py
"""

import os
import shutil
from pathlib import Path

# Source — your existing converted MRI PNGs
SOURCES = [
    Path("data/train/positive"),
    Path("data/train/negative"),
    Path("data/val/positive"),
    Path("data/val/negative"),
]

TRAIN_OUT = Path("gatekeeper_data/train/mri")
VAL_OUT   = Path("gatekeeper_data/val/mri")
TRAIN_OUT.mkdir(parents=True, exist_ok=True)
VAL_OUT.mkdir(parents=True, exist_ok=True)

count = 0
for src in SOURCES:
    is_val = "val" in str(src)
    dest = VAL_OUT if is_val else TRAIN_OUT
    for f in src.glob("*.png"):
        shutil.copy2(f, dest / f"{src.parent.name}_{src.name}_{f.name}")
        print(f"  Copied {f.name} → {'val' if is_val else 'train'}/mri")
        count += 1

print(f"\nDone! Copied {count} MRI images.")
print(f"  Train MRI : {len(list(TRAIN_OUT.iterdir()))} images")
print(f"  Val MRI   : {len(list(VAL_OUT.iterdir()))} images")
