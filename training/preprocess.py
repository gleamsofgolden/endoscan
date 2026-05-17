"""
Medical Image Preprocessor
============================
Handles DICOM (.dcm), standard images (PNG/JPG), and converts
them to RGB tensors suitable for the classifier.

Install deps:
    pip install pydicom Pillow numpy torch torchvision

Usage:
    from preprocess import load_medical_image, preprocess_for_model
    
    pil_img = load_medical_image("scan.dcm")          # or scan.png
    tensor  = preprocess_for_model(pil_img, size=224)
"""

import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms


# ── DICOM loading ─────────────────────────────────────────────────────────────

def load_dicom(path):
    """
    Load a DICOM file and return a normalised RGB PIL Image.
    
    Handles:
      - 8-bit and 16-bit pixel data
      - Windowing (if window center/width tags present)
      - Monochrome1/Monochrome2 photometric interpretation
    """
    try:
        import pydicom
    except ImportError:
        raise ImportError("Install pydicom:  pip install pydicom")

    ds = pydicom.dcmread(str(path))
    pixel_array = ds.pixel_array.astype(np.float32)

    # Apply DICOM windowing if available
    if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
        wc = float(ds.WindowCenter) if not isinstance(ds.WindowCenter, pydicom.multival.MultiValue) \
             else float(ds.WindowCenter[0])
        ww = float(ds.WindowWidth)  if not isinstance(ds.WindowWidth,  pydicom.multival.MultiValue) \
             else float(ds.WindowWidth[0])
        lo = wc - ww / 2
        hi = wc + ww / 2
        pixel_array = np.clip(pixel_array, lo, hi)
        pixel_array = (pixel_array - lo) / (hi - lo + 1e-8)
    else:
        # Fallback: min-max normalise
        pmin, pmax = pixel_array.min(), pixel_array.max()
        pixel_array = (pixel_array - pmin) / (pmax - pmin + 1e-8)

    # Invert if Monochrome1 (white = 0)
    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photometric == "MONOCHROME1":
        pixel_array = 1.0 - pixel_array

    # Convert to 8-bit
    pixel_array = (pixel_array * 255).astype(np.uint8)

    # Ensure 2D → RGB
    if pixel_array.ndim == 2:
        pil_img = Image.fromarray(pixel_array, mode="L").convert("RGB")
    elif pixel_array.ndim == 3 and pixel_array.shape[2] == 3:
        pil_img = Image.fromarray(pixel_array, mode="RGB")
    else:
        # Take first channel/slice
        pil_img = Image.fromarray(pixel_array[:, :, 0], mode="L").convert("RGB")

    return pil_img


def load_standard_image(path):
    """Load PNG/JPG/BMP as RGB PIL Image."""
    return Image.open(str(path)).convert("RGB")


def load_medical_image(path):
    """
    Auto-detect format and load as RGB PIL Image.
    Supports: .dcm (DICOM), .png, .jpg, .jpeg, .bmp, .tiff
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".dcm" or suffix == "":
        return load_dicom(path)
    else:
        return load_standard_image(path)


# ── Preprocessing for model ───────────────────────────────────────────────────

def preprocess_for_model(pil_image, size=224):
    """
    Resize + normalise PIL Image for inference.
    
    Returns: (1, 3, size, size) tensor ready for model.forward()
    """
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return transform(pil_image)


# ── Dataset preparation helpers ───────────────────────────────────────────────

def prepare_dataset_split(source_dir, dest_dir, val_ratio=0.2, seed=42):
    """
    Split a flat folder of images into train/val subdirectories.
    
    Expects source_dir to contain class subfolders:
        source_dir/
          positive/  *.png, *.jpg, *.dcm …
          negative/
    
    Creates:
        dest_dir/
          train/positive/, train/negative/
          val/positive/,   val/negative/
    """
    import shutil, random
    from pathlib import Path

    random.seed(seed)
    source_dir = Path(source_dir)
    dest_dir   = Path(dest_dir)

    for class_dir in source_dir.iterdir():
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        files = list(class_dir.iterdir())
        random.shuffle(files)

        n_val = max(1, int(len(files) * val_ratio))
        splits = {"val": files[:n_val], "train": files[n_val:]}

        for split, split_files in splits.items():
            out = dest_dir / split / class_name
            out.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, out / f.name)
                print(f"  {split}/{class_name}/{f.name}")

    print(f"\nDataset split complete → {dest_dir}")


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <image_path>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Loading: {path}")
    img = load_medical_image(path)
    print(f"Loaded PIL Image: {img.size} mode={img.mode}")

    tensor = preprocess_for_model(img)
    print(f"Tensor shape: {tensor.shape}  min={tensor.min():.3f}  max={tensor.max():.3f}")
    print("Preprocessing OK ✓")
