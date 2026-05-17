# EndoScan AI — Endometriosis Detection System

A computer vision pipeline for detecting endometriosis from ultrasound / MRI images, built on EfficientNet-B0 transfer learning with Grad-CAM visual explanations.

> ⚠️ **Research prototype only.** Not for clinical use. Always consult a qualified gynaecologist.

---

## Project Structure

```
endometriosis-ai/
├── training/
│   ├── train.py          ← Main training script
│   ├── gradcam.py        ← Grad-CAM heatmap utilities
│   ├── preprocess.py     ← DICOM + image loading utilities
│   └── requirements.txt  ← Python dependencies
├── app/
│   ├── index.html        ← Web interface (upload → result + heatmap)
│   └── server.py         ← Flask inference server
├── data/                 ← Put your images here (see below)
│   ├── train/
│   │   ├── positive/     ← Training images WITH endometriosis
│   │   └── negative/     ← Training images WITHOUT endometriosis
│   └── val/
│       ├── positive/
│       └── negative/
└── models/               ← Trained model saved here
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r training/requirements.txt
pip install flask          # for the web server
```

### 2. Prepare your dataset

Organise images into the folder structure above. Supported formats: `.png`, `.jpg`, `.tiff`, `.bmp`, `.dcm` (DICOM).

**Quick split helper** — if you have a flat folder of labelled images:
```python
from training.preprocess import prepare_dataset_split
prepare_dataset_split("my_raw_images/", "data/", val_ratio=0.2)
```

Recommended minimum: **100+ images per class** for meaningful results. More is always better.

### 3. Train the model

```bash
cd training
python train.py \
  --data_dir ../data \
  --model_dir ../models \
  --epochs 30 \
  --batch_size 16
```

Training uses **two phases**:
- **Phase 1** (first 5 epochs): backbone frozen, only the classifier head trains
- **Phase 2** (remaining epochs): full fine-tuning at a lower learning rate

Best model (by validation AUC) is saved automatically to `models/best_model.pt`.

### 4. Run the web app

```bash
cd app
python server.py --model ../models/best_model.pt
```

Then open **http://localhost:5000** in your browser.

Upload an ultrasound or MRI scan → get:
- ✅ Prediction (endometriosis / no endometriosis)
- 📊 Confidence score and probability
- 🔥 Grad-CAM heatmap showing which regions influenced the decision
- 📋 Key observations list

---

## Model Architecture

| Component | Choice | Reason |
|-----------|--------|--------|
| Backbone | EfficientNet-B0 | Strong accuracy/size tradeoff, pretrained on ImageNet |
| Head | Dropout → Linear(128) → ReLU → Dropout → Linear(2) | Reduces overfitting on small datasets |
| Loss | Cross-entropy with class weights | Handles class imbalance |
| Optimiser | AdamW + Cosine LR schedule | Stable convergence |
| Explainability | Grad-CAM on last conv layer | Shows spatial attention regions |

---

## Evaluation Metrics

For medical AI, track these (not just accuracy):

| Metric | Target | Why |
|--------|--------|-----|
| **Sensitivity (Recall)** | > 85% | Catching real cases — missing endometriosis is costly |
| **Specificity** | > 80% | Avoiding false alarms |
| **AUC-ROC** | > 0.88 | Overall discriminative ability |
| **F1 Score** | > 0.80 | Balance of precision + recall |

Training curves and confusion matrix are saved to `models/` after training.

---

## Improving the Model

1. **More data** — the single biggest lever. Aim for 500+ images per class.
2. **Data augmentation** — add elastic deformations, speckle noise (ultrasound-specific)
3. **Try larger backbone** — EfficientNet-B2 or ResNet-50 if you have enough data
4. **Segmentation** — instead of binary classification, predict pixel-level lesion masks (requires annotated segmentation masks)
5. **Ensemble** — average predictions from 3–5 models for better reliability

---

## Important Notes

- This model must be validated on an **independent test set** before any clinical evaluation
- Endometriosis diagnosis ultimately requires **laparoscopy** — imaging is supportive, not definitive
- Patient data must be handled according to **HIPAA / GDPR** regulations
- Any clinical deployment requires **regulatory approval** (FDA 510(k) in US, CE mark in Europe)

---

## Dataset Sources

- [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/)
- [Grand Challenge](https://grand-challenge.org) — search for pelvic imaging challenges
- [Kaggle](https://kaggle.com) — search for ultrasound datasets
- Academic collaborations with hospitals / radiology departments (recommended)
