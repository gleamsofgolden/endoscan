"""
Endometriosis Detection - Training Pipeline
============================================
Uses EfficientNet-B0 transfer learning for binary classification
of ultrasound/MRI images (endometriosis present vs absent).

Usage:
    python train.py --data_dir ../data --epochs 50 --batch_size 16

Directory structure expected:
    data/
      train/
        positive/   <- images WITH endometriosis
        negative/   <- images WITHOUT endometriosis
      val/
        positive/
        negative/
"""

import os
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt


# ── Config ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train endometriosis classifier")
    p.add_argument("--data_dir",      default="../data",   help="Root data directory")
    p.add_argument("--model_dir",     default="../models", help="Where to save model weights")
    p.add_argument("--epochs",        type=int,   default=50,   help="Number of training epochs")
    p.add_argument("--batch_size",    type=int,   default=16,   help="Batch size")
    p.add_argument("--lr",            type=float, default=1e-4, help="Learning rate (head phase)")
    p.add_argument("--img_size",      type=int,   default=224,  help="Input image size")
    p.add_argument("--freeze_epochs", type=int,   default=10,
                   help="Epochs to train only classifier head (backbone frozen). "
                        "Increased from 5 — give the head time to stabilise before unfreezing.")
    p.add_argument("--patience",      type=int,   default=10,
                   help="Early stopping patience (epochs without AUC improvement)")
    return p.parse_args()


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transforms(img_size, split="train"):
    """
    Medical imaging augmentations.
    Conservative colour ops (preserves ultrasound echogenicity / MRI signal).
    Added: RandomAffine and GaussianBlur — both safe for MRI/ultrasound.
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            # FIX: added affine (small shear/scale simulates probe angle variation)
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            # FIX: blur simulates scanner resolution variation
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes=2, pretrained=True):
    """
    EfficientNet-B0 with custom head.

    FIX 1: Dropout(inplace=True) removed — inplace ops corrupt the autograd
    graph that GradCAM's backward hooks depend on, causing silent wrong results
    in the attention maps. Never use inplace=True before a layer you'll hook.

    FIX 2: Dropout rates raised from 0.3/0.2 → 0.5/0.4 to combat overfitting
    on a small (<500 image) dataset.
    """
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),            # FIX 1: removed inplace=True
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(p=0.4),            # FIX 2: raised from 0.2
        nn.Linear(128, num_classes),
    )
    return model


def freeze_backbone(model):
    """Freeze everything except the classifier head."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_backbone(model):
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        # FIX: gradient clipping — prevents exploding gradients during fine-tuning
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return running_loss / total, correct / total, auc, all_probs, all_labels


# ── Plotting ──────────────────────────────────────────────────────────────────

def save_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("Loss"); axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"],   label="Val")
    axes[1].set_title("Accuracy"); axes[1].legend()

    axes[2].plot(history["val_auc"])
    axes[2].set_title("Val AUC-ROC")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Training curves saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  Endometriosis Detection — Training")
    print(f"  Device : {device}")
    print(f"  Data   : {args.data_dir}")
    print(f"  Epochs : {args.epochs}  (patience={args.patience})")
    print(f"{'='*55}\n")

    os.makedirs(args.model_dir, exist_ok=True)

    # ── Datasets
    train_ds = datasets.ImageFolder(
        os.path.join(args.data_dir, "train"),
        transform=get_transforms(args.img_size, "train")
    )
    val_ds = datasets.ImageFolder(
        os.path.join(args.data_dir, "val"),
        transform=get_transforms(args.img_size, "val")
    )

    # Class imbalance handling — counts for loss weighting
    class_counts = np.bincount([s[1] for s in train_ds.samples])
    weights = 1.0 / class_counts

    # FIX: REMOVE WeightedRandomSampler — it cancels out the loss weights!
    # Instead, let the loss function (CrossEntropyLoss with weight=) do the work.
    # This forces the model to learn positive cases properly, not just see balanced batches.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f"  Classes : {train_ds.classes}")
    print(f"  Train   : {len(train_ds)} images | Val: {len(val_ds)} images")
    print(f"  Class counts (train): {dict(zip(train_ds.classes, class_counts))}\n")

    # ── Model
    model = build_model(num_classes=2).to(device)

    # FIX: label_smoothing=0.1 prevents overconfidence on tiny datasets.
    # The model can't drive loss to zero by outputting extreme logits,
    # which is one of the main drivers of overfitting.
    class_weights = torch.tensor(weights / weights.sum() * 2, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # FIX: weight_decay raised from 1e-4 → 1e-2.
    # 1e-4 is almost no regularisation for a <500 image dataset.
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auc": []}
    best_auc        = 0.0
    epochs_no_improve = 0          # early stopping counter

    for epoch in range(1, args.epochs + 1):

        # Phase 1: freeze backbone for first N epochs
        if epoch == 1:
            print(f"  [Phase 1] Backbone frozen — training head only ({args.freeze_epochs} epochs)")
            freeze_backbone(model)
        if epoch == args.freeze_epochs + 1:
            print(f"\n  [Phase 2] Full fine-tuning — backbone unfrozen\n")
            unfreeze_backbone(model)
            # Lower LR for backbone — don't blow away pretrained features
            optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-2)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.freeze_epochs)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc, probs, labels = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        # Overfitting gap warning
        acc_gap = train_acc - val_acc
        gap_flag = f"  ⚠ gap={acc_gap:.2f}" if acc_gap > 0.15 else ""

        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"Train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"Val loss {val_loss:.4f} acc {val_acc:.3f} | "
              f"AUC {val_auc:.3f}{gap_flag}")

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_auc": val_auc,
                "val_acc": val_acc,
                "class_to_idx": train_ds.class_to_idx,
            }, os.path.join(args.model_dir, "best_model.pt"))
            print(f"    ✓ Best model saved (AUC {best_auc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\n  Early stopping — no AUC improvement for {args.patience} epochs.")
                break

    # ── Final evaluation
    print(f"\n{'='*55}")
    print("  Final Evaluation on Validation Set")
    print(f"{'='*55}")
    
    # IMPORTANT: For medical AI detecting disease, lower the threshold.
    # Default 0.5 is too conservative — use 0.3-0.4 to catch more positive cases.
    # Sensitivity (catching real endometriosis) is more important than specificity.
    INFERENCE_THRESHOLD = 0.35  # <-- Tune this based on sensitivity/specificity tradeoff
    pred_labels = [1 if p >= INFERENCE_THRESHOLD else 0 for p in probs]
    print(classification_report(labels, pred_labels, target_names=train_ds.classes))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, pred_labels))
    print(f"\nBest Val AUC-ROC : {best_auc:.4f}")

    # Save curves + history
    save_training_curves(history, os.path.join(args.model_dir, "training_curves.png"))
    with open(os.path.join(args.model_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Done! Model saved to {args.model_dir}/best_model.pt\n")


if __name__ == "__main__":
    main()