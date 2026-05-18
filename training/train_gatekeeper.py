"""
Gatekeeper — MRI vs Not-MRI Classifier
========================================
Trains a lightweight binary classifier to reject non-MRI images
before they reach the endometriosis detection model.

Directory structure expected:
    gatekeeper_data/
      train/
        mri/        <- your existing converted MRI PNGs
        not_mri/    <- random everyday photos
      val/
        mri/
        not_mri/

Usage:
    python training\train_gatekeeper.py
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V3_Small_Weights
from sklearn.metrics import classification_report, confusion_matrix


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default="gatekeeper_data")
    p.add_argument("--model_dir",  default="models")
    p.add_argument("--epochs",     type=int, default=15)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--img_size",   type=int, default=224)
    return p.parse_args()


def get_transforms(img_size, split="train"):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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


def build_model():
    """
    MobileNetV3-Small — very lightweight, fast on CPU.
    Perfect for a gatekeeper that just needs to say MRI vs not.
    """
    model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 2)
    return model


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * images.size(0)
        correct += out.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        loss = criterion(out, labels)
        loss_sum += loss.item() * images.size(0)
        preds = out.argmax(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return loss_sum / total, correct / total, all_preds, all_labels


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"  Gatekeeper Training — MRI vs Not-MRI")
    print(f"  Device : {device}")
    print(f"{'='*50}\n")

    os.makedirs(args.model_dir, exist_ok=True)

    train_ds = datasets.ImageFolder(
        os.path.join(args.data_dir, "train"),
        transform=get_transforms(args.img_size, "train")
    )
    val_ds = datasets.ImageFolder(
        os.path.join(args.data_dir, "val"),
        transform=get_transforms(args.img_size, "val")
    )

    # Weighted sampler for class imbalance
    counts = np.bincount([s[1] for s in train_ds.samples])
    weights = 1.0 / counts
    sample_weights = torch.tensor([weights[s[1]] for s in train_ds.samples])
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(train_ds))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    print(f"  Classes : {train_ds.classes}")
    print(f"  Train   : {len(train_ds)} | Val: {len(val_ds)}")
    print(f"  Class counts (train): {dict(zip(train_ds.classes, counts))}\n")

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, preds, labels = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  Epoch {epoch:2d}/{args.epochs} | "
              f"Train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"Val loss {val_loss:.4f} acc {val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "class_to_idx": train_ds.class_to_idx,
            }, os.path.join(args.model_dir, "gatekeeper.pt"))
            print(f"    ✓ Best gatekeeper saved (acc {best_acc:.4f})")

    print(f"\n{'='*50}")
    print("  Final Evaluation")
    print(f"{'='*50}")
    print(classification_report(labels, preds, target_names=train_ds.classes))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))
    print(f"\nBest Val Accuracy : {best_acc:.4f}")
    print(f"Saved → {args.model_dir}/gatekeeper.pt\n")


if __name__ == "__main__":
    main()
