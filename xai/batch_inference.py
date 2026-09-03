"""
Batch Inference & Baseline Metric Reproduction for RETFound (DR-EarlyXAI).
Supports full dataset inference, confusion matrix generation, QWK, Macro-F1, and ROC-AUC.
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    cohen_kappa_score,
    roc_auc_score
)

# Preprocessing transforms
from xai.gradcam import IMAGENET_MEAN, IMAGENET_STD

CLASS_NAMES = [
    "Grade 0 (No DR)",
    "Grade 1 (Mild)",
    "Grade 2 (Moderate)",
    "Grade 3 (Severe)",
    "Grade 4 (Proliferative DR)"
]


class FundusDataset(Dataset):
    def __init__(self, image_paths: List[Path], labels: Optional[List[int]] = None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        tensor = self.transform(image)
        label = self.labels[idx] if self.labels is not None else -1
        return tensor, str(img_path), label


def plot_confusion_matrix(cm: np.ndarray, output_path: str, accuracy: float, qwk: float):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = [f"G{i}" for i in range(5)]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=f"RETFound Confusion Matrix\nAccuracy: {accuracy*100:.2f}% | QWK: {qwk*100:.2f}%",
        ylabel="Ground Truth Grade",
        xlabel="Predicted Grade"
    )
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_batch_inference(
    image_dir: str = "test_images",
    checkpoint_path: str = "checkpoint-best.pth",
    output_csv: str = "outputs/predictions/predictions.csv",
    batch_size: int = 16,
    limit: Optional[int] = None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Batch Inference on {device} (Batch size: {batch_size})")

    # Reconstruct architecture
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root / "RETFound"))
    import models_vit as models

    model = models.RETFound_mae(img_size=224, num_classes=5, drop_path_rate=0.2, global_pool=True)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    # Discover images
    img_dir = Path(image_dir)
    image_paths = sorted(list(img_dir.rglob("*.png")) + list(img_dir.rglob("*.jpg")) + list(img_dir.rglob("*.jpeg")))

    if limit is not None and limit > 0:
        image_paths = image_paths[:limit]

    print(f"Total images to evaluate: {len(image_paths)}")
    if not image_paths:
        raise FileNotFoundError(f"No images found in {img_dir}")

    # Check for ground-truth directory structure (e.g. dir/0/img.png)
    has_labels = True
    labels = []
    for p in image_paths:
        parent_name = p.parent.name
        if parent_name in ["0", "1", "2", "3", "4"]:
            labels.append(int(parent_name))
        else:
            has_labels = False
            break

    dataset = FundusDataset(image_paths, labels if has_labels else None)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    all_paths = []
    all_preds = []
    all_confs = []
    all_probs = []
    all_targets = []

    print("Executing batch inference across dataset...")
    with torch.inference_mode():
        for tensors, paths, targets in dataloader:
            tensors = tensors.to(device)
            logits = model(tensors)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            for i in range(len(paths)):
                all_paths.append(paths[i])
                all_preds.append(int(preds[i]))
                all_confs.append(float(probs[i, preds[i]]))
                all_probs.append(probs[i])
                if has_labels:
                    all_targets.append(int(targets[i]))

    # Build DataFrame
    prob_cols = [f"prob_grade_{i}" for i in range(5)]
    prob_matrix = np.array(all_probs)
    df_data = {
        "image_path": all_paths,
        "predicted_grade": all_preds,
        "confidence": all_confs,
    }
    for i, col in enumerate(prob_cols):
        df_data[col] = prob_matrix[:, i]

    if has_labels:
        df_data["ground_truth"] = all_targets
        df_data["is_correct"] = [p == t for p, t in zip(all_preds, all_targets)]

    df = pd.DataFrame(df_data)
    out_file = Path(output_csv)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"\nSaved inference predictions to: {out_file.resolve()}")

    # Print class distribution
    print("\nPredicted Class Distribution:")
    for grade in range(5):
        count = sum(df["predicted_grade"] == grade)
        pct = (count / len(df)) * 100
        print(f"  Grade {grade} ({CLASS_NAMES[grade]}): {count:4d} images ({pct:5.1f}%)")

    # If ground-truth labels available, compute metrics
    if has_labels:
        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

        print("\n" + "=" * 50)
        print("BASELINE REPRODUCTION METRICS (Ground Truth Available):")
        print(f"  Accuracy : {acc * 100:.2f}%")
        print(f"  Macro-F1 : {f1 * 100:.2f}%")
        print(f"  Precision: {prec * 100:.2f}%")
        print(f"  Recall   : {rec * 100:.2f}%")
        print(f"  QWK      : {qwk * 100:.2f}%")
        print("=" * 50)

        # Plot confusion matrix
        cm_path = "outputs/metrics/confusion_matrix.png"
        plot_confusion_matrix(cm, cm_path, accuracy=acc, qwk=qwk)
        print(f"Confusion matrix saved to: {cm_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="test_images")
    parser.add_argument("--checkpoint", default="checkpoint-best.pth")
    parser.add_argument("--output_csv", default="outputs/predictions/predictions.csv")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_batch_inference(
        image_dir=args.image_dir,
        checkpoint_path=args.checkpoint,
        output_csv=args.output_csv,
        batch_size=args.batch_size,
        limit=args.limit
    )
