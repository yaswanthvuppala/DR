"""
Dual Grad-CAM implementation for RETFound (DR-EarlyXAI).
Generates side-by-side comparative explanations for misclassified retinal images:
1. CAM(target=predicted_class) - Why did RETFound make this error?
2. CAM(target=ground_truth_class) - What features correspond to the true grade?
"""

from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from xai.gradcam import (
    RETFoundGradCAM,
    load_model_input,
    load_visualization_image
)

CLASS_NAMES = [
    "Grade 0 (No DR)",
    "Grade 1 (Mild)",
    "Grade 2 (Moderate)",
    "Grade 3 (Severe)",
    "Grade 4 (Proliferative DR)"
]


class DualGradCAM:
    """
    Dual Grad-CAM Engine:
    Evaluates both predicted and ground-truth classes to explain diagnostic errors.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.gradcam = RETFoundGradCAM(model=model, target_layer=target_layer, device=device)

    def generate_dual(
        self,
        input_tensor: torch.Tensor,
        actual_class: int,
        predicted_class: Optional[int] = None
    ) -> Dict[str, Union[np.ndarray, int, float, Dict]]:
        """
        Generates CAM for predicted class and CAM for actual class.
        """
        # 1. Generate CAM for predicted class
        pred_result = self.gradcam.generate(input_tensor, class_idx=predicted_class)
        pred_class = pred_result["predicted_class"]

        # 2. Generate CAM for ground truth class
        true_result = self.gradcam.generate(input_tensor, class_idx=int(actual_class))

        # 3. Compute absolute difference heatmap
        cam_pred = pred_result["cam"]
        cam_true = true_result["cam"]
        cam_diff = np.abs(cam_pred - cam_true)

        return {
            "predicted_class": pred_class,
            "actual_class": int(actual_class),
            "is_correct": bool(pred_class == int(actual_class)),
            "probabilities": pred_result["probabilities"],
            "pred_confidence": float(pred_result["probabilities"][pred_class]),
            "true_confidence": float(pred_result["probabilities"][int(actual_class)]),
            "cam_pred": cam_pred,
            "cam_true": cam_true,
            "cam_diff": cam_diff,
            "pred_result": pred_result,
            "true_result": true_result
        }

    def remove_hooks(self):
        self.gradcam.remove_hooks()


def save_dual_gradcam_figure(
    image_path: Union[str, Image.Image],
    dual_result: Dict,
    output_path: str,
    dpi: int = 200
):
    """
    Saves a research-grade 5-panel figure:
    [Original] [CAM Pred] [Overlay Pred] [CAM True] [Overlay True]
    """
    original = load_visualization_image(image_path)
    actual = dual_result["actual_class"]
    pred = dual_result["predicted_class"]
    pred_conf = dual_result["pred_confidence"]
    true_conf = dual_result["true_confidence"]
    cam_pred = dual_result["cam_pred"]
    cam_true = dual_result["cam_true"]
    is_correct = dual_result["is_correct"]

    status_str = "CORRECT" if is_correct else "MISCLASSIFIED"

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

    # Panel 1: Original Retinal Fundus
    axes[0].imshow(original)
    axes[0].set_title(
        f"Input Fundus ({status_str})\n"
        f"Actual: Grade {actual} | Pred: Grade {pred}",
        fontsize=11,
        fontweight="bold",
        color="darkgreen" if is_correct else "darkred"
    )
    axes[0].axis("off")

    # Panel 2: CAM for Predicted Class
    im_pred = axes[1].imshow(cam_pred, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title(
        f"CAM (Target: Predicted Grade {pred})\n"
        f"Model Confidence: {pred_conf * 100:.1f}%",
        fontsize=11
    )
    axes[1].axis("off")

    # Panel 3: Overlay Predicted Class
    axes[2].imshow(original)
    axes[2].imshow(cam_pred, cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
    axes[2].set_title(f"Overlay: Predicted Grade {pred}\n({CLASS_NAMES[pred]})", fontsize=11)
    axes[2].axis("off")

    # Panel 4: CAM for Ground-Truth Class
    axes[3].imshow(cam_true, cmap="jet", vmin=0.0, vmax=1.0)
    axes[3].set_title(
        f"CAM (Target: True Grade {actual})\n"
        f"Model Confidence: {true_conf * 100:.1f}%",
        fontsize=11
    )
    axes[3].axis("off")

    # Panel 5: Overlay Ground-Truth Class
    axes[4].imshow(original)
    axes[4].imshow(cam_true, cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
    axes[4].set_title(f"Overlay: True Grade {actual}\n({CLASS_NAMES[actual]})", fontsize=11)
    axes[4].axis("off")

    # Shared colorbar
    fig.subplots_adjust(right=0.92, wspace=0.15)
    cbar_ax = fig.add_axes([0.93, 0.20, 0.012, 0.60])
    fig.colorbar(im_pred, cax=cbar_ax)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_file), dpi=dpi, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    import argparse
    import sys
    
    # Allow running directly as standalone script
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root / "RETFound"))
    sys.path.insert(0, str(project_root))
    
    import models_vit as models
    torch.serialization.add_safe_globals([argparse.Namespace])

    parser = argparse.ArgumentParser(description="Run Dual Grad-CAM on a retinal fundus image")
    parser.add_argument("--image", default=None, help="Path to input fundus image")
    parser.add_argument("--actual", type=int, default=3, help="Ground truth grade (0-4)")
    parser.add_argument("--checkpoint", default="checkpoint-best.pth", help="Checkpoint path")
    parser.add_argument("--output", default="outputs/gradcam/dual_gradcam_test.png", help="Output path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading RETFound on {device}...")
    model = models.RETFound_mae(img_size=224, num_classes=5, drop_path_rate=0.2, global_pool=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    if args.image is None:
        test_imgs = list((project_root / "test_images").glob("*.png"))
        args.image = str(test_imgs[0])

    print(f"Running Dual Grad-CAM on {args.image} with actual class {args.actual}...")
    dual_engine = DualGradCAM(model=model, target_layer=model.blocks[-1].norm1, device=device)
    inp = load_model_input(args.image, device)
    res = dual_engine.generate_dual(inp, actual_class=args.actual)

    save_dual_gradcam_figure(args.image, res, args.output)
    print(f"Saved Dual Grad-CAM figure to: {args.output}")
    print(f"Predicted Grade: {res['predicted_class']} (Conf: {res['pred_confidence']*100:.2f}%)")
    print(f"Actual Grade   : {res['actual_class']} (Conf: {res['true_confidence']*100:.2f}%)")
