"""
Quantitative XAI Evaluation Module for DR-EarlyXAI.
Implements objective, research-standard faithfulness metrics:
1. Deletion Metric (AUC of score decay under iterative top-patch masking)
2. Insertion Metric (AUC of score rise under iterative top-patch restoration)
3. Confidence Drop Rate (%)
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

import torch
import torch.nn as nn
from torchvision import transforms

from xai.gradcam import (
    RETFoundGradCAM,
    load_model_input,
    load_visualization_image,
    IMAGENET_MEAN,
    IMAGENET_STD
)


class XAIEvaluator:
    """
    Evaluator for explanation faithfulness and clinical relevance.
    Quantifies whether Grad-CAM heatmaps truly identify the decision-making features.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.eval_transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

    def _tensor_from_pil(self, img: Image.Image) -> torch.Tensor:
        return self.eval_transform(img).unsqueeze(0).to(self.device)

    def evaluate_deletion(
        self,
        image_path: Union[str, Path],
        cam: np.ndarray,
        target_class: int,
        steps: int = 10,
        blur_radius: float = 12.0
    ) -> Dict[str, Union[List[float], float]]:
        """
        Deletion Metric:
        Sequentially removes (blurs/zeros) top-attended regions in order of Grad-CAM intensity.
        Measures the rapid drop in predicted probability for target_class.
        Lower AUC indicates higher faithfulness (crucial evidence is quickly destroyed).
        """
        self.model.eval()
        orig_img = Image.open(image_path).convert("RGB")
        orig_display = Image.fromarray((load_visualization_image(orig_img) * 255).astype(np.uint8))
        blurred_img = orig_display.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        orig_np = np.array(orig_display, dtype=np.float32)
        blur_np = np.array(blurred_img, dtype=np.float32)

        # Flatten CAM values and find pixel rank thresholds
        flat_cam = cam.flatten()
        sorted_indices = np.argsort(flat_cam)[::-1]  # Highest to lowest importance

        total_pixels = len(flat_cam)
        step_fractions = np.linspace(0.0, 1.0, steps + 1)
        probabilities = []

        with torch.inference_mode():
            for frac in step_fractions:
                if frac == 0.0:
                    current_img = Image.fromarray(orig_np.astype(np.uint8))
                else:
                    num_to_mask = int(frac * total_pixels)
                    mask_indices = sorted_indices[:num_to_mask]
                    mask_flat = np.zeros(total_pixels, dtype=bool)
                    mask_flat[mask_indices] = True
                    mask_2d = mask_flat.reshape(cam.shape)[:, :, np.newaxis]

                    perturbed_np = np.where(mask_2d, blur_np, orig_np)
                    current_img = Image.fromarray(perturbed_np.astype(np.uint8))

                # Compute target class probability
                inp = self._tensor_from_pil(current_img)
                logits = self.model(inp)
                prob = float(torch.softmax(logits, dim=1)[0, target_class].item())
                probabilities.append(prob)

        # Compute Area Under the Deletion Curve (AUC) via trapezoidal rule
        deletion_auc = float(np.trapezoid(probabilities, step_fractions))
        confidence_drop = float((probabilities[0] - probabilities[-1]) / (probabilities[0] + 1e-8))

        return {
            "fractions": step_fractions.tolist(),
            "probabilities": probabilities,
            "deletion_auc": deletion_auc,
            "initial_prob": probabilities[0],
            "final_prob": probabilities[-1],
            "confidence_drop": confidence_drop
        }

    def evaluate_insertion(
        self,
        image_path: Union[str, Path],
        cam: np.ndarray,
        target_class: int,
        steps: int = 10,
        blur_radius: float = 12.0
    ) -> Dict[str, Union[List[float], float]]:
        """
        Insertion Metric:
        Starts from an uninformative blurred image and sequentially restores
        top-attended regions from the original image.
        Higher AUC indicates higher faithfulness (crucial evidence is quickly restored).
        """
        self.model.eval()
        orig_img = Image.open(image_path).convert("RGB")
        orig_display = Image.fromarray((load_visualization_image(orig_img) * 255).astype(np.uint8))
        blurred_img = orig_display.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        orig_np = np.array(orig_display, dtype=np.float32)
        blur_np = np.array(blurred_img, dtype=np.float32)

        flat_cam = cam.flatten()
        sorted_indices = np.argsort(flat_cam)[::-1]  # Highest to lowest importance

        total_pixels = len(flat_cam)
        step_fractions = np.linspace(0.0, 1.0, steps + 1)
        probabilities = []

        with torch.inference_mode():
            for frac in step_fractions:
                if frac == 0.0:
                    current_img = Image.fromarray(blur_np.astype(np.uint8))
                else:
                    num_to_restore = int(frac * total_pixels)
                    restore_indices = sorted_indices[:num_to_restore]
                    mask_flat = np.zeros(total_pixels, dtype=bool)
                    mask_flat[restore_indices] = True
                    mask_2d = mask_flat.reshape(cam.shape)[:, :, np.newaxis]

                    perturbed_np = np.where(mask_2d, orig_np, blur_np)
                    current_img = Image.fromarray(perturbed_np.astype(np.uint8))

                inp = self._tensor_from_pil(current_img)
                logits = self.model(inp)
                prob = float(torch.softmax(logits, dim=1)[0, target_class].item())
                probabilities.append(prob)

        insertion_auc = float(np.trapezoid(probabilities, step_fractions))
        confidence_gain = float((probabilities[-1] - probabilities[0]) / (probabilities[-1] + 1e-8))

        return {
            "fractions": step_fractions.tolist(),
            "probabilities": probabilities,
            "insertion_auc": insertion_auc,
            "initial_prob": probabilities[0],
            "final_prob": probabilities[-1],
            "confidence_gain": confidence_gain
        }


def save_faithfulness_figure(
    deletion_results: Dict,
    insertion_results: Dict,
    output_path: str,
    target_class: int,
    dpi: int = 200
):
    """
    Saves a research-quality plot of Deletion and Insertion curves side-by-side.
    """
    fracs = deletion_results["fractions"]
    del_probs = deletion_results["probabilities"]
    ins_probs = insertion_results["probabilities"]
    del_auc = deletion_results["deletion_auc"]
    ins_auc = insertion_results["insertion_auc"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Deletion Curve
    axes[0].plot(fracs, del_probs, marker="o", color="#d62728", linewidth=2.2, label=f"Deletion (AUC = {del_auc:.3f})")
    axes[0].fill_between(fracs, del_probs, color="#d62728", alpha=0.15)
    axes[0].set_title("Deletion Curve (Lower AUC = More Faithful)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Fraction of Salient Pixels Masked", fontsize=11)
    axes[0].set_ylabel(f"Grade {target_class} Probability", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=11)
    axes[0].set_ylim([-0.05, 1.05])

    # Insertion Curve
    axes[1].plot(fracs, ins_probs, marker="s", color="#1f77b4", linewidth=2.2, label=f"Insertion (AUC = {ins_auc:.3f})")
    axes[1].fill_between(fracs, ins_probs, color="#1f77b4", alpha=0.15)
    axes[1].set_title("Insertion Curve (Higher AUC = More Faithful)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Fraction of Salient Pixels Restored", fontsize=11)
    axes[1].set_ylabel(f"Grade {target_class} Probability", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=11)
    axes[1].set_ylim([-0.05, 1.05])

    plt.tight_layout()
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_file), dpi=dpi, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    import argparse
    import sys
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root / "RETFound"))
    sys.path.insert(0, str(project_root))

    import models_vit as models
    from xai.gradcam import RETFoundGradCAM, load_model_input

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Faithfulness Evaluation on {device}...")
    model = models.RETFound_mae(img_size=224, num_classes=5, drop_path_rate=0.2, global_pool=True)
    ckpt = torch.load("checkpoint-best.pth", map_location="cpu", weights_only=False)
    state = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    test_imgs = list((project_root / "test_images").glob("*.png"))
    sample_img = str(test_imgs[0])

    gradcam = RETFoundGradCAM(model=model, target_layer=model.blocks[-1].norm1, device=device)
    inp = load_model_input(sample_img, device)
    cam_res = gradcam.generate(inp)
    pred_class = cam_res["predicted_class"]
    gradcam.remove_hooks()

    evaluator = XAIEvaluator(model=model, device=device)
    del_res = evaluator.evaluate_deletion(sample_img, cam_res["cam"], target_class=pred_class, steps=10)
    ins_res = evaluator.evaluate_insertion(sample_img, cam_res["cam"], target_class=pred_class, steps=10)

    print(f"Image: {sample_img}")
    print(f"Predicted Class: Grade {pred_class}")
    print(f"Deletion AUC : {del_res['deletion_auc']:.4f} (Drop: {del_res['confidence_drop']*100:.1f}%)")
    print(f"Insertion AUC: {ins_res['insertion_auc']:.4f} (Gain: {ins_res['confidence_gain']*100:.1f}%)")

    out_fig = "outputs/metrics/faithfulness_curves.png"
    save_faithfulness_figure(del_res, ins_res, out_fig, target_class=pred_class)
    print(f"Saved evaluation figure to: {out_fig}")
