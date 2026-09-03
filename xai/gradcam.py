"""
Transformer-compatible Grad-CAM for RETFound ViT Architecture.
DR-EarlyXAI Project.
"""

import math
from typing import Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn as nn
from torchvision import transforms

# Evaluation & visualization preprocessing standards used by RETFound
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

eval_transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

display_transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
])


def load_visualization_image(image_path: Union[str, Image.Image]) -> np.ndarray:
    """
    Loads 224x224 center-cropped image without normalization,
    scaled to [0, 1] float32 RGB for display and overlay.
    """
    if isinstance(image_path, Image.Image):
        image = image_path.convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    image = display_transform(image)
    image_np = np.asarray(image).astype(np.float32) / 255.0
    return image_np


def load_model_input(
    image_path: Union[str, Image.Image],
    device: torch.device
) -> torch.Tensor:
    """
    Loads and normalizes an image using RETFound's ImageNet evaluation transform.
    Returns tensor with shape [1, 3, 224, 224].
    """
    if isinstance(image_path, Image.Image):
        image = image_path.convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    tensor = eval_transform(image).unsqueeze(0).to(device)
    return tensor


class RETFoundGradCAM:
    """
    Transformer Grad-CAM implementation tailored for Vision Transformer (RETFound).
    Extracts spatial attention heatmaps from the 196 patch tokens of the final block.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module, device: torch.device):
        self.model = model
        self.target_layer = target_layer
        self.device = device

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Register forward hook on target layer (e.g. model.blocks[-1].norm1)
        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._gradient_hook)

    def _gradient_hook(self, gradient):
        self.gradients = gradient

    def remove_hooks(self):
        """Removes registered PyTorch hooks to avoid memory leaks."""
        if self.forward_handle is not None:
            self.forward_handle.remove()
            self.forward_handle = None

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None
    ) -> Dict[str, Union[np.ndarray, int, float]]:
        """
        Generates Grad-CAM heatmap for the specified class index.
        If class_idx is None, defaults to the predicted class.
        """
        self.model.eval()
        self.activations = None
        self.gradients = None

        # Forward pass
        logits = self.model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = int(torch.argmax(probabilities, dim=1).item())

        if class_idx is None:
            target_class = predicted_class
        else:
            target_class = int(class_idx)

        # Backward pass from chosen class score
        score = logits[:, target_class].sum()
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Failed to capture activations or gradients for Grad-CAM.")

        activations = self.activations.detach()
        gradients = self.gradients.detach()

        if activations.ndim != 3:
            raise RuntimeError(f"Expected 3D activation tensor [B, tokens, C], got {activations.shape}")

        # Discard CLS token (token 0), keeping 196 patch tokens
        activations = activations[:, 1:, :]
        gradients = gradients[:, 1:, :]

        num_patches = activations.shape[1]
        grid_size = int(math.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            raise RuntimeError(f"Expected square patch grid from {num_patches} patches, but cannot compute integer sqrt.")

        # Channel-wise gradient pooling across tokens
        weights = gradients.mean(dim=1, keepdim=True)  # [B, 1, C]

        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=2)  # [B, num_patches]
        cam = torch.relu(cam)

        # Reshape to 2D grid: [B, 1, 14, 14]
        cam = cam.reshape(1, 1, grid_size, grid_size)

        # Upsample to 224 x 224 with bilinear interpolation
        cam = nn.functional.interpolate(
            cam,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )[0, 0]

        # Min-max normalization
        cam_min, cam_max = cam.min(), cam.max()
        if (cam_max - cam_min) > 1e-12:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return {
            "cam": cam.detach().cpu().numpy(),
            "logits": logits.detach().cpu().numpy()[0],
            "probabilities": probabilities.detach().cpu().numpy()[0],
            "confidence": float(probabilities[0, target_class].item()),
            "predicted_class": predicted_class,
            "target_class": target_class
        }


def save_single_gradcam_figure(
    image_path: Union[str, Image.Image],
    result: Dict[str, Union[np.ndarray, int, float]],
    output_path: str,
    actual_class: Optional[int] = None,
    dpi: int = 200
):
    """
    Saves a research-quality 3-panel figure:
    [Original Fundus] [Grad-CAM Heatmap] [Overlay]
    """
    original = load_visualization_image(image_path)
    cam = result["cam"]
    pred = result["predicted_class"]
    target = result["target_class"]
    conf = result["confidence"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Original
    axes[0].imshow(original)
    title_0 = f"Original Retinal Fundus\n"
    if actual_class is not None:
        title_0 += f"Actual: Grade {actual_class}"
    else:
        title_0 += "Input Image"
    axes[0].set_title(title_0, fontsize=12)
    axes[0].axis("off")

    # Panel 2: Grad-CAM
    im_cam = axes[1].imshow(cam, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"RETFound Grad-CAM\nTarget Class: Grade {target}", fontsize=12)
    axes[1].axis("off")
    fig.colorbar(im_cam, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Overlay
    axes[2].imshow(original)
    axes[2].imshow(cam, cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
    axes[2].set_title(
        f"Heatmap Overlay\nPredicted: Grade {pred} (Conf: {conf * 100:.1f}%)",
        fontsize=12
    )
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
