"""
Single-image inference and Grad-CAM validation script for RETFound.
DR-EarlyXAI Project.
"""

import os
import sys
import argparse
from pathlib import Path

# Add RETFound repository to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
RETFOUND_DIR = SCRIPT_DIR / "RETFound"
sys.path.insert(0, str(RETFOUND_DIR))

import torch
import torch.nn as nn
import numpy as np

# Allow argparse.Namespace in PyTorch checkpoint deserialization
torch.serialization.add_safe_globals([argparse.Namespace])

import models_vit as models
from xai.gradcam import RETFoundGradCAM, load_model_input, save_single_gradcam_figure

CLASS_NAMES = [
    "Grade 0 (No DR)",
    "Grade 1 (Mild)",
    "Grade 2 (Moderate)",
    "Grade 3 (Severe)",
    "Grade 4 (Proliferative DR)"
]


def validate_single_inference(
    checkpoint_path: str = "checkpoint-best.pth",
    image_path: str = None,
    output_path: str = "outputs/gradcam/single_test_gradcam.png"
):
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================================================")
    print(f"RETFound Single Image Inference & Grad-CAM Test")
    print(f"==================================================")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Allocated memory: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")

    # Select default image if none provided
    if image_path is None:
        test_dir = SCRIPT_DIR / "test_images"
        images = list(test_dir.glob("*.png"))
        if not images:
            raise FileNotFoundError(f"No PNG images found in {test_dir}")
        image_path = str(images[0])

    print(f"\n[1] Selected Test Image: {image_path}")

    # Verify checkpoint exists
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_file.resolve()}")
    print(f"[2] Checkpoint: {ckpt_file.resolve()} ({ckpt_file.stat().st_size / 1e9:.2f} GB)")

    # Reconstruct RETFound_mae architecture
    print("\n[3] Reconstructing RETFound_mae architecture...")
    model = models.RETFound_mae(
        img_size=224,
        num_classes=5,
        drop_path_rate=0.2,
        global_pool=True,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Total parameters: {total_params / 1e6:.2f}M")

    # Load weights
    print("\n[4] Loading checkpoint weights...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model' state_dict.")

    state_dict = checkpoint["model"]
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k[len("module."):] if k.startswith("module.") else k
        clean_state_dict[clean_k] = v

    missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
    print(f"    Missing keys   : {len(missing)} (first 5: {missing[:5]})")
    print(f"    Unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")

    if len(missing) > 10:
        raise RuntimeError(f"Too many missing keys ({len(missing)})! Architecture mismatch suspected.")

    model = model.to(device)
    model.eval()
    print("    Model successfully loaded and placed on device.")

    # Prepare model input
    print("\n[5] Preprocessing retinal image...")
    input_tensor = load_model_input(image_path, device)
    print(f"    Input tensor shape: {tuple(input_tensor.shape)}")

    # Target layer for Grad-CAM
    target_layer = model.blocks[-1].norm1
    print(f"\n[6] Grad-CAM Target Layer: {target_layer}")

    gradcam = RETFoundGradCAM(model=model, target_layer=target_layer, device=device)

    # Run inference and Grad-CAM
    print("\n[7] Running inference and Grad-CAM generation...")
    result = gradcam.generate(input_tensor)

    pred_class = result["predicted_class"]
    probs = result["probabilities"]
    confidence = result["confidence"]
    cam = result["cam"]

    print(f"\n==================================================")
    print(f"PREDICTION RESULTS:")
    print(f"Predicted Class: Grade {pred_class} -> {CLASS_NAMES[pred_class]}")
    print(f"Confidence     : {confidence * 100:.2f}%")
    print(f"--------------------------------------------------")
    print("Class Probability Distribution:")
    for grade, (name, p) in enumerate(zip(CLASS_NAMES, probs)):
        bar = "#" * int(p * 30)
        print(f"  [{grade}] {name:<27}: {p * 100:6.2f}% | {bar}")
    print(f"==================================================")

    # Grad-CAM statistics
    print(f"\n[8] Grad-CAM Heatmap Statistics:")
    print(f"    Shape: {cam.shape}")
    print(f"    Range: [{cam.min():.4f}, {cam.max():.4f}]")
    print(f"    Mean : {cam.mean():.4f}, Std: {cam.std():.4f}")

    # Save visualization
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_single_gradcam_figure(
        image_path=image_path,
        result=result,
        output_path=str(out_path)
    )
    print(f"\n[9] Saved visualization figure to: {out_path.resolve()}")
    gradcam.remove_hooks()
    print("\nMilestone 1 single-image validation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RETFound Single Image Inference & Grad-CAM")
    parser.add_argument("--checkpoint", default="checkpoint-best.pth", help="Path to checkpoint-best.pth")
    parser.add_argument("--image", default=None, help="Path to test image")
    parser.add_argument("--output", default="outputs/gradcam/single_test_gradcam.png", help="Output figure path")
    args = parser.parse_args()

    validate_single_inference(
        checkpoint_path=args.checkpoint,
        image_path=args.image,
        output_path=args.output
    )
