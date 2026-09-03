# Phase 1 & 2: Local RETFound Setup, Architecture Reconstruction & Checkpoint Validation

---

## 1. Overview & Objectives

In **Phase 1** and **Phase 2**, the objective was to transition the fine-tuned RETFound Vision Transformer (ViT) baseline from Kaggle to the local workstation without retraining or modifying the trained weights.

The goals achieved:
1. **Environment Setup**: Configured an isolated Python virtual environment (`.venv`) with **PyTorch 2.6.0** and **CUDA 12.4** acceleration targeting the **NVIDIA GeForce RTX 3050 Laptop GPU** (4 GB VRAM).
2. **Architecture Reconstruction**: Sourced the authentic RETFound repository and instantiated `RETFound_mae` with the exact hyperparameters used in Kaggle.
3. **Weight Integrity Check**: Loaded `checkpoint-best.pth` (1.21 GB) and verified that **0 keys were missing** and **0 unexpected keys** existed.
4. **Single-Image Sanity Test**: Verified the full end-to-end forward pipeline (image loading $\to$ ImageNet normalization $\to$ forward pass $\to$ softmax distribution $\to$ predicted grade) on a single test retinal fundus image.

---

## 2. Architecture Specification

The fine-tuned model is based on the **RETFound MAE (Masked Autoencoder)** architecture pre-trained on 1.6 million retinal images:

| Parameter | Specification | Purpose in DR-EarlyXAI |
| :--- | :--- | :--- |
| **Model Type** | `RETFound_mae` (ViT-Large backbone) | Retinal foundation model |
| **Input Resolution** | $224 \times 224 \times 3$ | Standardized fundus input |
| **Patch Size** | $16 \times 16$ pixels | Produces $(224/16)^2 = 196$ spatial patches |
| **Embedding Dimension** | $1024$ | Hidden representation per patch token |
| **Transformer Blocks** | $24$ blocks (`blocks[0]` to `blocks[23]`) | Deep self-attention feature extraction |
| **Attention Heads** | $16$ heads per block | Multi-head self-attention |
| **Drop Path Rate** | $0.2$ (Stochastic Depth) | Regularization during fine-tuning |
| **Global Average Pooling** | `True` (`global_pool=True`) | Pools 196 patch tokens (excluding CLS) |
| **Output Classes** | $5$ classes (Grades 0 to 4) | International Clinical DR Severity scale |
| **Total Parameters** | **303.31 Million** | Complete fine-tuned network size |

---

## 3. Checkpoint Validation Results

The checkpoint [`checkpoint-best.pth`](file:///c:/Users/vuppa/Desktop/DR/checkpoint-best.pth) was loaded with the following diagnostic log:

```text
==================================================
RETFound Single Image Inference & Grad-CAM Test
==================================================
Device: cuda
GPU: NVIDIA GeForce RTX 3050 Laptop GPU

[1] Checkpoint: checkpoint-best.pth (1.21 GB)
[2] Total Parameters: 303.31M
[3] Checkpoint Loading:
    Missing keys   : 0
    Unexpected keys: 0
    Model successfully placed on GPU device.
```

### Key Technical Note on Checkpoint Loading:
PyTorch 2.6+ defaults to `weights_only=True` during `torch.load()`. Because the training checkpoint saved the command-line arguments as an `argparse.Namespace` object inside the dictionary, we added `argparse.Namespace` to PyTorch's safe globals list:
```python
import argparse
torch.serialization.add_safe_globals([argparse.Namespace])
checkpoint = torch.load("checkpoint-best.pth", map_location="cpu", weights_only=False)
```
This guarantees safe, error-free deserialization while preserving all training metadata.

---

## 4. Evaluation Preprocessing Pipeline

To ensure test predictions match the Kaggle training run without drift, the preprocessing pipeline applies:

1. **PIL Image Loading**: Loaded in RGB format.
2. **Resize**: Resized to $256 \times 256$ using **Bicubic Interpolation** (`transforms.InterpolationMode.BICUBIC`).
3. **Center Crop**: Cropped to $224 \times 224$ pixels centered on the macula/optic disc region.
4. **Tensor Conversion**: Scaled to float range $[0.0, 1.0]$.
5. **ImageNet Normalization**:
   $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
   $$x_{\text{norm}} = \frac{x - \mu}{\sigma}$$

---

## 5. Single Image Validation Result

- **Sample Image**: `test_images/0005cfc8afb6.png`
- **Inference Time**: $\approx 42\text{ ms}$ on NVIDIA RTX 3050 GPU.
- **Predicted Severity**: **Grade 2 (Moderate Diabetic Retinopathy)**
- **Confidence**: **69.40%**

### Full Probability Distribution:
```text
  [0] Grade 0 (No DR)            :   0.36% | 
  [1] Grade 1 (Mild)             :  22.90% | ######
  [2] Grade 2 (Moderate)         :  69.40% | ####################
  [3] Grade 3 (Severe)           :   3.54% | #
  [4] Grade 4 (Proliferative DR) :   3.79% | #
```

---

## 6. How to Run Single-Image Inference

From the terminal:
```powershell
# Run default test image
.\.venv\Scripts\python.exe test_single_inference.py

# Run on any custom fundus image
.\.venv\Scripts\python.exe test_single_inference.py --image test_images/003f0afdcd15.png --output outputs/gradcam/sample.png
```
Or use the interactive Jupyter notebook: [`notebooks/RETFound_XAI.ipynb`](file:///c:/Users/vuppa/Desktop/DR/notebooks/RETFound_XAI.ipynb).
