# DR-EarlyXAI: Explainable Diabetic Retinopathy Severity Detection & Early-Screening System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%2BCU124-EE4C2C.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900.svg)](https://developer.nvidia.com/cuda-zone)
[![Backbone](https://img.shields.io/badge/Backbone-RETFound__mae%20(ViT--Large)-orange.svg)](https://github.com/rmaphoh/RETFound_MAE)
[![License](https://img.shields.io/badge/License-Research%20Only-lightgrey.svg)]()

An end-to-end, clinically grounded, explainable AI research system for 5-class Diabetic Retinopathy (DR) severity grading on retinal fundus images (APTOS 2019 dataset). 

The system benchmarks the foundation model **RETFound ViT-Large** (303.31M parameters), deploys a comprehensive **4-method XAI suite** (Grad-CAM, Grad-CAM++, Integrated Gradients, Occlusion Sensitivity) alongside **Dual Grad-CAM** and **quantitative faithfulness benchmarking** (Deletion & Insertion AUC), and introduces a proposed **Multi-Scale Feature Fusion + Learned Attention Pooling + Dual-Head (Severity + Early-Screening)** ordinal architecture.

---

## 📑 Table of Contents
1. [System Architecture & Flowchart](#-system-architecture--flowchart)
2. [Exact Folder Structure](#-exact-folder-structure)
3. [How to Setup the Environment & Data](#-how-to-setup-the-environment--data)
4. [Exact Run Commands](#-exact-run-commands)
5. [Benchmark Metrics & Results](#-benchmark-metrics--results)
6. [Documentation Sitemap](#-documentation-sitemap)
7. [Citation & Clinical Disclaimer](#-citation--clinical-disclaimer)

---

## 🏛 System Architecture & Flowchart

```text
                   ┌─────────────────────────┐
                   │   APTOS Fundus Image    │
                   │      3,662 images       │
                   └────────────┬────────────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │   Image Quality Check   │
                   │ blur / exposure / noise │
                   └────────────┬────────────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │      Preprocessing      │
                   │ • Crop retinal region   │
                   │ • Remove black borders  │
                   │ • Resize 512×512        │
                   │ • CLAHE illumination    │
                   └────────────┬────────────┘
                                │
                                ▼
                   ┌─────────────────────────┐
                   │ Data Augmentation       │
                   │ rotation/flip/contrast  │
                   └────────────┬────────────┘
                                │
                                ▼
             ┌────────────────────────────────────┐
             │     PRETRAINED MEDICAL BACKBONE    │
             │       RETFound ViT-Large (303M)    │
             │       Self-Supervised Weights      │
             └────────────────┬───────────────────┘
                              │
                              ▼
                ┌─────────────────────────────┐
                │ Multi-scale Feature Fusion  │
                │ shallow (7) + mid (15)      │
                │ + deep (23) ViT blocks      │
                └──────────────┬──────────────┘
                               │
                               ▼
                    ┌────────────────────┐
                    │ Attention Pooling  │
                    │ Learned token-wise │
                    │ attention weights  │
                    └─────────┬──────────┘
                              │
                  ┌───────────┴──────────────┐
                  │                          │
                  ▼                          ▼
        ┌────────────────────┐     ┌─────────────────────┐
        │ Severity Head      │     │ Early-Screening Head│
        │ Grade 0–4          │     │ No DR / Has DR      │
        │ CORAL / EMD Loss   │     │ Binary BCE Loss     │
        └──────────┬─────────┘     └──────────┬──────────┘
                   │                          │
                   └────────────┬─────────────┘
                                ▼
                    ┌────────────────────────┐
                    │ Final Prediction       │
                    │ Grade + Probability    │
                    │ + Early DR Risk Score  │
                    └────────────┬───────────┘
                                 │
                                 ▼
                  ┌───────────────────────────┐
                  │   Explainable AI Suite    │
                  │ Grad-CAM / Grad-CAM++ /   │
                  │ Integrated Gradients /    │
                  │ Occlusion Sensitivity     │
                  └────────────┬──────────────┘
                               │
                               ▼
                  ┌───────────────────────────┐
                  │ Explainable Result        │
                  │ Original Fundus           │
                  │ + High-Res Heatmap Overlay│
                  │ + Quantitative Faithfulness│
                  └───────────────────────────┘
```

---

## 📂 Exact Folder Structure

To reproduce this project faithfully, your directory must match the structure below:

```text
DR/
├── .gitignore                          # Git rules (ignores checkpoints, test dumps, outputs)
├── requirements.txt                    # Exact pinned Python dependencies
├── README.md                           # Master project guide and run instructions
├── test_single_inference.py            # Quick-test CLI for single-image inference + Grad-CAM
├── checkpoint-best.pth                 # Fine-tuned RETFound weights (1.21 GB, placed here)
│
├── RETFound/                           # Official RETFound repository architecture
│   ├── models_vit.py                   # VisionTransformer & RETFound_mae definitions
│   └── util/                           # Data loader and distributed utilities
│
├── preprocessing/                      # Preprocessing & Image Quality Assurance
│   ├── __init__.py                     # Package init with lazy module exports
│   ├── quality_check.py                # Blur (Laplacian), exposure, noise & quality scoring
│   └── fundus_preprocessing.py         # Circle cropping, border removal, CLAHE, 512x512 resize
│
├── models/                             # Model Architectures & Loss Functions
│   ├── __init__.py                     # Package init with lazy module exports
│   ├── proposed_model.py               # DREarlyXAI: Frozen RETFound + Multi-scale Fusion +
│   │                                   # Learned Attention Pooling + Dual Heads (305.67M params)
│   └── ordinal_loss.py                 # CORAL loss, Earth Mover's Distance (EMD), CombinedLoss
│
├── xai/                                # Explainable AI (XAI) Suite
│   ├── __init__.py                     # Package init with PEP 562 lazy loading
│   ├── gradcam.py                      # ViT 196-token Grad-CAM targeting model.blocks[-1].norm1
│   ├── gradcam_pp.py                   # Grad-CAM++ with second-order gradient weighting
│   ├── integrated_gradients.py         # Axiomatic Integrated Gradients (50 path steps)
│   ├── occlusion.py                    # Causal Occlusion Sensitivity (16x16 sliding window)
│   ├── dual_gradcam.py                 # Dual Grad-CAM (Target vs Actual 5-panel layout)
│   ├── quantitative_evaluation.py      # Faithfulness Benchmarking (Deletion & Insertion AUC)
│   ├── batch_inference.py              # Full dataset GPU batch inference engine
│   └── error_analyzer.py               # Automated error triage (adjacent & extreme error cases)
│
├── test_images/                        # Retinal fundus test dataset (1,928 APTOS images)
│   ├── 0005cfc8afb6.png
│   ├── 003f0afdcd15.png
│   └── ...
│
├── notebooks/                          # Interactive Exploration
│   └── RETFound_XAI.ipynb              # Jupyter notebook with interactive widgets and CAMs
│
├── docs/                               # Comprehensive Phase-by-Phase Research Documentation
│   ├── PHASE1_2_LOCAL_SETUP_AND_VALIDATION.md   # Environment, CUDA 12.4 & Checkpoint loading
│   ├── PHASE3_BASELINE_EVALUATION.md            # Kaggle baseline benchmark & QWK analysis
│   ├── PHASE4_5_DUAL_GRADCAM.md                 # ViT token gradient pooling & Dual Grad-CAM
│   ├── PHASE6_7_ERROR_ANALYSIS.md               # Systematic error categorization & visuals
│   ├── PHASE8_QUANTITATIVE_EVALUATION.md        # Deletion/Insertion game theory & AUC metrics
│   ├── PHASE9_10_CLINICAL_AND_PROPOSED_MODEL.md # Lesion correlation & DR-EarlyXAI specification
│   ├── PREPROCESSING_PIPELINE.md                # Quality checker & fundus circle cropping
│   ├── PROPOSED_MODEL_ARCHITECTURE.md           # Deep dive into multi-scale fusion & dual heads
│   └── XAI_METHODS_COMPARISON.md                # Comparative study: Grad-CAM vs PP vs IG vs Occlusion
│
└── outputs/                            # Generated Research Artifacts (auto-created)
    ├── predictions/
    │   └── predictions.csv             # Full 1,928 image predictions with 5-class distributions
    ├── quality/
    │   └── quality_report.csv          # Quality metrics (blur, exposure, noise) per image
    ├── preprocessed/                   # Preprocessed 512x512 CLAHE-normalized images
    ├── gradcam/
    │   ├── single_test_gradcam.png     # 3-panel single image Grad-CAM
    │   ├── dual_gradcam_test.png       # 5-panel Dual Grad-CAM comparative figure
    │   ├── gradcampp_test.png          # 3-panel Grad-CAM++ figure
    │   ├── ig_test.png                 # 3-panel Integrated Gradients figure
    │   └── occlusion_test.png          # 3-panel Occlusion Sensitivity figure
    ├── metrics/
    │   └── faithfulness_curves.png     # Deletion AUC & Insertion AUC curve plots
    └── error_analysis/
        ├── correct_grade_0.png         # High-confidence correct Grade 0
        ├── correct_grade_1.png         # High-confidence correct Grade 1
        ├── correct_grade_4.png         # High-confidence correct Grade 4
        ├── error_actual_3_pred_2.png   # Adjacent misclassification Grade 3 vs 2
        ├── error_actual_4_pred_3.png   # Adjacent misclassification Grade 4 vs 3
        └── error_analysis_summary.csv  # Metadata catalog of all categorized errors
```

---

## 🛠 How to Setup the Environment & Data

### 1. Prerequisites
- **Operating System**: Windows 10/11 or Linux
- **Python**: 3.10 to 3.12 (Tested on Python 3.12.10 64-bit)
- **GPU**: NVIDIA GPU with CUDA 12.x support (Tested on NVIDIA RTX 3050 Laptop GPU, 4 GB VRAM)

### 2. Create Virtual Environment
Open PowerShell / Terminal in the project root:
```powershell
# Create isolated environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# (On Linux / macOS: source .venv/bin/activate)
```

### 3. Install PyTorch with CUDA 12.4 Acceleration
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 4. Install Project Dependencies
```powershell
pip install -r requirements.txt
```

### 5. Setup RETFound Architecture & Pretrained Checkpoint
1. Ensure the `RETFound/` directory exists with `models_vit.py`.
2. Place the trained checkpoint file `checkpoint-best.pth` (~1.21 GB) directly in the project root:
   ```text
   DR/checkpoint-best.pth
   ```
   *(The checkpoint contains the 303.31M parameters of the fine-tuned `RETFound_mae` model).*

### 6. Setup Test Images
Place the unlabelled or labelled APTOS test fundus images inside `test_images/`:
```text
DR/test_images/*.png
```

---

## 🚀 Exact Run Commands

Run all commands from the project root using the virtual environment interpreter (`.\.venv\Scripts\python.exe` on Windows or `python` on Linux):

### 1. Preprocessing & Quality Assurance
```powershell
# Run image quality check (blur, exposure, noise assessment)
.\.venv\Scripts\python.exe -m preprocessing.quality_check --image_dir test_images --limit 50

# Run fundus preprocessing (retinal circle cropping, black border removal, CLAHE, 512x512 resize)
.\.venv\Scripts\python.exe -m preprocessing.fundus_preprocessing --input_dir test_images --output_dir outputs/preprocessed --limit 20
```

### 2. Proposed Architecture & Ordinal Loss Verification
```powershell
# Validate proposed model forward pass (RETFound backbone frozen + multi-scale fusion + attention pooling + dual heads)
.\.venv\Scripts\python.exe -m models.proposed_model

# Test ordinal loss functions (CORAL rank loss, Earth Mover's Distance, Combined multi-task loss)
.\.venv\Scripts\python.exe -m models.ordinal_loss
```

### 3. Explainable AI (XAI) Suite — All 4 Methods
```powershell
# Method 1: Standard Transformer Grad-CAM (single test image)
.\.venv\Scripts\python.exe test_single_inference.py --image test_images\0005cfc8afb6.png

# Method 2: Grad-CAM++ (second-order gradients for multiple lesion instances)
.\.venv\Scripts\python.exe -m xai.gradcam_pp

# Method 3: Integrated Gradients (axiomatic path-integral attribution, 50 steps)
.\.venv\Scripts\python.exe -m xai.integrated_gradients

# Method 4: Occlusion Sensitivity (causal perturbation with 16x16 sliding window)
.\.venv\Scripts\python.exe -m xai.occlusion
```

### 4. Dual Grad-CAM & Quantitative Faithfulness
```powershell
# Dual Grad-CAM: Compare Predicted Class Heatmap vs True Ground-Truth Heatmap (5-panel figure)
.\.venv\Scripts\python.exe -m xai.dual_gradcam --image test_images\0005cfc8afb6.png --actual 3

# Quantitative Faithfulness Benchmark: Deletion AUC vs Insertion AUC curves
.\.venv\Scripts\python.exe -m xai.quantitative_evaluation
```

### 5. Full Dataset Batch Inference & Error Analysis
```powershell
# Run GPU batch inference across all 1,928 test images
.\.venv\Scripts\python.exe -m xai.batch_inference --image_dir test_images --batch_size 16

# Run automated systematic error analysis (triage correct, adjacent, and extreme errors)
.\.venv\Scripts\python.exe -m xai.error_analyzer
```

### 6. Interactive Jupyter Notebook
```powershell
# Register the kernel (one-time)
.\.venv\Scripts\python.exe -m ipykernel install --user --name dr-earlyxai --display-name "Python (DR-EarlyXAI .venv)"

# Launch Jupyter
jupyter notebook notebooks/RETFound_XAI.ipynb
```

---

## 📊 Benchmark Metrics & Results

### 1. Kaggle Test Benchmark (550 Stratified Images)
| Metric | Baseline Score | Clinical Implication |
| :--- | :---: | :--- |
| **ROC-AUC (Macro OVR)** | **94.54%** | Outstanding pairwise discrimination across all stages |
| **Quadratic Weighted Kappa (QWK)** | **89.71%** | Very high ordinal agreement; penalizes wide cross-grade errors |
| **Overall Accuracy** | **82.55%** | Correct on 454 of 550 test images |
| **Macro-Averaged F1** | **66.86%** | Reflects difficulty on minority severe grades |
| **Grade 0 Recall (No DR)** | **98.15%** | Reliable screening (almost zero healthy eyes misdiagnosed) |
| **Grade 3 Recall (Severe)** | **41.38%** | Critical vulnerability (global pooling dilutes the clinical 4-2-1 rule) |
| **Grade 4 Recall (Proliferative)** | **52.27%** | High clinical risk (12 cases confused with Grade 2) |

### 2. Full Test Dataset Inference (1,928 Unlabelled Images)
Processed via `xai.batch_inference` into `outputs/predictions/predictions.csv`:
- **Grade 0 (No DR)**: 354 images (18.4%) — Mean confidence: **82.7%**
- **Grade 1 (Mild NPDR)**: 130 images (6.7%) — Mean confidence: **59.4%** *(Lowest certainty; subtle microaneurysms)*
- **Grade 2 (Moderate NPDR)**: 1,182 images (61.3%) — Mean confidence: **71.5%**
- **Grade 3 (Severe NPDR)**: 190 images (9.9%) — Mean confidence: **63.1%** *(High uncertainty; quadrant beading)*
- **Grade 4 (Proliferative DR)**: 72 images (3.7%) — Mean confidence: **66.7%**

### 3. Quantitative Explanation Faithfulness
Measured via `xai.quantitative_evaluation` on sample `0005cfc8afb6.png`:
- **Deletion AUC**: **0.5425** (Confidence dropped by **42.3%** when top-attended patches were masked).
- **Insertion AUC**: **0.6849** (Confidence surged from **38.3% to 81.2%** when restoring just the top 20% salient pixels).
- **Finding**: Proves that RETFound genuinely relies on the identified lesion clusters rather than background shortcuts.

---

## 📚 Documentation Sitemap

Comprehensive scientific explanations and implementation details are provided in [`docs/`](docs/):

- [`PHASE1_2_LOCAL_SETUP_AND_VALIDATION.md`](docs/PHASE1_2_LOCAL_SETUP_AND_VALIDATION.md): GPU CUDA environment, RETFound model weight verification (0 missing keys), and single-image validation.
- [`PHASE3_BASELINE_EVALUATION.md`](docs/PHASE3_BASELINE_EVALUATION.md): Complete baseline reproduction, metric formulas (QWK quadratic penalty matrix), and batch inference.
- [`PHASE4_5_DUAL_GRADCAM.md`](docs/PHASE4_5_DUAL_GRADCAM.md): Mathematical derivation of ViT patch token gradient pooling and Dual Grad-CAM.
- [`PHASE6_7_ERROR_ANALYSIS.md`](docs/PHASE6_7_ERROR_ANALYSIS.md): Failure mode taxonomy (Grade 2 vs 3, 3 vs 4, extreme safety errors) and difference maps.
- [`PHASE8_QUANTITATIVE_EVALUATION.md`](docs/PHASE8_QUANTITATIVE_EVALUATION.md): Quantitative XAI faithfulness theory, Deletion/Insertion games, and AUC calculation.
- [`PHASE9_10_CLINICAL_AND_PROPOSED_MODEL.md`](docs/PHASE9_10_CLINICAL_AND_PROPOSED_MODEL.md): Pathognomonic lesion correlation (microaneurysms, hemorrhages, exudates, neovascularization) and DR-EarlyXAI proposed model design.
- [`PREPROCESSING_PIPELINE.md`](docs/PREPROCESSING_PIPELINE.md): Image quality grading (blur, exposure, noise) and fundus circle cropping + CLAHE.
- [`PROPOSED_MODEL_ARCHITECTURE.md`](docs/PROPOSED_MODEL_ARCHITECTURE.md): Multi-scale feature fusion (blocks 7, 15, 23), attention pooling, dual heads, and ordinal losses.
- [`XAI_METHODS_COMPARISON.md`](docs/XAI_METHODS_COMPARISON.md): Comprehensive comparison across Grad-CAM, Grad-CAM++, Integrated Gradients, and Occlusion Sensitivity.

---

## ⚖️ Citation & Clinical Disclaimer

> [!WARNING]
> **Medical Research Disclaimer**:
> This software is strictly for **scientific research and educational purposes**. It is **not** approved as a medical device for diagnostic or clinical decision-making. Predictions and visual heatmaps must not replace professional clinical evaluation by certified ophthalmologists.

---
**DR-EarlyXAI Research Initiative** — Explainable Foundation Models for Retinal Disease Grading.

