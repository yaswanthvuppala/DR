# Preprocessing Pipeline: Image Quality Check & Fundus Preprocessing

---

## 1. Image Quality Assessment Module

Before DR severity classification, retinal fundus images must pass quality checks. Low-quality images (blurred, overexposed, noisy) can produce unreliable predictions and misleading XAI heatmaps.

### Quality Metrics Computed

| Metric | Method | Threshold | Clinical Rationale |
| :--- | :--- | :---: | :--- |
| **Blur Score** | Laplacian variance of grayscale image | $\geq 100$ | Blurry images obscure microaneurysms and fine vascular details |
| **Exposure** | Mean grayscale brightness | $40 \leq \mu \leq 220$ | Underexposed images hide hemorrhages; overexposed ones wash out exudates |
| **Noise Score** | Median Absolute Deviation (MAD) of Laplacian | Lower is better | High sensor noise mimics lesion textures and confuses the model |
| **Overall Quality** | Weighted composite (0–100) | $\geq 50$ | Combined acceptability metric |

### Verified Local Results (10-Image Sample)

Only **1 out of 10** sample APTOS images passed the quality threshold. This is expected — APTOS images were captured under real clinical conditions with varying equipment quality.

### How to Run

```powershell
# Check quality of all test images
.\.venv\Scripts\python.exe -m preprocessing.quality_check --image_dir test_images --limit 100

# Output: outputs/quality/quality_report.csv
```

---

## 2. Fundus Image Preprocessing Pipeline

### Processing Steps

```text
Raw APTOS Image (variable size, black borders)
        │
        ▼
┌──────────────────────────────┐
│ 1. Retinal Circle Detection  │  Threshold grayscale → find largest contour
│    & Cropping                │  → minimum enclosing circle → crop to bounding box
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ 2. Black Border Removal      │  Circular mask applied → black regions outside
│                              │  the fundus circle are zeroed out
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ 3. CLAHE Illumination        │  Convert to LAB color space → apply CLAHE
│    Normalization             │  (clip_limit=2.0, grid=8×8) on L channel
│                              │  → convert back to RGB
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ 4. Resize to 512×512         │  Bilinear interpolation to standard size
└──────────────────────────────┘
```

### How to Run

```powershell
# Preprocess sample images
.\.venv\Scripts\python.exe -m preprocessing.fundus_preprocessing --input_dir test_images --output_dir outputs/preprocessed --limit 20

# Output: outputs/preprocessed/<image_name>.png (512×512, border-removed, illumination-normalized)
```

### Implementation Files

| File | Class | Purpose |
| :--- | :--- | :--- |
| [`preprocessing/quality_check.py`](file:///c:/Users/vuppa/Desktop/DR/preprocessing/quality_check.py) | `QualityChecker` | Blur, exposure, noise assessment |
| [`preprocessing/fundus_preprocessing.py`](file:///c:/Users/vuppa/Desktop/DR/preprocessing/fundus_preprocessing.py) | `FundusPreprocessor` | Circle crop, border removal, CLAHE, resize |
