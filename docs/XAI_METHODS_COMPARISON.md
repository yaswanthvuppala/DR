# XAI Methods Comparison: Grad-CAM vs Grad-CAM++ vs Integrated Gradients vs Occlusion

---

## 1. Overview of All XAI Methods in DR-EarlyXAI

The project now implements **4 complementary explainability methods**, each offering a different perspective on model decision-making:

| Method | Mathematical Basis | What It Reveals | Computational Cost |
| :--- | :--- | :--- | :---: |
| **Grad-CAM** | First-order gradients, channel-wise pooling | Which spatial regions the model activates for a given class | Low (1 backward pass) |
| **Grad-CAM++** | Second/third-order gradients, alpha weighting | More precise localization, especially for multiple lesion instances | Low (1 backward pass) |
| **Integrated Gradients** | Path integral of gradients from baseline to input | Pixel-level attribution satisfying completeness axiom | High (50 forward+backward passes) |
| **Occlusion Sensitivity** | Perturbation-based (sliding gray patch) | Model-agnostic causal evidence — which regions causally affect the output | High (many forward passes) |

---

## 2. Mathematical Formulations

### 2.1 Grad-CAM (Selvaraju et al., 2017)
$$\alpha_k^c = \frac{1}{N}\sum_{i=1}^{N} \frac{\partial y^c}{\partial A_{i,k}}$$
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A_k\right)$$

### 2.2 Grad-CAM++ (Chattopadhyay et al., 2018)
$$\alpha_{ik}^c = \frac{\text{ReLU}\left(\frac{\partial^2 y^c}{\partial A_{i,k}^2}\right)}{2 \cdot \frac{\partial^2 y^c}{\partial A_{i,k}^2} + \sum_j A_{j,k} \cdot \frac{\partial^3 y^c}{\partial A_{i,k}^3} + \epsilon}$$
$$w_k^c = \sum_i \alpha_{ik}^c \cdot \text{ReLU}\left(\frac{\partial y^c}{\partial A_{i,k}}\right)$$

**Advantage over Grad-CAM**: Better handles multiple instances of the same lesion type (e.g., scattered microaneurysms across the retina).

### 2.3 Integrated Gradients (Sundararajan et al., 2017)
$$\text{IG}_i(x) = (x_i - x'_i) \times \int_0^1 \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} d\alpha$$

Satisfies the **Completeness Axiom**: $\sum_i \text{IG}_i(x) = F(x) - F(x')$

**Advantage**: Provides mathematically rigorous pixel-level attribution with axiomatic guarantees.

### 2.4 Occlusion Sensitivity (Zeiler & Fergus, 2014)
$$S(i, j) = P(Y=c \mid x) - P(Y=c \mid x_{\text{occluded at } (i,j)})$$

**Advantage**: Model-agnostic and perturbation-based — directly measures causal impact by physically blocking regions.

---

## 3. Verified GPU Test Results

All 4 methods were tested on the same image (`0005cfc8afb6.png`, Predicted: Grade 2, Confidence: 69.4%):

| Method | Output Figure | Key Observation |
| :--- | :--- | :--- |
| **Grad-CAM** | [`single_test_gradcam.png`](file:///c:/Users/vuppa/Desktop/DR/outputs/gradcam/single_test_gradcam.png) | Broad attention on hard exudate cluster in inferior macula |
| **Grad-CAM++** | [`gradcampp_test.png`](file:///c:/Users/vuppa/Desktop/DR/outputs/gradcam/gradcampp_test.png) | More focal hot-spots, better separating individual lesion clusters |
| **Integrated Gradients** | [`ig_test.png`](file:///c:/Users/vuppa/Desktop/DR/outputs/gradcam/ig_test.png) | Fine-grained pixel-level attribution showing patch-boundary structure |
| **Occlusion Sensitivity** | [`occlusion_test.png`](file:///c:/Users/vuppa/Desktop/DR/outputs/gradcam/occlusion_test.png) | Sparse, highly specific causal hot-spots where occluding drops confidence most |

---

## 4. How to Run Each Method

```powershell
# Grad-CAM (standard)
.\.venv\Scripts\python.exe test_single_inference.py --image test_images\0005cfc8afb6.png

# Grad-CAM++
.\.venv\Scripts\python.exe -m xai.gradcam_pp

# Integrated Gradients (50 interpolation steps)
.\.venv\Scripts\python.exe -m xai.integrated_gradients

# Occlusion Sensitivity (16×16 patch, stride 8)
.\.venv\Scripts\python.exe -m xai.occlusion
```

---

## 5. Implementation Files

| File | Class | Method |
| :--- | :--- | :--- |
| [`xai/gradcam.py`](file:///c:/Users/vuppa/Desktop/DR/xai/gradcam.py) | `RETFoundGradCAM` | Standard Grad-CAM |
| [`xai/gradcam_pp.py`](file:///c:/Users/vuppa/Desktop/DR/xai/gradcam_pp.py) | `GradCAMPP` | Grad-CAM++ |
| [`xai/integrated_gradients.py`](file:///c:/Users/vuppa/Desktop/DR/xai/integrated_gradients.py) | `IntegratedGradients` | Integrated Gradients |
| [`xai/occlusion.py`](file:///c:/Users/vuppa/Desktop/DR/xai/occlusion.py) | `OcclusionSensitivity` | Occlusion Sensitivity |
| [`xai/dual_gradcam.py`](file:///c:/Users/vuppa/Desktop/DR/xai/dual_gradcam.py) | `DualGradCAM` | Predicted vs Actual CAM comparison |
