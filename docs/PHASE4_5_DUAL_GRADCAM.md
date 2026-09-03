# Phase 4 & 5: Transformer Grad-CAM and Dual Grad-CAM Implementation

---

## 1. Motivation: Why Standard Grad-CAM Fails on Vision Transformers

Standard Grad-CAM was designed for Convolutional Neural Networks (CNNs) with 2D spatial feature maps:
$$A \in \mathbb{R}^{C \times H \times W}$$
In contrast, Vision Transformers (ViT) such as **RETFound** process images as a 1D sequence of flattened patch tokens:
$$Z \in \mathbb{R}^{B \times (1 + N) \times D}$$
Where:
- $B = 1$ is the batch size.
- Token $0$ is the special `[CLS]` token representing global image representation.
- $N = (224 / 16)^2 = 196$ are the spatial patch tokens corresponding to non-overlapping $16 \times 16$ pixel regions.
- $D = 1024$ is the hidden embedding dimension.

To produce meaningful visual explanations, we must:
1. Target the final Transformer layer normalization: `model.blocks[-1].norm1`.
2. Extract the activation tensor $A \in \mathbb{R}^{1 \times 197 \times 1024}$.
3. **Discard the CLS token** ($A_{[:, 0, :]}$), isolating the $196$ patch tokens ($A_{[:, 1:, :]}$).
4. Compute the gradient of the target class score $y^c$ with respect to the patch activations:
   $$G = \frac{\partial y^c}{\partial A_{[:, 1:, :]}} \in \mathbb{R}^{1 \times 196 \times 1024}$$
5. Compute channel importance weights $\alpha_k^c$ by pooling gradients across all $196$ patch tokens:
   $$\alpha_k^c = \frac{1}{196} \sum_{i=1}^{196} G_{i, k}$$
6. Compute the spatially weighted sum followed by a Rectified Linear Unit (ReLU) to isolate positive contributions:
   $$L_{\text{CAM}}^c = \text{ReLU}\left(\sum_{k=1}^{1024} \alpha_k^c A_{[:, 1:, k]}\right) \in \mathbb{R}^{196}$$
7. Reshape the 196 tokens into a 2D spatial grid of $14 \times 14$:
   $$L_{\text{grid}}^c = \text{reshape}(L_{\text{CAM}}^c, (14, 14))$$
8. Bilinearly interpolate from $14 \times 14$ to the original input resolution of $224 \times 224$:
   $$L_{\text{upsampled}}^c = \text{BilinearInterpolate}(L_{\text{grid}}^c, (224, 224))$$
9. Min-Max normalize to $[0.0, 1.0]$.

---

## 2. Phase 5: The Dual Grad-CAM Concept

In real-world medical AI, visual explanations are most crucial **when the model is wrong**.
A single Grad-CAM only shows where the model looked when it made a mistake. It does NOT explain **why the correct diagnosis was missed**.

**Dual Grad-CAM** solves this by computing two simultaneous backward passes for any misclassified image:

$$\text{Dual Grad-CAM} = \begin{cases} 
\text{CAM}(\text{target} = \hat{y}_{\text{pred}}) & \text{"Why did the model predict class } \hat{y}_{\text{pred}}\text{?"} \\
\text{CAM}(\text{target} = y_{\text{true}}) & \text{"What features correspond to true class } y_{\text{true}}\text{?"} 
\end{cases}$$

### The Attention Difference Map ($\Delta_{\text{CAM}}$)
We quantify the attention shift between the mistaken class and the true class:
$$\Delta_{\text{CAM}} = \left| \text{CAM}(\hat{y}_{\text{pred}}) - \text{CAM}(y_{\text{true}}) \right|$$
Regions where $\Delta_{\text{CAM}}$ is large indicate **clinical conflict zones**—areas where the model's visual attention was hijacked by confounding lesions.

---

## 3. The 5-Panel Publication-Quality Visual Layout

Dual Grad-CAM produces a comprehensive 5-panel comparative layout:

```text
+-------------------+--------------------+--------------------+--------------------+--------------------+
|  Panel 1: Fundus  | Panel 2: Pred CAM  | Panel 3: Pred Over |  Panel 4: True CAM | Panel 5: True Over |
| Actual: Grade 3   | Target: Grade 2    | Overlay Grade 2    | Target: Grade 3    | Overlay Grade 3    |
| Pred  : Grade 2   | Conf: 69.4%        | (Moderate NPDR)    | Conf: 3.5%         | (Severe NPDR)      |
+-------------------+--------------------+--------------------+--------------------+--------------------+
```

### Verified Local Example:
In our local test on `test_images/0005cfc8afb6.png`:
- **Actual Grade**: Grade 3 (Severe)
- **Predicted Grade**: Grade 2 (Moderate) with 69.4% confidence (Actual class confidence: 3.5%)
- **Saved Figure**: [`outputs/gradcam/dual_gradcam_test.png`](file:///c:/Users/vuppa/Desktop/DR/outputs/gradcam/dual_gradcam_test.png)

#### Clinical Finding from Dual Grad-CAM:
* **Predicted Grade 2 CAM**: Attention is concentrated centrally around the dense circinate ring of hard exudates in the lower-right macula.
* **True Grade 3 CAM**: Attention shifts to the peripheral venules and lower retinal margin, searching for venous beading and extensive multi-quadrant hemorrhages required for Grade 3 criteria.
* **Conclusion**: RETFound misclassified the eye as Grade 2 because the visually striking exudate cluster overwhelmed the subtle venous changes.

---

## 4. How to Run Dual Grad-CAM Locally

Dual Grad-CAM is implemented in [`xai/dual_gradcam.py`](file:///c:/Users/vuppa/Desktop/DR/xai/dual_gradcam.py).

### Command Line:
```powershell
# Run on an image with a specified ground-truth grade
.\.venv\Scripts\python.exe -m xai.dual_gradcam `
    --image test_images/0005cfc8afb6.png `
    --actual 3 `
    --output outputs/gradcam/my_dual_gradcam.png
```

### Python API:
```python
from xai.dual_gradcam import DualGradCAM, save_dual_gradcam_figure
from xai.gradcam import load_model_input

dual_engine = DualGradCAM(model=model, target_layer=model.blocks[-1].norm1, device=device)
inp = load_model_input("test_images/sample.png", device)
res = dual_engine.generate_dual(inp, actual_class=3)

# Save 5-panel figure
save_dual_gradcam_figure("test_images/sample.png", res, "outputs/gradcam/dual.png")
```
