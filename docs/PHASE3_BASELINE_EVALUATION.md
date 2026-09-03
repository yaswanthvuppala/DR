# Phase 3: Baseline RETFound Reproduction & Evaluation Metrics

---

## 1. Overview & Research Context

Diabetic Retinopathy (DR) severity classification is inherently an **ordinal multi-class classification** problem. The target severity scale follows the International Clinical Diabetic Retinopathy (ICDR) standard:

| Grade | Clinical Designation | Pathological Hallmarks |
| :---: | :--- | :--- |
| **0** | No DR | Normal healthy retina, no vascular lesions |
| **1** | Mild Non-Proliferative DR (NPDR) | Microaneurysms only |
| **2** | Moderate NPDR | Multiple microaneurysms, blot hemorrhages, hard exudates, cotton-wool spots |
| **3** | Severe NPDR | $\ge 20$ intraretinal hemorrhages in all 4 quadrants, venous beading in $\ge 2$ quadrants, or IRMA in $\ge 1$ quadrant ("4-2-1 rule") |
| **4** | Proliferative DR (PDR) | Neovascularization (new fragile vessels), vitreous/preretinal hemorrhage, or retinal detachment |

---

## 2. Completed Baseline Benchmark (Kaggle Evaluation)

The baseline model was fully fine-tuned on the APTOS 2019 dataset using the RETFound Vision Transformer backbone. The final evaluation across the 550 test images yielded the following benchmark:

| Metric | Baseline Score | Clinical Interpretation |
| :--- | :---: | :--- |
| **ROC-AUC (Macro OVR)** | **94.54%** | Exceptional discriminative capacity across all pairwise thresholds |
| **Quadratic Weighted Kappa (QWK)** | **89.71%** | Very high ordinal agreement, heavily penalizing wide cross-grade errors |
| **Overall Accuracy** | **82.55%** | Correct classification on 454 of 550 images |
| **Macro-Averaged F1** | **66.86%** | Reflects grade-specific performance disparities |
| **Macro-Averaged Precision** | **68.14%** | Average precision across all 5 grades |
| **Macro-Averaged Recall** | **65.92%** | Average sensitivity across all 5 grades |
| **Average Precision (mAP)** | **69.02%** | Precision-Recall curve area across grades |
| **Hamming Loss** | **6.98%** | Fraction of misclassified labels |
| **Jaccard Index** | **53.71%** | Strict intersection-over-union metric |
| **Test Loss (Cross-Entropy)** | **0.4517** | Well-calibrated prediction loss |

---

## 3. Confusion Matrix Breakdown & Asymmetry Analysis

The confusion matrix from the 550 test images reveals critical clinical insights into the model's behavior:

```text
                  Predicted Grade
                0      1      2      3      4    | Total
Actual Grade 0 [266     4      1      0      0 ] |  271  (Recall: 98.15%)
Actual Grade 1 [  4    32     20      0      0 ] |   56  (Recall: 57.14%)
Actual Grade 2 [  3     9    121     10      7 ] |  150  (Recall: 80.67%)
Actual Grade 3 [  0     0      7     12     10 ] |   29  (Recall: 41.38%)
Actual Grade 4 [  1     1     12      7     23 ] |   44  (Recall: 52.27%)
-------------------------------------------------+-------
Total Pred:     274    46    161     29     40   |  550
```

### Key Diagnostic Findings:
1. **Near-Perfect Healthy Eye Screening (Grade 0 Recall = 98.15%)**:
   The model almost never misdiagnoses a healthy retina as severe DR (only 1 false positive as Grade 2, 4 as Grade 1, 0 as Grade 3/4).
2. **The "Grade 3 Vulnerability" (Recall = 41.38%)**:
   Grade 3 is by far the most difficult category. Out of 29 actual Grade 3 cases:
   - Only **12 are correctly identified**.
   - **7 are downgraded to Grade 2** (underestimating disease severity).
   - **10 are upgraded to Grade 4** (overestimating disease severity).
   Clinically, Grade 3 is defined by the subtle "4-2-1 rule" (quantifying venous beading and hemorrhages across quadrants), which is hard for global pooling ViTs without multi-scale quadrant-aware attention.
3. **Severe Grade Confusion (Grades 2 $\leftrightarrow$ 3 $\leftrightarrow$ 4)**:
   Notice that out of 44 Grade 4 (Proliferative) cases, **12 are downgraded to Grade 2** and **7 to Grade 3**. Missing Grade 4 is high risk because proliferative DR requires urgent laser photocoagulation or anti-VEGF injections.

---

## 4. Evaluation Metric Mathematical Formulations

### Quadratic Weighted Kappa (QWK)
The QWK metric evaluates agreement for ordinal classifications:
$$\kappa = 1 - \frac{\sum_{i=1}^k \sum_{j=1}^k w_{i,j} O_{i,j}}{\sum_{i=1}^k \sum_{j=1}^k w_{i,j} E_{i,j}}$$
Where the quadratic penalty weight is:
$$w_{i,j} = \frac{(i - j)^2}{(k - 1)^2}$$
An error of Grade $0 \to 1$ costs $(1-0)^2 / 16 = 0.0625$, whereas an error of Grade $0 \to 4$ costs $(4-0)^2 / 16 = 1.0$.
RETFound's QWK of **89.71%** indicates that most misclassifications are off by only $\pm 1$ grade.

---

## 5. Running the Local Batch Inference Engine

The batch inference engine is implemented in [`xai/batch_inference.py`](file:///c:/Users/vuppa/Desktop/DR/xai/batch_inference.py).

### Usage:
```powershell
# Run batch inference across all images in test_images/
.\.venv\Scripts\python.exe -m xai.batch_inference --image_dir test_images --batch_size 16

# Run a quick sanity subset on 25 images
.\.venv\Scripts\python.exe -m xai.batch_inference --image_dir test_images --limit 25
```
Output:
- CSV: [`outputs/predictions/predictions.csv`](file:///c:/Users/vuppa/Desktop/DR/outputs/predictions/predictions.csv) containing all image paths, predicted grades, confidences, and 5-class probability distributions.
- Confusion Matrix: [`outputs/metrics/confusion_matrix.png`](file:///c:/Users/vuppa/Desktop/DR/outputs/metrics/confusion_matrix.png) (when ground-truth subdirectories or labels are available).
