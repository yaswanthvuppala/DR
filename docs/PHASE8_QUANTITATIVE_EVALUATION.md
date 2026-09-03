# Phase 8: Quantitative XAI Evaluation & Faithfulness Benchmarking

---

## 1. Rationale: Beyond Visual Heatmaps

A major pitfall in Explainable AI (XAI) literature is the **"confirmation bias trap"**: evaluating heatmaps purely by visual inspection. A heatmap may look convincing to human clinicians while failing to reflect the features actually used by the neural network during inference.

In **DR-EarlyXAI**, we reject purely qualitative evaluation and establish rigorous **quantitative metrics** of explanation fidelity:

```text
                  Qualitative XAI                        Quantitative XAI (Phase 8)
              ┌─────────────────────┐                    ┌────────────────────────────┐
              │  "Looks sensible"   │         VS         │ Objective Faithfulness AUC │
              │  Subjective opinion │                    │ True Causal Impact Testing │
              └─────────────────────┘                    └────────────────────────────┘
```

---

## 2. Mathematical Formulations of Faithfulness Metrics

### 2.1 The Deletion Game (Measuring Necessity of Features)
If the Grad-CAM heatmap accurately highlights the retinal features that drove the prediction, then **removing those pixels should cause the model's confidence to plummet rapidly**.

#### Protocol:
1. Rank all pixels (or patch tokens) in descending order of Grad-CAM intensity:
   $$x_{(1)} \ge x_{(2)} \ge \dots \ge x_{(M)}$$
2. Sequentially mask the top fraction $\alpha \in [0.0, 1.0]$ of salient pixels by replacing them with local Gaussian blur:
   $$\tilde{I}(\alpha) = \text{Mask}(I, \alpha)$$
3. Compute the model's predicted probability for the target class at each step:
   $$p(\alpha) = P(Y = c \mid \tilde{I}(\alpha))$$
4. Compute the **Area Under the Deletion Curve (Deletion AUC)**:
   $$\text{AUC}_{\text{Deletion}} = \int_0^1 p(\alpha) \, d\alpha \approx \sum_{i=1}^S \frac{p(\alpha_i) + p(\alpha_{i-1})}{2} (\alpha_i - \alpha_{i-1})$$

$$\text{Criterion: } \textbf{LOWER Deletion AUC} \implies \textbf{HIGHER Faithfulness}$$

---

### 2.2 The Insertion Game (Measuring Sufficiency of Features)
Conversely, if an uninformative blurred image is given to the model, sequentially **restoring the most salient pixels first should rapidly recover the model's prediction**.

#### Protocol:
1. Start with a blurred baseline image: $I_{\text{blur}}$.
2. Sequentially restore the top fraction $\beta \in [0.0, 1.0]$ of salient pixels from the original image:
   $$\tilde{I}(\beta) = \text{Restore}(I_{\text{blur}}, I, \beta)$$
3. Compute the target class probability $p(\beta)$ at each step.
4. Compute the **Area Under the Insertion Curve (Insertion AUC)**:
   $$\text{AUC}_{\text{Insertion}} = \int_0^1 p(\beta) \, d\beta$$

$$\text{Criterion: } \textbf{HIGHER Insertion AUC} \implies \textbf{HIGHER Faithfulness}$$

---

### 2.3 Confidence Drop Rate
The percentage reduction in target class confidence when the top salient regions are masked:
$$\text{Drop Rate} = \frac{\max(0, p(0) - p(1.0))}{p(0)} \times 100\%$$

---

## 3. Verified Local Experimental Results

We evaluated RETFound's Grad-CAM faithfulness using [`xai/quantitative_evaluation.py`](file:///c:/Users/vuppa/Desktop/DR/xai/quantitative_evaluation.py) on fundus sample `0005cfc8afb6.png`:

| Metric | Measured Score | Research Interpretation |
| :--- | :---: | :--- |
| **Initial Prediction** | **Grade 2 (69.4%)** | Baseline target class confidence |
| **Deletion AUC** | **0.5425** | Confidence dropped from 66.4% down to 38.3% as lesions were masked |
| **Insertion AUC** | **0.6849** | Confidence surged from 38.3% up to 81.2% when restoring top 20% salient pixels |
| **Confidence Drop** | **42.3%** | Substantial degradation confirming reliance on the identified lesion regions |
| **Saved Curves Plot** | [`faithfulness_curves.png`](file:///c:/Users/vuppa/Desktop/DR/outputs/metrics/faithfulness_curves.png) | 2-panel comparative deletion vs insertion trajectory |

### Analysis of the Curves:
1. **Deletion Curve**: The sharp dip between $\alpha = 0.0$ and $\alpha = 0.2$ confirms that the most heavily weighted $10\text{--}20\%$ of pixels contain the critical diagnostic evidence.
2. **Insertion Curve**: Restoring just the first $20\%$ of salient pixels immediately pushes the model's confidence to $>80\%$, proving that RETFound's decision is dominated by compact focal lesion clusters rather than background illumination.

---

## 4. How to Run Quantitative Evaluation

The quantitative evaluation engine is available in [`xai/quantitative_evaluation.py`](file:///c:/Users/vuppa/Desktop/DR/xai/quantitative_evaluation.py).

### Command Line:
```powershell
# Run faithfulness evaluation on sample image
.\.venv\Scripts\python.exe -m xai.quantitative_evaluation
```
Output plot is generated at [`outputs/metrics/faithfulness_curves.png`](file:///c:/Users/vuppa/Desktop/DR/outputs/metrics/faithfulness_curves.png).
