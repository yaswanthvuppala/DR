# Proposed DR-EarlyXAI Model Architecture

---

## 1. Architecture Overview

The proposed DR-EarlyXAI model wraps the **frozen RETFound ViT-Large backbone** and adds three targeted architectural improvements to address the baseline's clinical failure modes:

```text
                     Input Retinal Fundus (224×224)
                                │
                                ▼
              ┌──────────────────────────────────────┐
              │    RETFound ViT-Large (FROZEN)        │
              │    303.31M parameters (all frozen)    │
              │    24 Transformer Blocks               │
              │                                        │
              │    Extract 3 feature scales:           │
              │    • blocks[7]  → shallow features    │
              │    • blocks[15] → mid-level features  │
              │    • blocks[23] → deep features       │
              │    Each: [B, 196, 1024]                │
              └──────────────────┬───────────────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────────┐
              │    Multi-Scale Feature Fusion         │
              │    3 × Linear(1024, 512) projections  │
              │    Concat → [B, 196, 1536]            │
              │    Linear(1536, 512) → LayerNorm      │
              │    → GELU → [B, 196, 512]             │
              └──────────────────┬───────────────────┘
                                 │
                                 ▼
              ┌──────────────────────────────────────┐
              │    Learned Attention Pooling           │
              │    Linear(512, 1) per token            │
              │    Softmax over 196 tokens             │
              │    Weighted sum → [B, 512]             │
              │    (replaces naive mean pooling)        │
              └──────────────────┬───────────────────┘
                                 │
                       ┌─────────┴──────────┐
                       ▼                    ▼
              ┌────────────────┐   ┌────────────────────┐
              │ Severity Head  │   │ Screening Head     │
              │ Dropout(0.3)   │   │ Dropout(0.3)       │
              │ Linear(512, 5) │   │ Linear(512, 1)     │
              │ Grade 0–4      │   │ Sigmoid → DR Risk  │
              └────────────────┘   └────────────────────┘
```

---

## 2. Parameter Budget

| Component | Parameters | Trainable? |
| :--- | :---: | :---: |
| RETFound ViT-Large backbone | 303,306,757 | ❄️ **Frozen** |
| Multi-Scale Feature Fusion | 2,098,688 | ✅ **Yes** |
| Attention Pooling | 513 | ✅ **Yes** |
| Severity Head (5-class) | 2,565 | ✅ **Yes** |
| Screening Head (binary) | 513 | ✅ **Yes** |
| **Total** | **305,672,716** | |
| **Trainable Only** | **2,365,959** | ✅ **0.77% of total** |

> Only **2.37M parameters** (0.77%) are trainable. The frozen RETFound backbone provides medical foundation knowledge while the new modules learn DR-specific refinements.

---

## 3. Why Multi-Scale Feature Fusion Helps

RETFound baseline uses ONLY the final block output. By extracting features from 3 different depths:

| Block | Depth | What It Captures | Clinical Value |
| :---: | :--- | :--- | :--- |
| `blocks[7]` | Shallow (⅓) | Low-level edges, textures, vessel boundaries | Microaneurysm detection (Grade 1) |
| `blocks[15]` | Middle (⅔) | Mid-level patterns, lesion shapes | Hemorrhage and exudate clusters (Grade 2–3) |
| `blocks[23]` | Deep (final) | High-level semantic understanding | Disease severity gestalt (Grade 3–4) |

---

## 4. Why Attention Pooling Replaces Mean Pooling

**Baseline RETFound** uses naive global average pooling:
```python
x[:, 1:, :].mean(dim=1)  # Average all 196 tokens equally
```
This treats the optic disc, macula, peripheral retina, and background all with equal weight.

**Proposed Attention Pooling** learns which tokens matter:
$$\text{attention}(i) = \frac{\exp(W \cdot z_i)}{\sum_{j=1}^{196} \exp(W \cdot z_j)}$$
$$\text{output} = \sum_{i=1}^{196} \text{attention}(i) \cdot z_i$$

This allows the model to focus on lesion-bearing patches while downweighting uninformative background tokens.

---

## 5. Ordinal Loss Functions

### CORAL Loss (Consistent Rank Logits)
Instead of treating DR grades as independent categories, CORAL enforces ordinal consistency:
- 4 binary thresholds: $P(Y > 0)$, $P(Y > 1)$, $P(Y > 2)$, $P(Y > 3)$
- Penalty naturally increases with the distance between predicted and true grade

### Earth Mover's Distance (EMD) Loss
Treats the probability distribution over grades as a discrete histogram and minimizes the "transportation cost" to move predicted mass to the true grade.

### Combined Loss
$$\mathcal{L}_{\text{total}} = \underbrace{1.0 \cdot \mathcal{L}_{\text{CE}}}_{\text{Severity}} + \underbrace{0.5 \cdot \mathcal{L}_{\text{BCE}}}_{\text{Screening}} + \underbrace{0.3 \cdot \mathcal{L}_{\text{EMD}}}_{\text{Ordinal penalty}}$$

---

## 6. How to Run

```powershell
# Test the proposed model architecture (forward pass validation)
.\.venv\Scripts\python.exe -m models.proposed_model

# Test the ordinal loss functions with synthetic data
.\.venv\Scripts\python.exe -m models.ordinal_loss
```

### Implementation Files

| File | Key Classes | Purpose |
| :--- | :--- | :--- |
| [`models/proposed_model.py`](file:///c:/Users/vuppa/Desktop/DR/models/proposed_model.py) | `DREarlyXAI`, `load_proposed_model` | Complete proposed architecture |
| [`models/ordinal_loss.py`](file:///c:/Users/vuppa/Desktop/DR/models/ordinal_loss.py) | `CORALLoss`, `EMDLoss`, `CombinedLoss` | Ordinal-aware training objectives |
