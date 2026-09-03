"""
Systematic XAI Error Analyzer for RETFound (DR-EarlyXAI).
Analyzes failure modes across DR severity grades:
- High-confidence correct predictions (Grades 0-4)
- Difficult adjacent grade transitions (2<->3, 3<->4, 2<->4)
- Extreme clinical errors (0->3, 0->4, 4->0)
Generates research figures and failure mode summaries.
"""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from xai.gradcam import load_model_input
from xai.dual_gradcam import DualGradCAM, save_dual_gradcam_figure, CLASS_NAMES


class ErrorAnalyzer:
    """
    Automated error analyzer that isolates characteristic clinical error patterns
    and produces Dual Grad-CAM explanations for research comparison.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.dual_engine = DualGradCAM(model=model, target_layer=model.blocks[-1].norm1, device=device)

    def analyze_predictions_df(
        self,
        df: pd.DataFrame,
        output_dir: str = "outputs/error_analysis"
    ) -> pd.DataFrame:
        """
        Takes a predictions dataframe with columns:
        ['image_path', 'predicted_grade', 'confidence', 'ground_truth']
        and generates categorized Dual Grad-CAM figures.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        records = []

        if "ground_truth" not in df.columns:
            print("Note: DataFrame does not have 'ground_truth'. Analyzing highest and lowest confidence predictions.")
            for grade in range(5):
                sub = df[df["predicted_grade"] == grade].sort_values("confidence", ascending=False)
                if not sub.empty:
                    top_row = sub.iloc[0]
                    img = top_row["image_path"]
                    pred = int(top_row["predicted_grade"])
                    conf = float(top_row["confidence"])

                    inp = load_model_input(img, self.device)
                    dual_res = self.dual_engine.generate_dual(inp, actual_class=pred)
                    fig_name = f"grade_{grade}_pred_{pred}_conf_{conf:.2f}.png"
                    fig_path = str(out_path / fig_name)
                    save_dual_gradcam_figure(img, dual_res, fig_path)

                    records.append({
                        "category": "representative_prediction",
                        "image_path": img,
                        "actual": pred,
                        "predicted": pred,
                        "confidence": conf,
                        "figure_path": fig_path
                    })
            return pd.DataFrame(records)

        # 1. High-confidence correct predictions
        for grade in range(5):
            sub = df[(df["ground_truth"] == grade) & (df["predicted_grade"] == grade)].sort_values("confidence", ascending=False)
            if not sub.empty:
                row = sub.iloc[0]
                img = row["image_path"]
                inp = load_model_input(img, self.device)
                dual_res = self.dual_engine.generate_dual(inp, actual_class=grade)
                fig_path = str(out_path / f"correct_grade_{grade}.png")
                save_dual_gradcam_figure(img, dual_res, fig_path)
                records.append({
                    "category": "high_confidence_correct",
                    "image_path": img,
                    "actual": grade,
                    "predicted": grade,
                    "confidence": float(row["confidence"]),
                    "figure_path": fig_path
                })

        # 2. Key clinical confusion pairs
        critical_pairs = [
            (2, 3), (3, 2),  # Moderate <-> Severe
            (3, 4), (4, 3),  # Severe <-> Proliferative
            (2, 4), (4, 2),  # Moderate <-> Proliferative
            (0, 1), (1, 0),  # Early DR detection threshold
            (0, 3), (0, 4), (4, 0)  # Extreme safety-critical errors
        ]

        for actual, pred in critical_pairs:
            sub = df[(df["ground_truth"] == actual) & (df["predicted_grade"] == pred)].sort_values("confidence", ascending=False)
            if not sub.empty:
                row = sub.iloc[0]
                img = row["image_path"]
                inp = load_model_input(img, self.device)
                dual_res = self.dual_engine.generate_dual(inp, actual_class=actual)
                cat = "extreme_error" if abs(actual - pred) >= 3 else "adjacent_error"
                fig_path = str(out_path / f"error_actual_{actual}_pred_{pred}.png")
                save_dual_gradcam_figure(img, dual_res, fig_path)
                records.append({
                    "category": cat,
                    "image_path": img,
                    "actual": actual,
                    "predicted": pred,
                    "confidence": float(row["confidence"]),
                    "figure_path": fig_path
                })

        summary_df = pd.DataFrame(records)
        summary_csv = out_path / "error_analysis_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"Error analysis figures and summary saved to {out_path.resolve()}")
        return summary_df


if __name__ == "__main__":
    import sys
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root / "RETFound"))
    sys.path.insert(0, str(project_root))
    import models_vit as models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading RETFound for Error Analysis on {device}...")
    model = models.RETFound_mae(img_size=224, num_classes=5, drop_path_rate=0.2, global_pool=True)
    ckpt = torch.load("checkpoint-best.pth", map_location="cpu", weights_only=False)
    state = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    # Create demonstration cases
    test_imgs = sorted(list((project_root / "test_images").glob("*.png")))
    demo_df = pd.DataFrame({
        "image_path": [str(p) for p in test_imgs[:5]],
        "predicted_grade": [2, 1, 0, 3, 4],
        "confidence": [0.694, 0.752, 0.992, 0.810, 0.885],
        "ground_truth": [3, 1, 0, 4, 4]  # Shows 2->3 and 3->4 error cases
    })

    analyzer = ErrorAnalyzer(model=model, device=device)
    analyzer.analyze_predictions_df(demo_df)
