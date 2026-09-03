"""XAI package for DR-EarlyXAI."""

# Lazy module attribute access (PEP 562) to prevent runpy RuntimeWarnings
# when running scripts via `python -m xai.<module>`
def __getattr__(name):
    if name in ("RETFoundGradCAM", "load_model_input", "load_visualization_image", "save_single_gradcam_figure"):
        from . import gradcam
        return getattr(gradcam, name)
    elif name in ("DualGradCAM", "save_dual_gradcam_figure"):
        from . import dual_gradcam
        return getattr(dual_gradcam, name)
    elif name in ("GradCAMPP", "save_gradcampp_figure"):
        from . import gradcam_pp
        return getattr(gradcam_pp, name)
    elif name in ("IntegratedGradients", "save_ig_figure"):
        from . import integrated_gradients
        return getattr(integrated_gradients, name)
    elif name in ("OcclusionSensitivity", "save_occlusion_figure"):
        from . import occlusion
        return getattr(occlusion, name)
    elif name in ("XAIEvaluator", "save_faithfulness_figure"):
        from . import quantitative_evaluation
        return getattr(quantitative_evaluation, name)
    elif name == "ErrorAnalyzer":
        from . import error_analyzer
        return getattr(error_analyzer, name)
    elif name == "run_batch_inference":
        from . import batch_inference
        return getattr(batch_inference, name)
    raise AttributeError(f"module 'xai' has no attribute '{name}'")

__all__ = [
    "RETFoundGradCAM",
    "DualGradCAM",
    "GradCAMPP",
    "IntegratedGradients",
    "OcclusionSensitivity",
    "XAIEvaluator",
    "ErrorAnalyzer",
    "run_batch_inference",
    "load_model_input",
    "load_visualization_image",
    "save_single_gradcam_figure",
    "save_dual_gradcam_figure",
    "save_gradcampp_figure",
    "save_ig_figure",
    "save_occlusion_figure",
    "save_faithfulness_figure"
]
