import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2

from xai.gradcam import load_model_input, load_visualization_image

class IntegratedGradients:
    """
    Integrated Gradients for Vision Transformers.
    
    Generates attributions by computing the path integral of gradients 
    along the straight line from a baseline image to the input image.
    Formula: IG(x) = (x - x') * Integral_0^1 ( grad(F(x' + alpha * (x - x'))) ) d(alpha)
    """
    def __init__(self, model, device, steps=50):
        self.model = model
        self.device = device
        self.steps = steps

    def generate(self, input_tensor, class_idx=None, baseline=None):
        """
        Generate Integrated Gradients attribution map.
        
        Args:
            input_tensor (torch.Tensor): Input image tensor [1, C, H, W]
            class_idx (int, optional): Target class index. Defaults to predicted class.
            baseline (torch.Tensor, optional): Baseline image tensor. If None, uses a zero tensor.
            
        Returns:
            dict: Contains attribution_map, logits, probabilities, confidence, predicted_class, target_class
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        
        if baseline is None:
            baseline = torch.zeros_like(input_tensor, device=self.device)
        else:
            baseline = baseline.to(self.device)

        # First pass to get predicted class if not specified
        with torch.inference_mode():
            orig_logits = self.model(input_tensor)
            probs = F.softmax(orig_logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()
            
            if class_idx is None:
                class_idx = pred_class

        alphas = torch.linspace(0, 1, self.steps, device=self.device)
        gradients = []
        
        # Batch size for memory limits
        batch_size = 10
        for i in range(0, self.steps, batch_size):
            alpha_batch = alphas[i:i+batch_size].view(-1, 1, 1, 1)
            interpolated = baseline + alpha_batch * (input_tensor - baseline)
            interpolated.requires_grad_(True)
            
            self.model.zero_grad()
            logits = self.model(interpolated)
            target_scores = logits[:, class_idx]
            
            grad = torch.autograd.grad(outputs=target_scores, 
                                       inputs=interpolated, 
                                       grad_outputs=torch.ones_like(target_scores),
                                       create_graph=False)[0]
            gradients.append(grad.detach())
            
        # Concatenate and compute Riemann sum approximation of the integral
        gradients = torch.cat(gradients, dim=0) # [steps, C, H, W]
        avg_gradients = gradients.mean(dim=0, keepdim=True) # [1, C, H, W]
        
        # Multiply by (input - baseline)
        attributions = (input_tensor - baseline) * avg_gradients
        
        # Sum across channels
        attr_map = attributions.squeeze(0).sum(dim=0).cpu().numpy() # [H, W]
        
        # Absolute value and min-max normalize
        attr_map = np.abs(attr_map)
        epsilon = 1e-7
        attr_min, attr_max = attr_map.min(), attr_map.max()
        if attr_max - attr_min > epsilon:
            attr_map = (attr_map - attr_min) / (attr_max - attr_min)
        else:
            attr_map = np.zeros_like(attr_map)
            
        return {
            'attribution_map': attr_map,
            'logits': orig_logits.detach().cpu(),
            'probabilities': probs.detach().cpu(),
            'confidence': probs[0, pred_class].item(),
            'predicted_class': pred_class,
            'target_class': class_idx
        }

def save_ig_figure(image_path, result, output_path):
    """Save a side-by-side figure of original image, Attribution Map, and overlay."""
    orig_img = load_visualization_image(image_path)
    orig_img = cv2.resize(orig_img, (224, 224))
    orig_img_uint8 = np.uint8(orig_img * 255)
    
    attr = result['attribution_map']
    attr_heatmap = cv2.applyColorMap(np.uint8(255 * attr), cv2.COLORMAP_JET)
    attr_heatmap = cv2.cvtColor(attr_heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(orig_img_uint8, 0.5, attr_heatmap, 0.5, 0)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(orig_img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(attr_heatmap)
    axes[1].set_title(f"Integrated Gradients\nPred: {result['predicted_class']} (Conf: {result['confidence']:.2f})")
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root / 'RETFound'))
    import models_vit as models
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = models.RETFound_mae(img_size=224, num_classes=5, drop_path_rate=0.2, global_pool=True)
    ckpt_path = project_root / 'checkpoint-best.pth'
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    
    ig_extractor = IntegratedGradients(model, device, steps=50)
    
    img_path = project_root / 'test_images' / '0005cfc8afb6.png'
    input_tensor = load_model_input(img_path, device)
    
    result = ig_extractor.generate(input_tensor)
    
    out_dir = project_root / 'outputs' / 'gradcam'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'ig_test.png'
    
    save_ig_figure(img_path, result, out_path)
    print(f"Saved Integrated Gradients figure to {out_path}")
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")
