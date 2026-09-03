import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2

from xai.gradcam import load_model_input, load_visualization_image

class GradCAMPP:
    """
    Grad-CAM++ for Vision Transformers.
    
    Generates visual explanations using second-order gradients.
    Unlike standard Grad-CAM which uses global average pooling of gradients,
    Grad-CAM++ uses a weighted combination of positive partial derivatives.
    
    Alpha_k^c = relu(grad^2) / (2 * grad^2 + sum(activations * grad^3) + epsilon)
    """
    def __init__(self, model, target_layer, device):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.activations = None
        self.gradients = None
        self.handlers = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.handlers.append(self.target_layer.register_forward_hook(forward_hook))
        self.handlers.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        """Remove registered hooks."""
        for handler in self.handlers:
            handler.remove()
        self.handlers = []

    def generate(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM++ attribution map.
        
        Args:
            input_tensor (torch.Tensor): Input image tensor [B, C, H, W]
            class_idx (int, optional): Target class index. Defaults to predicted class.
            
        Returns:
            dict: Contains cam, logits, probabilities, confidence, predicted_class, target_class
        """
        self.model.eval()
        self.model.zero_grad()
        
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True

        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        
        if class_idx is None:
            class_idx = pred_class
            
        target_score = logits[0, class_idx]
        target_score.backward()

        # Get activations and gradients [B, N, C]
        activations = self.activations[0] # [N, C]
        gradients = self.gradients[0] # [N, C]
        
        # Discard CLS token
        activations = activations[1:] # [196, C]
        gradients = gradients[1:] # [196, C]
        
        # Calculate Grad-CAM++ weights
        grad_2 = gradients.pow(2)
        grad_3 = gradients.pow(3)
        
        # sum over spatial dimensions (N=196)
        global_sum = (activations * grad_3).sum(dim=0, keepdim=True)
        
        epsilon = 1e-7
        alpha = grad_2 / (2 * grad_2 + global_sum + epsilon)
        alpha = torch.where(gradients != 0.0, alpha, torch.zeros_like(alpha))
        
        # Apply ReLU to gradients
        weights = (alpha * F.relu(gradients)).sum(dim=0) # [C]
        
        # Weighted sum of activations
        cam = (activations * weights.unsqueeze(0)).sum(dim=-1) # [196]
        cam = F.relu(cam)
        
        # Reshape to 14x14 grid
        H = W = int(np.sqrt(cam.shape[0]))
        cam = cam.reshape(1, 1, H, W)
        
        # Bilinear upsample to 224x224
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Min-Max Normalization
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > epsilon:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return {
            'cam': cam,
            'logits': logits.detach().cpu(),
            'probabilities': probs.detach().cpu(),
            'confidence': probs[0, pred_class].item(),
            'predicted_class': pred_class,
            'target_class': class_idx
        }

def save_gradcampp_figure(image_path, result, output_path):
    """Save a side-by-side figure of original image, CAM, and overlay."""
    orig_img = load_visualization_image(image_path)
    orig_img = cv2.resize(orig_img, (224, 224))
    orig_img_uint8 = np.uint8(orig_img * 255)
    
    cam = result['cam']
    cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(orig_img_uint8, 0.5, cam_heatmap, 0.5, 0)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(orig_img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(cam_heatmap)
    axes[1].set_title(f"Grad-CAM++\nPred: {result['predicted_class']} (Conf: {result['confidence']:.2f})")
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
    
    target_layer = model.blocks[-1].norm1
    cam_extractor = GradCAMPP(model, target_layer, device)
    
    img_path = project_root / 'test_images' / '0005cfc8afb6.png'
    input_tensor = load_model_input(img_path, device)
    
    result = cam_extractor.generate(input_tensor)
    cam_extractor.remove_hooks()
    
    out_dir = project_root / 'outputs' / 'gradcam'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'gradcampp_test.png'
    
    save_gradcampp_figure(img_path, result, out_path)
    print(f"Saved Grad-CAM++ figure to {out_path}")
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")
