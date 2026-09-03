import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2

from xai.gradcam import load_model_input, load_visualization_image

class OcclusionSensitivity:
    """
    Occlusion Sensitivity for Vision Transformers.
    
    Slides an occluding patch across the input image and measures 
    the drop in target class probability. This creates a sensitivity map.
    """
    def __init__(self, model, device, patch_size=16, stride=8):
        self.model = model
        self.device = device
        self.patch_size = patch_size
        self.stride = stride

    def generate(self, input_tensor, class_idx=None):
        """
        Generate Occlusion Sensitivity map.
        
        Args:
            input_tensor (torch.Tensor): Input image tensor [1, C, H, W]
            class_idx (int, optional): Target class index. Defaults to predicted class.
            
        Returns:
            dict: Contains sensitivity_map, logits, probabilities, confidence, predicted_class, target_class, baseline_prob
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        
        _, C, H, W = input_tensor.shape
        
        # Get baseline predictions
        with torch.inference_mode():
            orig_logits = self.model(input_tensor)
            probs = F.softmax(orig_logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()
            
            if class_idx is None:
                class_idx = pred_class
                
            baseline_prob = probs[0, class_idx].item()

        # Compute output dimensions for sliding window
        out_h = (H - self.patch_size) // self.stride + 1
        out_w = (W - self.patch_size) // self.stride + 1
        sensitivity_map = np.zeros((out_h, out_w), dtype=np.float32)

        # Occlusion value: assuming input is normalized, 
        # a gray patch before normalization is 0.5. 
        # We can just use mean value of the image or 0 for normalized space.
        # We will use 0.0 which represents mean in standard ImageNet normalization.
        occlusion_value = 0.0

        batch_size = 16
        batch_inputs = []
        batch_coords = []

        def process_batch(inputs, coords):
            inputs_tensor = torch.cat(inputs, dim=0).to(self.device)
            with torch.inference_mode():
                logits = self.model(inputs_tensor)
                batch_probs = F.softmax(logits, dim=-1)[:, class_idx].cpu().numpy()
                
            for prob, (i, j) in zip(batch_probs, coords):
                sensitivity_map[i, j] = baseline_prob - prob

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = h_start + self.patch_size
                w_end = w_start + self.patch_size

                occluded_img = input_tensor.clone()
                occluded_img[0, :, h_start:h_end, w_start:w_end] = occlusion_value

                batch_inputs.append(occluded_img)
                batch_coords.append((i, j))

                if len(batch_inputs) == batch_size:
                    process_batch(batch_inputs, batch_coords)
                    batch_inputs = []
                    batch_coords = []

        if len(batch_inputs) > 0:
            process_batch(batch_inputs, batch_coords)

        # Resize to 224x224 and normalize
        sensitivity_map = cv2.resize(sensitivity_map, (W, H), interpolation=cv2.INTER_CUBIC)
        
        # Min-max normalization (we only care about positive drops indicating importance)
        sensitivity_map = np.maximum(sensitivity_map, 0)
        epsilon = 1e-7
        map_min, map_max = sensitivity_map.min(), sensitivity_map.max()
        if map_max - map_min > epsilon:
            sensitivity_map = (sensitivity_map - map_min) / (map_max - map_min)
        else:
            sensitivity_map = np.zeros_like(sensitivity_map)
            
        return {
            'sensitivity_map': sensitivity_map,
            'logits': orig_logits.detach().cpu(),
            'probabilities': probs.detach().cpu(),
            'confidence': baseline_prob,
            'predicted_class': pred_class,
            'target_class': class_idx,
            'baseline_prob': baseline_prob
        }

def save_occlusion_figure(image_path, result, output_path):
    """Save a side-by-side figure of original image, Sensitivity Map, and overlay."""
    orig_img = load_visualization_image(image_path)
    orig_img = cv2.resize(orig_img, (224, 224))
    orig_img_uint8 = np.uint8(orig_img * 255)
    
    sens_map = result['sensitivity_map']
    sens_heatmap = cv2.applyColorMap(np.uint8(255 * sens_map), cv2.COLORMAP_JET)
    sens_heatmap = cv2.cvtColor(sens_heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(orig_img_uint8, 0.5, sens_heatmap, 0.5, 0)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(orig_img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(sens_heatmap)
    axes[1].set_title(f"Occlusion Sensitivity\nPred: {result['predicted_class']} (Conf: {result['confidence']:.2f})")
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
    
    # 16x16 patch with stride 8
    occlusion_extractor = OcclusionSensitivity(model, device, patch_size=16, stride=8)
    
    img_path = project_root / 'test_images' / '0005cfc8afb6.png'
    input_tensor = load_model_input(img_path, device)
    
    result = occlusion_extractor.generate(input_tensor)
    
    out_dir = project_root / 'outputs' / 'gradcam'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'occlusion_test.png'
    
    save_occlusion_figure(img_path, result, out_path)
    print(f"Saved Occlusion Sensitivity figure to {out_path}")
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Baseline Confidence: {result['baseline_prob']:.4f}")
