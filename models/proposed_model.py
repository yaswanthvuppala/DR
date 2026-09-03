import sys
from pathlib import Path
import torch
import torch.nn as nn

class MultiScaleFeatureFusion(nn.Module):
    """
    Multi-Scale Feature Fusion module.
    Takes features from multiple scales (shallow, middle, deep),
    projects them to a common fusion dimension, concatenates them,
    and then applies a final linear projection, LayerNorm, and GELU.
    """
    def __init__(self, in_dim=1024, fusion_dim=512, num_scales=3):
        super(MultiScaleFeatureFusion, self).__init__()
        # Individual projections for each scale
        self.projections = nn.ModuleList([
            nn.Linear(in_dim, fusion_dim) for _ in range(num_scales)
        ])
        
        # Final projection after concatenation
        self.final_proj = nn.Linear(num_scales * fusion_dim, fusion_dim)
        self.norm = nn.LayerNorm(fusion_dim)
        self.act = nn.GELU()

    def forward(self, features):
        """
        Args:
            features: List of 3 tensors, each of shape [B, N, in_dim]
        Returns:
            fused_features: Tensor of shape [B, N, fusion_dim]
        """
        projected = []
        for i, proj in enumerate(self.projections):
            projected.append(proj(features[i]))
            
        # Concatenate along the feature dimension
        concat_features = torch.cat(projected, dim=-1)  # [B, N, 3 * fusion_dim]
        
        # Final projection, norm, and activation
        out = self.final_proj(concat_features)  # [B, N, fusion_dim]
        out = self.norm(out)
        out = self.act(out)
        
        return out

class AttentionPooling(nn.Module):
    """
    Attention Pooling module.
    Learns a weight for each token and aggregates the tokens via a weighted sum.
    """
    def __init__(self, in_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(in_dim, 1)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, N, in_dim]
        Returns:
            pooled: Tensor of shape [B, in_dim]
            weights: Tensor of shape [B, N] containing attention weights
        """
        # Calculate attention scores
        scores = self.attention(x).squeeze(-1)  # [B, N]
        weights = torch.softmax(scores, dim=1)  # [B, N]
        
        # Weighted sum: expand weights to [B, N, 1] for broadcasting
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)  # [B, in_dim]
        
        return pooled, weights

class DREarlyXAI(nn.Module):
    """
    Proposed DR-EarlyXAI Model.
    Wraps a frozen RETFound backbone and adds multi-scale feature fusion,
    attention pooling, and dual classification heads for severity and screening.
    """
    def __init__(self, retfound_model, num_classes=5, fusion_dim=512, dropout=0.3, 
                 shallow_idx=7, middle_idx=15, deep_idx=23):
        super(DREarlyXAI, self).__init__()
        self.backbone = retfound_model
        
        # Freeze ALL backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.shallow_idx = shallow_idx
        self.middle_idx = middle_idx
        self.deep_idx = deep_idx
        
        # Assuming RETFound ViT-Large with embed_dim=1024
        in_dim = self.backbone.embed_dim
        
        self.fusion = MultiScaleFeatureFusion(in_dim=in_dim, fusion_dim=fusion_dim, num_scales=3)
        self.pool = AttentionPooling(in_dim=fusion_dim)
        
        self.severity_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes)
        )
        
        self.screening_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 1)
        )

    def forward(self, x):
        """
        Args:
            x: Input images [B, 3, 224, 224]
        Returns:
            dict containing logits, pooled features, and attention weights
        """
        B = x.shape[0]
        
        # 1. Forward through the patch embedding
        x = self.backbone.patch_embed(x)
        
        # 2. Add CLS token and position embedding
        cls_tokens = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)
        
        # 3. Iterate through transformer blocks and capture features
        features = []
        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x)
            if i in [self.shallow_idx, self.middle_idx, self.deep_idx]:
                # Drop CLS token, keep only patch tokens: [B, 196, 1024]
                features.append(x[:, 1:, :])
                
        # 4. Multi-Scale Feature Fusion
        fused = self.fusion(features)  # [B, 196, fusion_dim]
        
        # 5. Attention Pooling
        pooled, attn_weights = self.pool(fused)  # [B, fusion_dim], [B, 196]
        
        # 6. Dual Classification Heads
        sev_logits = self.severity_head(pooled)  # [B, 5]
        scr_logits = self.screening_head(pooled) # [B, 1]
        
        return {
            'severity_logits': sev_logits,
            'screening_logits': scr_logits,
            'features': pooled,
            'attention_weights': attn_weights
        }

    def get_prediction(self, x):
        """Convenience method for inference."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            sev_probs = torch.softmax(outputs['severity_logits'], dim=-1)
            sev_preds = torch.argmax(sev_probs, dim=-1)
            
            scr_probs = torch.sigmoid(outputs['screening_logits'])
            scr_preds = (scr_probs > 0.5).float().squeeze(-1)
            
            return {
                'severity_pred': sev_preds,
                'screening_pred': scr_preds,
                'severity_prob': sev_probs,
                'screening_prob': scr_probs,
                'attention_weights': outputs['attention_weights']
            }

def count_parameters(model, trainable_only=True):
    """Utility to count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def load_proposed_model(checkpoint_path, device, fusion_dim=512, dropout=0.3):
    """
    Loads RETFound from checkpoint, wraps it in DREarlyXAI, and moves to device.
    """
    project_root = Path(__file__).resolve().parent.parent
    retfound_path = project_root / 'RETFound'
    if str(retfound_path) not in sys.path:
        sys.path.insert(0, str(retfound_path))
        
    import models_vit as models
    
    # Initialize base RETFound model architecture
    # Based on the prompt: ViT-Large, patch_size=16, embed_dim=1024, depth=24, num_heads=16
    model_base = models.RETFound_mae(img_size=224, num_classes=5, drop_path_rate=0.2, global_pool=True)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    # Strip 'module.' prefix if present (from DistributedDataParallel)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    msg = model_base.load_state_dict(state_dict, strict=False)
    print(f"Loaded RETFound checkpoint with msg: {msg}")
    
    # Wrap in DREarlyXAI
    model = DREarlyXAI(model_base, fusion_dim=fusion_dim, dropout=dropout)
    model.to(device)
    
    return model

if __name__ == '__main__':
    # Test script
    import torch
    import sys
    from pathlib import Path

    project_root = Path(r"c:\Users\vuppa\Desktop\DR")
    checkpoint_file = project_root / "checkpoint-best.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not checkpoint_file.exists():
        print(f"Checkpoint not found at {checkpoint_file}. Cannot run full test.")
        # We can still test with randomly initialized weights
        print("Testing with randomly initialized base model...")
        
        retfound_path = project_root / 'RETFound'
        sys.path.insert(0, str(retfound_path))
        try:
            import models_vit as models
            base = models.RETFound_mae(global_pool=True)
            model = DREarlyXAI(base).to(device)
            
            x = torch.randn(2, 3, 224, 224).to(device)
            out = model(x)
            print("\nOutput shapes:")
            for k, v in out.items():
                print(f"  {k}: {v.shape}")
                
            print(f"\nTotal params: {count_parameters(model, False):,}")
            print(f"Trainable params: {count_parameters(model, True):,}")
        except Exception as e:
            print(f"Failed to test with dummy model: {e}")
    else:
        print(f"Loading checkpoint from {checkpoint_file}")
        model = load_proposed_model(checkpoint_file, device)
        
        print(f"\nTotal params: {count_parameters(model, False):,}")
        print(f"Trainable params: {count_parameters(model, True):,}")
        
        # Test forward pass
        print("\nTesting forward pass...")
        x = torch.randn(2, 3, 224, 224).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(x)
            
        print("\nOutput shapes:")
        for k, v in outputs.items():
            print(f"  {k}: {v.shape}")
            
        preds = model.get_prediction(x)
        print("\nPredictions:")
        print(f"  Severity: {preds['severity_pred']}")
        print(f"  Screening: {preds['screening_pred']}")
