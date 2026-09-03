import torch
import torch.nn as nn
import torch.nn.functional as F

class CORALLoss(nn.Module):
    """
    Consistent Rank Logits (CORAL) Loss for ordinal regression.
    Instead of standard K-class softmax cross-entropy, this tests K-1 binary
    classifiers predicting P(Y > k) for k in 0, 1, ..., K-2.
    """
    def __init__(self, num_classes=5):
        super(CORALLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, K-1] unnormalized scores for P(Y > k).
            targets: [B] target class indices (0 to K-1).
        Returns:
            Scalar loss.
        """
        B = logits.size(0)
        num_tasks = self.num_classes - 1
        
        # Create binary target matrix of shape [B, K-1]
        # Element (i, j) is 1 if target[i] > j, else 0
        expanded_targets = targets.view(B, 1).expand(B, num_tasks)
        thresholds = torch.arange(num_tasks, device=targets.device).view(1, num_tasks).expand(B, num_tasks)
        
        binary_targets = (expanded_targets > thresholds).float()
        
        # Binary cross entropy with logits computes stable sigmoid cross entropy
        loss = F.binary_cross_entropy_with_logits(logits, binary_targets, reduction='mean')
        return loss

class EMDLoss(nn.Module):
    """
    Earth Mover's Distance Loss.
    Treats the classification task as comparing discrete probability distributions.
    EMD is calculated as the sum of squared cumulative distribution differences.
    """
    def __init__(self):
        super(EMDLoss, self).__init__()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, K] unnormalized scores.
            targets: [B] target class indices (0 to K-1).
        Returns:
            Scalar loss.
        """
        K = logits.size(1)
        probs = F.softmax(logits, dim=-1)
        
        # Convert targets to one-hot distributions
        target_probs = F.one_hot(targets, num_classes=K).float()
        
        # Calculate cumulative distributions
        cdf_pred = torch.cumsum(probs, dim=-1)
        cdf_target = torch.cumsum(target_probs, dim=-1)
        
        # Calculate squared differences
        emd = torch.mean((cdf_pred - cdf_target) ** 2)
        return emd

class CombinedLoss(nn.Module):
    """
    Combined loss for DR-EarlyXAI including:
    - Severity loss (Cross-Entropy or CORAL)
    - Screening loss (BCE)
    - Ordinal penalty (EMD)
    """
    def __init__(self, severity_weight=1.0, screening_weight=0.5, ordinal_weight=0.3, use_coral=False):
        super(CombinedLoss, self).__init__()
        self.severity_weight = severity_weight
        self.screening_weight = screening_weight
        self.ordinal_weight = ordinal_weight
        self.use_coral = use_coral
        
        if self.use_coral:
            self.severity_criterion = CORALLoss(num_classes=5)
        else:
            self.severity_criterion = nn.CrossEntropyLoss()
            
        self.screening_criterion = nn.BCEWithLogitsLoss()
        self.ordinal_criterion = EMDLoss()

    def forward(self, severity_logits, screening_logits, severity_targets, screening_targets=None):
        """
        Args:
            severity_logits: [B, 5] if use_coral is False, else [B, 4]
            screening_logits: [B, 1]
            severity_targets: [B] containing grades 0-4
            screening_targets: [B] containing 0 (No DR) or 1 (DR). If None, calculated from severity_targets.
        """
        if screening_targets is None:
            screening_targets = (severity_targets > 0).float()
            
        screening_targets = screening_targets.view(-1, 1).float()
        
        loss_sev = self.severity_criterion(severity_logits, severity_targets)
        loss_scr = self.screening_criterion(screening_logits, screening_targets)
        
        # EMD Loss expects [B, 5] logits. If using CORAL, we might not have a direct [B, 5] logit,
        # so typically EMD is applied when severity_logits is [B, 5].
        # If CORAL is used, we'll skip EMD or compute it if logits can be converted.
        # Assuming for this implementation that when use_coral=True, ordinal_weight=0 or we ignore it
        if self.use_coral:
            loss_ord = 0.0
        else:
            loss_ord = self.ordinal_criterion(severity_logits, severity_targets)
            
        total_loss = (self.severity_weight * loss_sev) + \
                     (self.screening_weight * loss_scr) + \
                     (self.ordinal_weight * loss_ord)
                     
        return total_loss

if __name__ == '__main__':
    # Synthetic test
    B = 4
    K = 5
    
    # Random targets between 0 and 4
    targets = torch.randint(0, K, (B,))
    
    # Test CORAL
    coral_logits = torch.randn(B, K-1)
    coral = CORALLoss(num_classes=K)
    loss_coral = coral(coral_logits, targets)
    print(f"CORAL Loss: {loss_coral.item():.4f}")
    
    # Test EMD
    ce_logits = torch.randn(B, K)
    emd = EMDLoss()
    loss_emd = emd(ce_logits, targets)
    print(f"EMD Loss: {loss_emd.item():.4f}")
    
    # Test Combined
    scr_logits = torch.randn(B, 1)
    combined = CombinedLoss(use_coral=False)
    loss_comb = combined(ce_logits, scr_logits, targets)
    print(f"Combined Loss (CE+BCE+EMD): {loss_comb.item():.4f}")
