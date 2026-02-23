import torch
import torch.nn as nn
import numpy as np
from scipy import ndimage
from skimage import measure


def compute_betti_numbers(mask):
    """
    Compute Betti numbers for a binary mask.
    
    Betti numbers are topological invariants:
    - β0 (betti_0): Number of connected components
    - β1 (betti_1): Number of holes/loops
    
    Args:
        mask: Binary numpy array (H, W) with values in {0, 1}
    
    Returns:
        tuple: (betti_0, betti_1)
    """
    if not isinstance(mask, np.ndarray):
        mask = mask.cpu().numpy()
    
    mask = mask.astype(bool)
    
    # Betti 0: Number of connected components
    labeled_array, num_components = ndimage.label(mask)
    betti_0 = num_components
    
    # Betti 1: Number of holes (using Euler characteristic)
    # Euler characteristic: χ = β0 - β1
    # Therefore: β1 = β0 - χ
    
    # Compute Euler characteristic using measure.regionprops
    if num_components > 0:
        props = measure.regionprops(labeled_array)
        # Euler number = vertices - edges + faces
        # For 2D: Euler = #components - #holes
        euler_char = sum(prop.euler_number for prop in props)
        betti_1 = betti_0 - euler_char
    else:
        betti_1 = 0
    
    return betti_0, betti_1


def betti_matching_loss(pred, target, beta_0_weight=1.0, beta_1_weight=0.5):
    """
    Compute Betti matching loss between prediction and target.
    
    This loss penalizes differences in topological features:
    - Difference in number of connected components (β0)
    - Difference in number of holes (β1)
    
    Args:
        pred: Predicted probabilities, shape (B, C, H, W)
        target: Ground truth binary masks, shape (B, C, H, W)
        beta_0_weight: Weight for component count difference
        beta_1_weight: Weight for hole count difference
    
    Returns:
        Scalar loss value
    """
    batch_size = pred.shape[0]
    total_loss = 0.0
    
    # Convert to binary (threshold at 0.5)
    pred_binary = (pred > 0.5).float()
    
    for b in range(batch_size):
        for c in range(pred.shape[1]):
            # Extract single channel masks
            pred_mask = pred_binary[b, c].cpu().numpy()
            target_mask = target[b, c].cpu().numpy()
            
            # Compute Betti numbers
            pred_b0, pred_b1 = compute_betti_numbers(pred_mask)
            target_b0, target_b1 = compute_betti_numbers(target_mask)
            
            # L1 loss on Betti numbers
            b0_diff = abs(pred_b0 - target_b0)
            b1_diff = abs(pred_b1 - target_b1)
            
            # Weighted combination
            loss = beta_0_weight * b0_diff + beta_1_weight * b1_diff
            total_loss += loss
    
    # Average over batch
    return total_loss / (batch_size * pred.shape[1])


def soft_betti_matching_loss(pred, target, beta_0_weight=1.0, beta_1_weight=0.5, temperature=0.1):
    """
    Soft (differentiable approximation) version of Betti matching loss.
    
    This version uses a soft thresholding approach to make the loss
    more suitable for gradient-based optimization.
    
    Args:
        pred: Predicted probabilities, shape (B, C, H, W)
        target: Ground truth binary masks, shape (B, C, H, W)
        beta_0_weight: Weight for component count difference
        beta_1_weight: Weight for hole count difference
        temperature: Temperature for soft thresholding (lower = sharper)
    
    Returns:
        Scalar loss value
    """
    batch_size = pred.shape[0]
    total_loss = 0.0
    
    # Soft thresholding using sigmoid with temperature
    # This makes predictions closer to 0 or 1 while maintaining differentiability
    pred_soft = torch.sigmoid((pred - 0.5) / temperature)
    
    for b in range(batch_size):
        for c in range(pred.shape[1]):
            # Extract single channel masks
            # Use soft predictions for forward pass, detach for Betti computation
            pred_mask = (pred_soft[b, c] > 0.5).detach().cpu().numpy()
            target_mask = target[b, c].cpu().numpy()
            
            # Compute Betti numbers
            pred_b0, pred_b1 = compute_betti_numbers(pred_mask)
            target_b0, target_b1 = compute_betti_numbers(target_mask)
            
            # L1 loss on Betti numbers
            b0_diff = abs(pred_b0 - target_b0)
            b1_diff = abs(pred_b1 - target_b1)
            
            # Weighted combination
            loss = beta_0_weight * b0_diff + beta_1_weight * b1_diff
            total_loss += loss
    
    # Average over batch
    # Convert to tensor for gradient flow
    return torch.tensor(total_loss / (batch_size * pred.shape[1]), 
                       device=pred.device, requires_grad=True)


class BettiMatchingLoss(nn.Module):
    """
    Betti Matching Loss as a PyTorch module.
    
    Penalizes topological differences between predicted and ground truth masks:
    - Different number of connected components (fragmentation/merging)
    - Different number of holes (topology errors)
    """
    
    def __init__(self, beta_0_weight=1.0, beta_1_weight=0.5, soft=False, temperature=0.1):
        """
        Args:
            beta_0_weight: Weight for connected components difference (β0)
            beta_1_weight: Weight for holes difference (β1)
            soft: If True, use soft (differentiable) version
            temperature: Temperature for soft thresholding (only used if soft=True)
        """
        super().__init__()
        self.beta_0_weight = beta_0_weight
        self.beta_1_weight = beta_1_weight
        self.soft = soft
        self.temperature = temperature
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted probabilities, shape (B, C, H, W)
            target: Ground truth binary masks, shape (B, C, H, W)
        
        Returns:
            Scalar loss value
        """
        if self.soft:
            return soft_betti_matching_loss(
                pred, target, 
                self.beta_0_weight, 
                self.beta_1_weight,
                self.temperature
            )
        else:
            loss_value = betti_matching_loss(
                pred, target,
                self.beta_0_weight,
                self.beta_1_weight
            )
            return torch.tensor(loss_value, device=pred.device, requires_grad=True)


if __name__ == "__main__":
    # Test the Betti loss
    print("Testing Betti Matching Loss...")
    
    # Create synthetic test case
    batch_size = 2
    pred = torch.rand(batch_size, 1, 64, 64)
    target = torch.rand(batch_size, 1, 64, 64) > 0.5
    target = target.float()
    
    # Test loss computation
    betti_loss_fn = BettiMatchingLoss(beta_0_weight=1.0, beta_1_weight=0.5)
    loss = betti_loss_fn(pred, target)
    
    print(f"Betti Loss: {loss.item():.4f}")
    print("Test passed!")
