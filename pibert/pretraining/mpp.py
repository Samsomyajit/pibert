import torch
import torch.nn as nn

class MaskedPhysicsPrediction(nn.Module):
    """Masked Physics Prediction pretraining task for PIBERT.
    
    Randomly masks portions of the physics field and trains the model to reconstruct them.
    """
    
    def __init__(self, mask_ratio=0.15):
        super().__init__()
        self.mask_ratio = mask_ratio
        
    def random_masking(self, x, mask_ratio=None):
        """
        Randomly mask portions of the input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, feature_dim]
            mask_ratio: Ratio of tokens to mask
            
        Returns:
            masked_x: Tensor with masked values
            mask: Binary mask indicating which values were masked
            ids_restore: Indexes to restore the original ordering
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
            
        batch_size, seq_length, _ = x.shape
        num_mask = int(seq_length * mask_ratio)
        
        # Random shuffle indices
        noise = torch.rand(batch_size, seq_length, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Create mask
        mask = torch.ones([batch_size, seq_length], device=x.device)
        mask[:, :num_mask] = 0
        mask = torch.gather(mask, dim=1, index=ids_shuffle)
        
        # Apply mask
        masked_x = x.clone()
        masked_x[mask == 0] = 0  # Simple zero masking
        
        return masked_x, mask, ids_restore
    
    def forward(self, x, coords=None, pde_equation=None):
        """
        Perform masked physics prediction.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, feature_dim]
            coords: Spatial coordinates
            pde_equation: PDE equation definition
            
        Returns:
            pred: Predicted values for masked positions
            mask: Binary mask indicating which values were masked
            loss: Reconstruction loss
        """
        # Mask input
        masked_x, mask, _ = self.random_masking(x)
        
        # Forward through model
        pred = self.model(masked_x, coords)
        
        # Calculate loss only on masked tokens
        loss = (pred - x) ** 2
        loss = loss.mean(dim=-1)  # [B, S]
        loss = (loss * mask).sum() / mask.sum()  # Mean only over masked tokens
        
        return pred, mask, loss
