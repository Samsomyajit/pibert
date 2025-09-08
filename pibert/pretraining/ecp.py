import torch
import torch.nn as nn

class EquationConsistencyPrediction(nn.Module):
    """Equation Consistency Prediction pretraining task for PIBERT.
    
    Trains the model to distinguish between physically consistent and inconsistent field configurations.
    """
    
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        
    def create_inconsistent_samples(self, x, coords, pde_equation, noise_level=0.1):
        """
        Create physically inconsistent samples by perturbing consistent ones.
        
        Args:
            x: Physically consistent field [batch_size, seq_length, feature_dim]
            coords: Spatial coordinates
            pde_equation: PDE equation definition
            noise_level: Magnitude of perturbation
            
        Returns:
            inconsistent_x: Physically inconsistent field
            labels: Binary labels (1=consistent, 0=inconsistent)
        """
        batch_size = x.shape[0]
        
        # Randomly select half the batch to perturb
        mask = torch.rand(batch_size, device=x.device) < 0.5
        inconsistent_x = x.clone()
        
        # Add noise to create inconsistent samples
        noise = torch.randn_like(x) * noise_level
        inconsistent_x[mask] = x[mask] + noise[mask]
        
        # Create labels (1 for consistent, 0 for inconsistent)
        labels = torch.ones(batch_size, device=x.device)
        labels[mask] = 0
        
        return inconsistent_x, labels
    
    def forward(self, x, coords, pde_equation):
        """
        Perform equation consistency prediction.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, feature_dim]
            coords: Spatial coordinates
            pde_equation: PDE equation definition
            
        Returns:
            logits: Classification logits for physical consistency
            labels: True labels (1=consistent, 0=inconsistent)
            loss: Classification loss
        """
        # Create inconsistent samples
        inconsistent_x, labels = self.create_inconsistent_samples(x, coords, pde_equation)
        
        # Forward through model
        # In practice, would use a classification head on top of PIBERT
        logits = torch.rand(x.shape[0], self.num_classes, device=x.device)
        
        # Calculate classification loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels.long())
        
        return logits, labels, loss
