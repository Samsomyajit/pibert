import torch
import torch.nn as nn
import numpy as np

class FourierEmbedding(nn.Module):
    """Fourier embedding layer for capturing global patterns in PDE solutions."""
    
    def __init__(self, input_dim, hidden_dim, num_frequencies=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_frequencies = num_frequencies
        
        # Create frequency bands
        freq_bands = 2 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer('freq_bands', freq_bands)
        
        # Linear projection to hidden dimension
        self.projection = nn.Linear(input_dim * num_frequencies * 2, hidden_dim)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_length, input_dim]
            
        Returns:
            Embedded tensor of shape [batch_size, seq_length, hidden_dim]
        """
        # Compute Fourier features
        x_proj = x.unsqueeze(-1) * self.freq_bands  # [B, S, D, F]
        x_proj = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        x_proj = x_proj.reshape(*x.shape[:-1], -1)  # Flatten frequency and sin/cos dims
        
        # Project to hidden dimension
        return self.projection(x_proj)
