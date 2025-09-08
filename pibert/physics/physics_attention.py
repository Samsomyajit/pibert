import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsBiasedAttention(nn.Module):
    """Attention mechanism biased by physics-informed residuals."""
    
    def __init__(self, hidden_dim, num_heads, dropout=0.1, lambda_param=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.lambda_param = lambda_param  # Controls physics bias strength
        
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, coords=None, pde_residuals=None, mask=None):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_length, hidden_dim]
            coords: Spatial coordinates [batch_size, seq_length, spatial_dim]
            pde_residuals: PDE residuals [batch_size, seq_length, residual_dim]
            mask: Attention mask [batch_size, seq_length, seq_length]
            
        Returns:
            Output tensor of shape [batch_size, seq_length, hidden_dim]
        """
        batch_size, seq_length, _ = x.shape
        
        # Project to query, key, value
        q = self.q_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Standard scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply physics bias if residuals are provided
        if pde_residuals is not None and self.lambda_param > 0:
            # Compute physics-based attention bias
            # In practice, this would use the actual PDE residuals
            # Simple example: use residual magnitude as bias
            residual_bias = torch.norm(pde_residuals, dim=-1)  # [B, S]
            residual_bias = residual_bias.unsqueeze(1) - residual_bias.unsqueeze(2)  # [B, S, S]
            residual_bias = residual_bias.unsqueeze(1)  # [B, 1, S, S]
            
            # Apply bias to attention scores
            attn_scores = attn_scores - self.lambda_param * residual_bias
            
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
            
        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_dim)
        
        # Output projection
        return self.out_proj(attn_output)
