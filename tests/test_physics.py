import pytest
import torch
from pibert.physics import PhysicsBiasedAttention

def test_physics_attention_initialization():
    """Test initialization of PhysicsBiasedAttention."""
    attention = PhysicsBiasedAttention(
        hidden_dim=64,
        num_heads=4,
        dropout=0.1,
        lambda_param=1.0
    )
    assert attention is not None

def test_physics_attention_forward():
    """Test forward pass of PhysicsBiasedAttention without physics bias."""
    attention = PhysicsBiasedAttention(
        hidden_dim=64,
        num_heads=4,
        dropout=0.1,
        lambda_param=0.0  # No physics bias
    )
    
    # Create dummy input [batch_size, seq_length, hidden_dim]
    x = torch.randn(4, 64, 64)
    
    # Test attention
    output = attention(x)
    assert output.shape == (4, 64, 64)

def test_physics_attention_with_residuals():
    """Test forward pass of PhysicsBiasedAttention with physics bias."""
    attention = PhysicsBiasedAttention(
        hidden_dim=64,
        num_heads=4,
        dropout=0.1,
        lambda_param=1.0
    )
    
    # Create dummy input
    x = torch.randn(4, 64, 64)
    coords = torch.randn(4, 64, 2)
    pde_residuals = torch.randn(4, 64, 3)  # 3 residual components
    
    # Test attention with physics bias
    output = attention(x, coords=coords, pde_residuals=pde_residuals)
    assert output.shape == (4, 64, 64)
