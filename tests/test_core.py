import pytest
import torch
from pibert.core import PhysicsTransformer

def test_physics_transformer_initialization():
    """Test basic initialization of PhysicsTransformer."""
    model = PhysicsTransformer(
        input_dim=2,
        hidden_dim=64,
        num_layers=2,
        num_heads=4
    )
    assert model is not None

def test_physics_transformer_forward():
    """Test forward pass of PhysicsTransformer."""
    model = PhysicsTransformer(
        input_dim=2,
        hidden_dim=64,
        num_layers=2,
        num_heads=4
    )
    
    # Create dummy input [batch_size, seq_length, input_dim]
    x = torch.randn(4, 64*64, 2)
    
    # Test with and without coordinates
    output = model(x)
    assert output.shape == (4, 64*64, 64)
    
    # Test with coordinates
    coords = torch.randn(4, 64*64, 2)
    output = model(x, coords=coords)
    assert output.shape == (4, 64*64, 64)

def test_physics_transformer_predict():
    """Test predict method of PhysicsTransformer."""
    model = PhysicsTransformer(
        input_dim=2,
        hidden_dim=64,
        num_layers=2,
        num_heads=4
    )
    
    # Create dummy input
    x = torch.randn(4, 64*64, 2)
    coords = torch.randn(4, 64*64, 2)
    
    # Test prediction
    pred = model.predict(x, coords=coords)
    assert pred.shape == (4, 64*64, 2)
