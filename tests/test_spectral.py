import pytest
import torch
from pibert.spectral import FourierEmbedding, WaveletEmbedding

def test_fourier_embedding_initialization():
    """Test initialization of FourierEmbedding."""
    embedding = FourierEmbedding(
        input_dim=2,
        hidden_dim=64,
        num_frequencies=128
    )
    assert embedding is not None

def test_fourier_embedding_forward():
    """Test forward pass of FourierEmbedding."""
    embedding = FourierEmbedding(
        input_dim=2,
        hidden_dim=64,
        num_frequencies=32
    )
    
    # Create dummy input [batch_size, seq_length, input_dim]
    x = torch.randn(4, 64, 2)
    
    # Test embedding
    output = embedding(x)
    assert output.shape == (4, 64, 64)

def test_wavelet_embedding_initialization():
    """Test initialization of WaveletEmbedding."""
    embedding = WaveletEmbedding(
        input_dim=2,
        hidden_dim=64,
        wavelet='db4',
        level=3
    )
    assert embedding is not None

@pytest.mark.skip(reason="PyWavelets dependency not installed in test environment")
def test_wavelet_embedding_forward():
    """Test forward pass of WaveletEmbedding."""
    embedding = WaveletEmbedding(
        input_dim=2,
        hidden_dim=64,
        wavelet='db4',
        level=3
    )
    
    # Create dummy input [batch_size, seq_length, input_dim]
    x = torch.randn(4, 64, 2)
    
    # Test embedding
    output = embedding(x)
    assert output.shape == (4, 64, 64)
