import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

def load_dataset(dataset_name: str, **kwargs) -> Dict:
    """Load a physics dataset for training or evaluation.
    
    Args:
        dataset_name: Name of the dataset to load
        **kwargs: Additional dataset-specific parameters
        
    Returns:
        Dictionary containing dataset components
    """
    # Placeholder implementation
    if dataset_name == "cylinder_wake":
        # In practice, would load actual dataset
        return {
            "train": {
                "x": torch.randn(1000, 64, 64, 2),
                "coords": torch.randn(1000, 64, 64, 2),
                "pde_params": torch.randn(1000, 3)
            },
            "test": {
                "x": torch.randn(200, 64, 64, 2),
                "coords": torch.randn(200, 64, 64, 2),
                "pde_params": torch.randn(200, 3)
            }
        }
    elif dataset_name == "reaction_diffusion":
        # Placeholder for reaction-diffusion dataset
        return {
            "train": {
                "x": torch.randn(800, 128, 1),
                "coords": torch.randn(800, 128, 1),
                "pde_params": torch.randn(800, 2)
            },
            "test": {
                "x": torch.randn(200, 128, 1),
                "coords": torch.randn(200, 128, 1),
                "pde_params": torch.randn(200, 2)
            }
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def save_model(model, path: str, optimizer=None, epoch=None, metrics=None):
    """Save model checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'metrics': metrics
    }
    if optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(checkpoint, path)

def plot_results(true, pred, coords=None, title="Results", save_path=None):
    """Plot comparison between true and predicted fields."""
    if coords is None:
        # Simple 1D plot
        plt.figure(figsize=(10, 6))
        plt.plot(true, 'b-', label='True')
        plt.plot(pred, 'r--', label='Predicted')
        plt.title(title)
        plt.legend()
    else:
        # For 2D data
        if true.ndim == 3:  # [H, W, C] or [C, H, W]
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(true[:, :, 0], cmap='viridis')
            plt.title('True Field')
            plt.colorbar()
            
            plt.subplot(1, 3, 2)
            plt.imshow(pred[:, :, 0], cmap='viridis')
            plt.title('Predicted Field')
            plt.colorbar()
            
            plt.subplot(1, 3, 3)
            plt.imshow(np.abs(true[:, :, 0] - pred[:, :, 0]), cmap='hot')
            plt.title('Absolute Error')
            plt.colorbar()
            
            plt.suptitle(title)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
