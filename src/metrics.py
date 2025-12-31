# metrics.py
import numpy as np, torch, pandas as pd

def mae(pred, gt):
    return torch.mean(torch.abs(pred - gt)).item()

def mse(pred, gt):
    return torch.mean((pred - gt)**2).item()

def nmse(pred, gt, eps=1e-8):
    num = torch.sum((pred - gt)**2).item()
    den = torch.sum(gt**2).item() + eps
    return num / den

def relative_error(pred, gt, eps=1e-8):
    return torch.abs(pred - gt) / (torch.abs(gt) + eps)

def table_from_results(rows):
    df = pd.DataFrame(rows, columns=["Model", "Param(M)", "MAE(1e-3)", "MSE", "NMSE"])
    return df
