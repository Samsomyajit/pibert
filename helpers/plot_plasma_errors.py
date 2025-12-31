#!/usr/bin/env python3
"""
Raincloud-style L2 error comparison for plasma_npz between PINN and PIBERT.

Loads the trained models from results_plasma_quick, evaluates on the test split,
and plots the per-sample relative L2 error distribution for each model.

Example:
  python plot_plasma_errors.py --config config_plasma_quick.json --out plasma_errors.png
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from src.data import make_loaders
from src.runner import build_model, pick_device


def rel_l2(yhat, ygt):
    """Relative L2 over all spatial points."""
    num = torch.linalg.vector_norm((yhat - ygt).reshape(yhat.size(0), -1), ord=2, dim=1)
    den = torch.linalg.vector_norm(ygt.reshape(ygt.size(0), -1), ord=2, dim=1).clamp_min(1e-12)
    return (num / den).cpu().numpy()


def eval_model(model, loader, device, unnorm):
    model.eval()
    y_mean = torch.tensor(unnorm["y_mean"], device=device, dtype=torch.float32)
    y_std  = torch.tensor(unnorm["y_std"],  device=device, dtype=torch.float32)
    errs = []
    with torch.no_grad():
        for x, y, xyt in loader:
            x, y, xyt = x.to(device), y.to(device), xyt.to(device)
            pred = model(xyt, x)
            pred = pred.float()
            yhat = pred * y_std + y_mean
            ygt  = y    * y_std + y_mean
            if not torch.isfinite(yhat).all() or not torch.isfinite(ygt).all():
                continue
            errs.append(rel_l2(yhat, ygt))
    if not errs:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(errs, axis=0)


def main():
    ap = argparse.ArgumentParser(description="Plasma relative L2 error raincloud for PINN vs PIBERT.")
    ap.add_argument("--config", default="config_plasma_quick.json")
    ap.add_argument("--out", default="plasma_errors.png")
    args = ap.parse_args()

    cfg = json.load(open(args.config, "r"))
    device = pick_device(cfg["train"].get("device", "auto"))

    # Load data (use small batch for precise L2 per sample)
    train_loader, val_loader, test_loader, norm, shapes = make_loaders(
        cfg["data"]["root"], fmt=cfg["data"].get("format", "npz"),
        batch_size=1, normalize=True,
    )

    models = ["PINN", "PIBERT"]
    rows = []
    for name in models:
        print(f"[eval] {name}")
        model = build_model(name, shapes["Cin"], shapes["Cout"], cfg.get("model_cfg", {})).to(device)
        ckpt_dir = os.path.join(cfg["eval"]["outdir"], f"{name}_seed{cfg['train'].get('seed', 42)}")
        ckpt = os.path.join(ckpt_dir, "last.pt")
        if not os.path.exists(ckpt):
            print(f"[warn] missing checkpoint {ckpt}, skipping {name}")
            continue
        state = torch.load(ckpt, map_location=device)["model"]
        model.load_state_dict(state, strict=False)
        errs = eval_model(model, test_loader, device, norm)
        for e in errs:
            rows.append({"Model": name, "rel_L2": float(e)})

    if not rows:
        raise SystemExit("No errors computed; check checkpoints and data.")

    df = pd.DataFrame(rows)

    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
    })

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    order = ["PINN", "PIBERT"]
    pal = ["#999999", "#D55E00"]

    sns.violinplot(data=df, x="Model", y="rel_L2", order=order, palette=pal,
                   cut=0, inner=None, linewidth=0, ax=ax)
    sns.boxplot(data=df, x="Model", y="rel_L2", order=order,
                width=0.25, showcaps=True, boxprops={"zorder":3, "facecolor":"white"},
                whiskerprops={"linewidth":1.2}, medianprops={"color":"k","linewidth":1.4}, ax=ax)
    sns.stripplot(data=df, x="Model", y="rel_L2", order=order,
                  color="0.25", alpha=0.6, size=3.0, jitter=0.12, ax=ax)

    ax.set_ylabel(r"Relative $L_2$ Error", fontsize=14)
    ax.set_xlabel("")
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, axis="y", ls=":", lw=0.7, alpha=0.6)

    fig.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print("Saved", args.out)


if __name__ == "__main__":
    main()
