#!/usr/bin/env python3
"""
Cylinder strip plot: Ground Truth + PINN/PIBERT absolute error across multiple timesteps.
Uses config_cylinder_vort300.json and results_cylinder_vort300 checkpoints by default.
"""
import argparse, json, os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from src.data import make_loaders
from src.runner import build_model, pick_device


def field_mag(arr):
    if arr.ndim == 4:
        if arr.shape[1] >= 2:
            return np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2)
        return arr[:, 0]
    if arr.ndim == 3:
        if arr.shape[0] >= 2:
            return np.sqrt(arr[0] ** 2 + arr[1] ** 2)
        return arr[0]
    raise ValueError(f"Unexpected shape {arr.shape}")


def collect_preds(model, loader, norm, device):
    model.eval()
    y_mean = torch.tensor(norm["y_mean"], device=device, dtype=torch.float32)
    y_std  = torch.tensor(norm["y_std"],  device=device, dtype=torch.float32)
    preds, gts = [], []
    with torch.no_grad():
        for x, y, xyt in loader:
            x, y, xyt = x.to(device), y.to(device), xyt.to(device)
            p = model(xyt, x).float()
            yhat = (p * y_std + y_mean).cpu().numpy()
            ygt  = (y * y_std + y_mean).cpu().numpy()
            preds.append(yhat[0])
            gts.append(ygt[0])
    return np.stack(preds, axis=0), np.stack(gts, axis=0)


def load_models(cfg_path, models=("PINN", "PIBERT"), device_pref="auto"):
    cfg = json.load(open(cfg_path, "r"))
    device = pick_device(cfg.get("train", {}).get("device", device_pref))
    _, _, test_loader, norm, shapes = make_loaders(
        cfg["data"]["root"], fmt=cfg["data"].get("format", "npz"),
        batch_size=1, normalize=True,
    )
    preds, gts = {}, None
    for name in models:
        model = build_model(name, shapes["Cin"], shapes["Cout"], cfg.get("model_cfg", {})).to(device)
        ckpt_dir = os.path.join(cfg["eval"]["outdir"], f"{name}_seed{cfg['train'].get('seed', 42)}")
        ckpt = os.path.join(ckpt_dir, "last.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Missing checkpoint {ckpt}")
        raw_state = torch.load(ckpt, map_location=device)["model"]
        current = model.state_dict()
        state = {k: v for k, v in raw_state.items() if k in current and current[k].shape == v.shape}
        model.load_state_dict(state, strict=False)
        p, gt_arr = collect_preds(model, test_loader, norm, device)
        preds[name] = p
        if gts is None:
            gts = gt_arr
    return preds, gts


def main():
    ap = argparse.ArgumentParser(description="Cylinder GT + abs error strip (PINN vs PIBERT).")
    ap.add_argument("--config", default="config_cylinder_vort300.json")
    ap.add_argument("--out", default="panel_cylinder_strip.png")
    ap.add_argument("--timesteps", type=int, default=6, help="number of sample indices to show")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    preds, gts = load_models(args.config, device_pref=args.device)
    n = gts.shape[0]
    steps = min(args.timesteps, n)
    idxs = np.linspace(0, n - 1, steps, dtype=int)

    sns.set_theme(style="white", context="talk")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "axes.titleweight": "bold",
    })

    gt_mag = field_mag(gts)
    pinn_err = np.abs(field_mag(preds["PINN"]) - gt_mag)
    pib_err  = np.abs(field_mag(preds["PIBERT"]) - gt_mag)

    gt_vmin, gt_vmax = np.percentile(gt_mag, [1, 99])
    err_vmin, err_vmax = 0, np.percentile(np.concatenate([pinn_err, pib_err]).reshape(-1), 99)

    fig = plt.figure(figsize=(14, 5.8), dpi=300)
    gs = fig.add_gridspec(nrows=3, ncols=steps, hspace=0.08, wspace=0.05)

    cm_field = "coolwarm"
    cm_err = "magma"

    gt_axes, pinn_axes, pib_axes = [], [], []

    for c, idx in enumerate(idxs):
        fig.text((c + 0.5) / steps, 0.98, f"{idx}", ha="center", va="top",
                 fontsize=13, fontweight="bold", color="0.15")

    for c, idx in enumerate(idxs):
        ax = fig.add_subplot(gs[0, c])
        im_gt = ax.imshow(gt_mag[idx], cmap=cm_field, origin="lower", vmin=gt_vmin, vmax=gt_vmax)
        ax.set_xticks([]); ax.set_yticks([])
        gt_axes.append(ax)

        ax = fig.add_subplot(gs[1, c])
        im_pinn = ax.imshow(pinn_err[idx], cmap=cm_err, origin="lower", vmin=err_vmin, vmax=err_vmax)
        ax.set_xticks([]); ax.set_yticks([])
        pinn_axes.append(ax)

        ax = fig.add_subplot(gs[2, c])
        im_pib = ax.imshow(pib_err[idx], cmap=cm_err, origin="lower", vmin=err_vmin, vmax=err_vmax)
        ax.set_xticks([]); ax.set_yticks([])
        pib_axes.append(ax)

    fig.text(0.015, 0.78, "Ground\nTruth", ha="left", va="center", fontsize=13, fontweight="bold")
    fig.text(0.015, 0.47, "PINN\nAbs Error", ha="left", va="center", fontsize=13, fontweight="bold")
    fig.text(0.015, 0.17, "PIBERT\nAbs Error", ha="left", va="center", fontsize=13, fontweight="bold")

    cax_gt = fig.add_axes([0.92, 0.70, 0.015, 0.20])
    cb_gt = fig.colorbar(im_gt, cax=cax_gt)
    cb_gt.ax.tick_params(labelsize=9)
    cb_gt.set_label("Field", fontsize=10)

    cax_err = fig.add_axes([0.92, 0.15, 0.015, 0.45])
    cb_err = fig.colorbar(im_pinn, cax=cax_err)
    cb_err.ax.tick_params(labelsize=9)
    cb_err.set_label("|Error|", fontsize=10)

    fig.suptitle("Cylinder: GT and Absolute Errors", y=0.995, fontsize=16, weight="bold")
    plt.savefig(args.out, bbox_inches="tight")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
