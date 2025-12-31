#!/usr/bin/env python3
"""
Plasma panel figure: raincloud L2 errors, cumulative error curves, and GT/abs error strips.

Uses trained PINN/PIBERT checkpoints from results_plasma_quick and plasma_npz data.
"""
import argparse, json, os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from src.data import make_loaders
from src.runner import build_model, pick_device


def field_mag(arr):
    # Accepts (C,H,W) or (N,C,H,W); uses first two channels for magnitude.
    if arr.ndim == 4:
        if arr.shape[1] >= 2:
            return np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2)
        return arr[:, 0]
    if arr.ndim == 3:
        if arr.shape[0] >= 2:
            return np.sqrt(arr[0] ** 2 + arr[1] ** 2)
        return arr[0]
    raise ValueError(f"Unexpected field shape {arr.shape}")


def rel_l2(yhat, ygt):
    num = np.linalg.norm((yhat - ygt).reshape(yhat.shape[0], -1), ord=2, axis=1)
    den = np.linalg.norm(ygt.reshape(ygt.shape[0], -1), ord=2, axis=1) + 1e-12
    return num / den


def collect_preds(model, loader, norm, device):
    model.eval()
    y_mean = torch.tensor(norm["y_mean"], device=device, dtype=torch.float32)
    y_std  = torch.tensor(norm["y_std"],  device=device, dtype=torch.float32)
    preds = []
    gts   = []
    errs  = []
    with torch.no_grad():
        for x, y, xyt in loader:
            x, y, xyt = x.to(device), y.to(device), xyt.to(device)
            p = model(xyt, x).float()
            yhat = (p * y_std + y_mean).cpu().numpy()
            ygt  = (y * y_std + y_mean).cpu().numpy()
            preds.append(yhat[0])
            gts.append(ygt[0])
            errs.append(rel_l2(yhat, ygt)[0])
    return np.stack(preds, axis=0), np.stack(gts, axis=0), np.array(errs)


def raincloud(ax, df, order, palette):
    sns.violinplot(data=df, x="Model", y="rel_L2", order=order, palette=palette,
                   cut=0, inner=None, linewidth=0, ax=ax)
    sns.boxplot(data=df, x="Model", y="rel_L2", order=order,
                width=0.25, showcaps=True, boxprops={"zorder":3, "facecolor":"white"},
                whiskerprops={"linewidth":1.2}, medianprops={"color":"k","linewidth":1.4}, ax=ax)
    sns.stripplot(data=df, x="Model", y="rel_L2", order=order,
                  color="0.25", alpha=0.6, size=3.0, jitter=0.12, ax=ax)
    ax.set_ylabel(r"Relative $L_2$ Error", fontsize=13)
    ax.set_xlabel("")
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, axis="y", ls=":", lw=0.7, alpha=0.6)


def main():
    ap = argparse.ArgumentParser(description="Plasma panel: raincloud, curves, and snapshots.")
    ap.add_argument("--config", default="config_plasma_quick.json")
    ap.add_argument("--out", default="plasma_panel.png")
    ap.add_argument("--timesteps", type=int, default=6, help="Number of sample indices to show in strip.")
    args = ap.parse_args()

    cfg = json.load(open(args.config, "r"))
    device = pick_device(cfg["train"].get("device", "auto"))

    train_loader, val_loader, test_loader, norm, shapes = make_loaders(
        cfg["data"]["root"], fmt=cfg["data"].get("format", "npz"),
        batch_size=1, normalize=True,
    )

    models = ["PINN", "PIBERT"]
    preds = {}
    gts = None
    errs = {}

    for name in models:
        model = build_model(name, shapes["Cin"], shapes["Cout"], cfg.get("model_cfg", {})).to(device)
        ckpt_dir = os.path.join(cfg["eval"]["outdir"], f"{name}_seed{cfg['train'].get('seed', 42)}")
        ckpt = os.path.join(ckpt_dir, "last.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Missing checkpoint {ckpt}")
        state = torch.load(ckpt, map_location=device)["model"]
        model.load_state_dict(state, strict=False)
        p, gt_arr, e = collect_preds(model, test_loader, norm, device)
        preds[name] = p
        errs[name] = e
        if gts is None:
            gts = gt_arr

    # DataFrames for raincloud
    rows = []
    for name in models:
        for e in errs[name]:
            rows.append({"Model": name, "rel_L2": float(e)})
    df = pd.DataFrame(rows)

    # Cumulative curves (running mean)
    curves = {}
    for name in models:
        e = errs[name]
        curves[name] = np.cumsum(e) / (np.arange(len(e)) + 1)

    # pick indices for snapshots
    n = gts.shape[0]
    steps = min(args.timesteps, n)
    idxs = np.linspace(0, n - 1, steps, dtype=int)

    # plot
    sns.set_theme(style="white", context="talk")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
    })

    fig = plt.figure(figsize=(14, 10), dpi=300)
    gs = fig.add_gridspec(nrows=5, ncols=steps, height_ratios=[1.1, 1.0, 1.0, 1.0, 1.0],
                          hspace=0.35, wspace=0.05)

    # Raincloud (top)
    ax_cloud = fig.add_subplot(gs[0, :])
    raincloud(ax_cloud, df, order=models, palette=["#999999", "#D55E00"])

    # Curves
    ax_curve = fig.add_subplot(gs[1, :])
    for name, col in zip(models, ["#999999", "#D55E00"]):
        ax_curve.plot(curves[name], label=name, color=col, lw=2.2)
    ax_curve.set_xlabel("Sample index", fontsize=12)
    ax_curve.set_ylabel(r"Running mean $L_2$", fontsize=12)
    ax_curve.tick_params(labelsize=11)
    ax_curve.grid(True, ls=":", lw=0.7, alpha=0.6)
    ax_curve.legend(frameon=False, fontsize=11)

    # Snapshot strips: GT row, PINN err row, PIBERT err row
    cm_field = "jet"
    cm_err = "magma"
    gt_axes = []
    pinn_axes = []
    pib_axes = []
    # precompute vmin/vmax for errors for consistency
    pinn_err_all = np.abs(field_mag(preds["PINN"]) - field_mag(gts)).reshape(len(gts), -1)
    pib_err_all = np.abs(field_mag(preds["PIBERT"]) - field_mag(gts)).reshape(len(gts), -1)
    pinn_vmax = np.percentile(pinn_err_all, 99)
    pib_vmax = np.percentile(pib_err_all, 99)

    for c, idx in enumerate(idxs):
        # GT
        ax = fig.add_subplot(gs[2, c])
        im_gt = ax.imshow(field_mag(gts[idx]), cmap=cm_field, origin="lower", extent=(0, 1, 0, 1))
        ax.set_xticks([]); ax.set_yticks([])
        if c == 0:
            ax.set_ylabel("GT", fontsize=11)
        ax.set_title(f"t={idx}", fontsize=10)
        gt_axes.append(ax)
        # PINN err
        ax = fig.add_subplot(gs[3, c])
        err = np.abs(field_mag(preds["PINN"][idx]) - field_mag(gts[idx]))
        im_pinn = ax.imshow(err, cmap=cm_err, origin="lower", extent=(0, 1, 0, 1),
                            vmin=0, vmax=pinn_vmax)
        ax.set_xticks([]); ax.set_yticks([])
        if c == 0:
            ax.set_ylabel("PINN | Abs Err", fontsize=11)
        pinn_axes.append(ax)
        # PIBERT err
        ax = fig.add_subplot(gs[4, c])
        err = np.abs(field_mag(preds["PIBERT"][idx]) - field_mag(gts[idx]))
        im_pib = ax.imshow(err, cmap=cm_err, origin="lower", extent=(0, 1, 0, 1),
                           vmin=0, vmax=pib_vmax)
        ax.set_xticks([]); ax.set_yticks([])
        if c == 0:
            ax.set_ylabel("PIBERT | Abs Err", fontsize=11)
        pib_axes.append(ax)

    # Shared colorbars on rightmost column
    fig.colorbar(im_gt, ax=gt_axes, fraction=0.025, pad=0.01, location="right").ax.tick_params(labelsize=8)
    fig.colorbar(im_pinn, ax=pinn_axes, fraction=0.025, pad=0.01, location="right").ax.tick_params(labelsize=8)
    fig.colorbar(im_pib, ax=pib_axes, fraction=0.025, pad=0.01, location="right").ax.tick_params(labelsize=8)

    fig.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print("Saved", args.out)


if __name__ == "__main__":
    main()
