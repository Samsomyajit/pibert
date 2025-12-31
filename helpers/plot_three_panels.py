#!/usr/bin/env python3
"""
Generate PhysGTO-style panels (raincloud, running-mean curves, GT/pred abs-error strips)
for CFDBench Cylinder, ICP Plasma, and EAGLE (spline subset).

Each panel shows PINN vs PIBERT with consistent color scales.
"""
import argparse, json, os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from src.data import make_loaders
from src.runner import build_model, pick_device


def field_mag(arr):
    """
    Supports (N,C,H,W) or (C,H,W). Uses first two channels as vector magnitude,
    otherwise returns the first channel (scalar).
    """
    if arr.ndim == 4:
        if arr.shape[1] >= 2:
            return np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2)
        return arr[:, 0]
    if arr.ndim == 3:
        if arr.shape[0] >= 2:
            return np.sqrt(arr[0] ** 2 + arr[1] ** 2)
        return arr[0]
    raise ValueError(f"Unsupported field shape {arr.shape}")


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
    sns.violinplot(data=df, x="Model", y="rel_L2", hue="Model", order=order,
                   palette=palette, cut=0, inner=None, linewidth=0, ax=ax, legend=False)
    sns.boxplot(data=df, x="Model", y="rel_L2", order=order,
                width=0.25, showcaps=True, boxprops={"zorder":3, "facecolor":"white"},
                whiskerprops={"linewidth":1.2}, medianprops={"color":"k","linewidth":1.4}, ax=ax)
    sns.stripplot(data=df, x="Model", y="rel_L2", hue="Model", order=order,
                  palette=palette, alpha=0.7, size=3.0, jitter=0.10, ax=ax, dodge=False, legend=False)
    ax.set_ylabel(r"$L_2$ Errors", fontsize=12)
    ax.set_xlabel("")
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, axis="y", ls=":", lw=0.7, alpha=0.6)


def make_panel(label, cfg_path, out_path, timesteps=4, device_pref="auto"):
    cfg = json.load(open(cfg_path, "r"))
    device = pick_device(cfg.get("train", {}).get("device", device_pref))

    train_loader, val_loader, test_loader, norm, shapes = make_loaders(
        cfg["data"]["root"], fmt=cfg["data"].get("format", "npz"),
        batch_size=1, normalize=True,
    )

    models = ["PINN", "PIBERT"]
    palette = {"PINN": "#2AA198", "PIBERT": "#D55E00"}
    preds = {}
    gts = None
    errs = {}

    for name in models:
        model = build_model(name, shapes["Cin"], shapes["Cout"], cfg.get("model_cfg", {})).to(device)
        ckpt_dir = os.path.join(cfg["eval"]["outdir"], f"{name}_seed{cfg['train'].get('seed', 42)}")
        ckpt = os.path.join(ckpt_dir, "last.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Missing checkpoint {ckpt}")
        raw_state = torch.load(ckpt, map_location=device)["model"]
        current = model.state_dict()
        state = {k: v for k, v in raw_state.items() if k in current and current[k].shape == v.shape}
        dropped = sorted(set(raw_state.keys()) - set(state.keys()))
        if dropped:
            print(f"[WARN] Dropping {len(dropped)} mismatched keys for {name}: {dropped[:3]}{'...' if len(dropped)>3 else ''}")
        model.load_state_dict(state, strict=False)
        p, gt_arr, e = collect_preds(model, test_loader, norm, device)
        preds[name] = p
        errs[name] = e
        if gts is None:
            gts = gt_arr

    # DataFrame for raincloud
    rows = []
    for name in models:
        for e in errs[name]:
            rows.append({"Model": name, "rel_L2": float(e)})
    import pandas as pd
    df = pd.DataFrame(rows)

    # Running-mean curves
    curves = {name: np.cumsum(errs[name]) / (np.arange(len(errs[name])) + 1) for name in models}

    # Snapshot indices
    n = gts.shape[0]
    steps = min(timesteps, n)
    idxs = np.linspace(0, n - 1, steps, dtype=int)

    sns.set_theme(style="white", context="talk")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
    })

    fig = plt.figure(figsize=(12, 9), dpi=300)
    gs = fig.add_gridspec(nrows=5, ncols=steps, height_ratios=[1.1, 1.0, 1.0, 1.0, 1.0],
                          hspace=0.32, wspace=0.05)

    # Raincloud (full width)
    ax_cloud = fig.add_subplot(gs[0, :])
    raincloud(ax_cloud, df, order=models, palette=palette)
    ax_cloud.set_title(label, fontsize=14)

    # Curves
    ax_curve = fig.add_subplot(gs[1, :])
    for name in models:
        ax_curve.plot(curves[name], label=name, color=palette[name], lw=2.2)
    ax_curve.set_xlabel("Sample index", fontsize=12)
    ax_curve.set_ylabel(r"Running mean $L_2$", fontsize=12)
    ax_curve.tick_params(labelsize=11)
    ax_curve.grid(True, ls=":", lw=0.7, alpha=0.6)
    ax_curve.legend(frameon=False, fontsize=11)

    cm_field = "coolwarm"
    gt_axes = []
    pinn_axes = []
    pib_axes = []

    # shared vmin/vmax across GT and predictions
    all_fields = np.concatenate([
        field_mag(gts).reshape(len(gts), -1),
        field_mag(preds["PINN"]).reshape(len(gts), -1),
        field_mag(preds["PIBERT"]).reshape(len(gts), -1),
    ], axis=1)
    f_vmin = np.percentile(all_fields, 1)
    f_vmax = np.percentile(all_fields, 99)

    for c, idx in enumerate(idxs):
        # GT
        ax = fig.add_subplot(gs[2, c])
        im_gt = ax.imshow(field_mag(gts[idx]), cmap=cm_field, origin="lower", extent=(0, 1, 0, 1),
                          vmin=f_vmin, vmax=f_vmax)
        ax.set_xticks([]); ax.set_yticks([])
        if c == 0:
            ax.set_ylabel("Ground Truth", fontsize=11)
        ax.set_title(f"t={idx}", fontsize=10)
        gt_axes.append(ax)

        # PINN prediction
        ax = fig.add_subplot(gs[3, c])
        im_pinn = ax.imshow(field_mag(preds["PINN"][idx]), cmap=cm_field, origin="lower", extent=(0, 1, 0, 1),
                            vmin=f_vmin, vmax=f_vmax)
        ax.set_xticks([]); ax.set_yticks([])
        if c == 0:
            ax.set_ylabel("PINN Pred", fontsize=11)
        pinn_axes.append(ax)

        # PIBERT prediction
        ax = fig.add_subplot(gs[4, c])
        im_pib = ax.imshow(field_mag(preds["PIBERT"][idx]), cmap=cm_field, origin="lower", extent=(0, 1, 0, 1),
                           vmin=f_vmin, vmax=f_vmax)
        ax.set_xticks([]); ax.set_yticks([])
        if c == 0:
            ax.set_ylabel("PIBERT Pred", fontsize=11)
        pib_axes.append(ax)

    # Colorbars (rightmost column)
    fig.colorbar(im_gt, ax=gt_axes, fraction=0.025, pad=0.01, location="right").ax.tick_params(labelsize=8)
    fig.colorbar(im_pinn, ax=pinn_axes, fraction=0.025, pad=0.01, location="right").ax.tick_params(labelsize=8)
    fig.colorbar(im_pib, ax=pib_axes, fraction=0.025, pad=0.01, location="right").ax.tick_params(labelsize=8)

    fig.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    print(f"[saved] {out_path}")


def main():
    ap = argparse.ArgumentParser(description="PhysGTO-style panels for CFDBench, Plasma, and EAGLE.")
    ap.add_argument("--timesteps", type=int, default=4)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    panels = [
        ("Cylinder Flow", "config_cylinder.json", "panel_cfdb.png"),
        ("ICP Plasma", "config_plasma_quick.json", "panel_plasma.png"),
        ("EAGLE (spline)", "config_eagle_spline_quick.json", "panel_eagle.png"),
    ]

    for label, cfg_path, out_path in panels:
        if not os.path.exists(cfg_path):
            print(f"[skip] missing {cfg_path}")
            continue
        make_panel(label, cfg_path, out_path, timesteps=args.timesteps, device_pref=args.device)


if __name__ == "__main__":
    main()
