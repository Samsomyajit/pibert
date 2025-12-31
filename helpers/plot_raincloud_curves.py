#!/usr/bin/env python3
"""
PhysGTO-style raincloud + running-mean curves for three datasets:
- Cylinder Flow (results_cylinder_vort300, config_cylinder_vort300.json)
- ICP Plasma (config_plasma_quick.json)
- EAGLE spline (config_eagle_spline_quick.json)

Models: PINN vs PIBERT. Outputs a single 2-row figure with 3 columns.
Matches provided aesthetics: no jitter dots, bold spines, cool serif text.
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
    if arr.ndim == 4:
        if arr.shape[1] >= 2:
            return np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2)
        return arr[:, 0]
    if arr.ndim == 3:
        if arr.shape[0] >= 2:
            return np.sqrt(arr[0] ** 2 + arr[1] ** 2)
        return arr[0]
    raise ValueError(f"bad shape {arr.shape}")


def rel_l2(yhat, ygt):
    num = np.linalg.norm((yhat - ygt).reshape(yhat.shape[0], -1), ord=2, axis=1)
    den = np.linalg.norm(ygt.reshape(ygt.shape[0], -1), ord=2, axis=1) + 1e-12
    return num / den


def collect_preds(model, loader, norm, device):
    model.eval()
    y_mean = torch.tensor(norm["y_mean"], device=device, dtype=torch.float32)
    y_std  = torch.tensor(norm["y_std"],  device=device, dtype=torch.float32)
    preds, gts, errs = [], [], []
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


def load_models(cfg_path, models=("PINN", "PIBERT"), device_pref="auto"):
    cfg = json.load(open(cfg_path, "r"))
    device = pick_device(cfg.get("train", {}).get("device", device_pref))
    train_loader, val_loader, test_loader, norm, shapes = make_loaders(
        cfg["data"]["root"], fmt=cfg["data"].get("format", "npz"),
        batch_size=1, normalize=True,
    )
    preds, gts, errs = {}, None, {}
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
        p, gt_arr, e = collect_preds(model, test_loader, norm, device)
        preds[name] = p
        errs[name] = e
        if gts is None:
            gts = gt_arr
    return preds, gts, errs


def draw_raincloud(ax, df, order, palette):
    sns.violinplot(data=df, x="Model", y="rel_L2", hue="Model", order=order,
                   palette=palette, cut=0, inner=None, linewidth=1.2, ax=ax, legend=False, saturation=0.9)
    sns.boxplot(data=df, x="Model", y="rel_L2", order=order,
                width=0.25, showcaps=True, boxprops={"zorder":3, "facecolor":"white", "edgecolor":"0.2", "linewidth":1.6},
                whiskerprops={"linewidth":1.6, "color":"0.2"}, medianprops={"color":"0.1","linewidth":1.8},
                flierprops={"marker":"o","markersize":3,"markerfacecolor":"0.25","markeredgecolor":"0.25"},
                ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel(r"$L_2$ Errors", fontsize=13)
    ax.tick_params(axis="both", labelsize=11, width=1.4, length=5, color="0.2")
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_color("0.15")
    ax.grid(True, axis="y", ls="--", lw=0.8, alpha=0.5, color="0.5")


def draw_curve(ax, curves, palette):
    for name in curves:
        ax.plot(curves[name]["mean"], color=palette[name], lw=2.8, label=name)
    ax.set_xlabel("Timestep", fontsize=13)
    ax.set_ylabel(r"$L_2$ Errors", fontsize=13)
    ax.tick_params(labelsize=11, width=1.4, length=5, color="0.2")
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_color("0.15")
    ax.grid(True, ls="--", lw=0.8, alpha=0.5, color="0.5")


def main():
    ap = argparse.ArgumentParser(description="Combined raincloud + curves for three datasets.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="panel_stats.png")
    args = ap.parse_args()

    datasets = [
        ("Cylinder Flow", "config_cylinder_vort300.json"),
        ("ICP Plasma", "config_plasma_quick.json"),
        ("EAGLE", "config_eagle_spline_quick.json"),
    ]
    order = ["PINN", "PIBERT"]
    palette = {"PINN": "#1f77b4", "PIBERT": "#d62728"}

    sns.set_theme(style="white", context="talk")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "axes.titleweight": "bold",
    })

    fig = plt.figure(figsize=(12, 6.4), dpi=300)
    gs = fig.add_gridspec(nrows=2, ncols=3, hspace=0.40, wspace=0.25)

    for j, (title, cfg_path) in enumerate(datasets):
        preds, gts, errs = load_models(cfg_path, device_pref=args.device)
        # raincloud data
        rows = []
        for m in order:
            for e in errs[m]:
                rows.append({"Model": m, "rel_L2": float(e)})
        df = pd.DataFrame(rows)
        ax_rain = fig.add_subplot(gs[0, j])
        draw_raincloud(ax_rain, df, order, palette)
        ax_rain.set_title(title, fontsize=13)
        # curves (running mean as proxy over sample index)
        curves = {}
        for m in order:
            e = errs[m]
            curves[m] = {"mean": np.cumsum(e) / (np.arange(len(e)) + 1)}
        ax_curve = fig.add_subplot(gs[1, j])
        draw_curve(ax_curve, curves, palette)
        if j == 2:
            ax_curve.legend(frameon=False, fontsize=10, loc="upper right")

    fig.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
