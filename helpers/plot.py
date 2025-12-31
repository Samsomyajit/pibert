#!/usr/bin/env python3
"""
PhysGTO-style raincloud (half violin + box + right-side points) + error curves.
Models: PINN vs PIBERT. Outputs a single 2-row figure with N columns.
"""

import argparse, json, os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.data import make_loaders
from src.runner import build_model, pick_device


def rel_l2(yhat, ygt):
    num = np.linalg.norm((yhat - ygt).reshape(yhat.shape[0], -1), ord=2, axis=1)
    den = np.linalg.norm(ygt.reshape(ygt.shape[0], -1), ord=2, axis=1) + 1e-12
    return num / den


def collect_errs(model, loader, norm, device):
    model.eval()
    y_mean = torch.tensor(norm["y_mean"], device=device, dtype=torch.float32)
    y_std  = torch.tensor(norm["y_std"],  device=device, dtype=torch.float32)
    errs = []
    with torch.no_grad():
        for x, y, xyt in loader:
            x, y, xyt = x.to(device), y.to(device), xyt.to(device)
            p = model(xyt, x).float()
            yhat = (p * y_std + y_mean).cpu().numpy()
            ygt  = (y * y_std + y_mean).cpu().numpy()
            errs.append(float(rel_l2(yhat, ygt)[0]))
    return np.array(errs)


def load_one(cfg_path, model_name, seed_override=None, device_pref="auto"):
    cfg = json.load(open(cfg_path, "r"))
    if seed_override is not None:
        cfg.setdefault("train", {})
        cfg["train"]["seed"] = int(seed_override)

    device = pick_device(cfg.get("train", {}).get("device", device_pref))

    train_loader, val_loader, test_loader, norm, shapes = make_loaders(
        cfg["data"]["root"],
        fmt=cfg["data"].get("format", "npz"),
        batch_size=1,
        normalize=True,
    )

    model = build_model(model_name, shapes["Cin"], shapes["Cout"], cfg.get("model_cfg", {})).to(device)

    seed = cfg["train"].get("seed", 42)
    ckpt_dir = os.path.join(cfg["eval"]["outdir"], f"{model_name}_seed{seed}")
    ckpt = os.path.join(ckpt_dir, "last.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Missing checkpoint {ckpt}")

    raw_state = torch.load(ckpt, map_location=device)["model"]
    current = model.state_dict()
    state = {k: v for k, v in raw_state.items() if k in current and current[k].shape == v.shape}
    model.load_state_dict(state, strict=False)

    errs = collect_errs(model, test_loader, norm, device)
    return errs


# ---------- PhysGTO-style raincloud (matplotlib) ----------

def _half_violin(ax, data, pos, color, width=0.85, side="left", alpha=1.0):
    parts = ax.violinplot([data], positions=[pos], widths=width,
                          showmeans=False, showmedians=False, showextrema=False)
    body = parts["bodies"][0]
    body.set_facecolor(color)
    body.set_edgecolor("none")
    body.set_alpha(alpha)

    # Clip to half (left or right)
    path = body.get_paths()[0]
    verts = path.vertices
    if side == "left":
        verts[verts[:, 0] > pos, 0] = pos
    else:
        verts[verts[:, 0] < pos, 0] = pos
    path.vertices = verts

    return parts


def draw_raincloud(ax, df, order, colors):
    # positions: 0,1,... for each model
    positions = np.arange(len(order), dtype=float)

    # half violins + box + right-side points
    for i, m in enumerate(order):
        vals = df.loc[df["Model"] == m, "rel_L2"].values

        _half_violin(ax, vals, pos=positions[i], color=colors[m], width=0.85, side="left", alpha=0.95)

        ax.boxplot([vals], positions=[positions[i]], widths=0.18, patch_artist=True,
                   showfliers=True,
                   boxprops=dict(facecolor="white", edgecolor="0.2", linewidth=1.6),
                   medianprops=dict(color="0.2", linewidth=2.0),
                   whiskerprops=dict(color="0.2", linewidth=1.6),
                   capprops=dict(color="0.2", linewidth=1.6),
                   flierprops=dict(marker="d", markersize=4, markerfacecolor="0.2",
                                   markeredgecolor="0.2", alpha=0.9))

        # points on right side
        rng = np.random.default_rng(12345 + i)
        x = positions[i] + rng.uniform(0.08, 0.30, size=len(vals))
        ax.scatter(x, vals, s=16, c=colors[m], alpha=0.95, edgecolors="none")

    ax.set_xticks(positions)
    ax.set_xticklabels(order, fontsize=13)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel(r"$L_2$ Errors", fontsize=14)

    # PhysGTO-like clean axes
    ax.grid(True, axis="y", linestyle="-", linewidth=0.8, alpha=0.25)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", labelsize=12, width=1.0)

    # show ×10^k or 1e7 when needed (matches your reference style)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax.yaxis.get_offset_text().set_size(12)


def draw_curves(ax, curves_mean, curves_std, order, colors, show_legend=False):
    for m in order:
        y = curves_mean[m]
        x = np.arange(len(y))
        ax.plot(x, y, lw=3.0, color=colors[m], label=m)

        if curves_std is not None and curves_std.get(m, None) is not None:
            s = curves_std[m]
            ax.fill_between(x, y - s, y + s, color=colors[m], alpha=0.18, linewidth=0)

    ax.set_xlabel("Timestep", fontsize=14)
    ax.set_ylabel(r"$L_2$ Errors", fontsize=14)
    ax.grid(True, linestyle="-", linewidth=0.8, alpha=0.25)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", labelsize=12, width=1.0)

    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax.yaxis.get_offset_text().set_size(12)

    if show_legend:
        ax.legend(frameon=False, fontsize=13, loc="upper right")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="panel_stats_physgto")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="Optional: multiple seeds to compute mean±std bands (e.g. --seeds 0 1 2 3)")
    args = ap.parse_args()

    datasets = [
        ("Cylinder Flow", "config_cylinder_vort300.json"),
        ("ICP Plasma",    "config_plasma_quick.json"),
        ("EAGLE",         "config_eagle_spline_quick.json"),
    ]

    order = ["PINN", "PIBERT"]
    colors = {"PINN": "#1f77b4", "PIBERT": "#d62728"}

    # Typography close to the paper look
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.titleweight": "regular",
    })

    ncol = len(datasets)
    fig = plt.figure(figsize=(5.2 * ncol, 6.2), dpi=300)
    gs = fig.add_gridspec(2, ncol, hspace=0.38, wspace=0.28)

    # panel letters like the reference
    fig.text(0.01, 0.98, "a", fontsize=18, fontweight="bold", va="top")
    fig.text(0.01, 0.49, "b", fontsize=18, fontweight="bold", va="top")

    for j, (title, cfg_path) in enumerate(datasets):
        # Load errors (optionally multiple seeds for bands)
        err_runs = {m: [] for m in order}
        if args.seeds is None or len(args.seeds) == 0:
            for m in order:
                err_runs[m].append(load_one(cfg_path, m, seed_override=None, device_pref=args.device))
        else:
            for seed in args.seeds:
                for m in order:
                    err_runs[m].append(load_one(cfg_path, m, seed_override=seed, device_pref=args.device))

        # ---- Raincloud dataframe ----
        rows = []
        for m in order:
            for e in err_runs[m][0]:  # distribution plot: use first run (or change to concat all runs)
                rows.append({"Model": m, "rel_L2": float(e)})
        df = pd.DataFrame(rows)

        ax_top = fig.add_subplot(gs[0, j])
        draw_raincloud(ax_top, df, order, colors)
        ax_top.set_title(title, fontsize=18, pad=6)

        # ---- Curves ----
        curves_mean, curves_std = {}, {}
        for m in order:
            # stack runs -> mean±std
            # NOTE: assumes same length across runs
            stack = np.stack(err_runs[m], axis=0)
            curves_mean[m] = stack.mean(axis=0)
            curves_std[m]  = stack.std(axis=0) if stack.shape[0] > 1 else None

        ax_bot = fig.add_subplot(gs[1, j])
        draw_curves(ax_bot, curves_mean, curves_std, order, colors, show_legend=(j == ncol - 1))

    # Save as both vector + raster
    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    fig.savefig(args.out + ".png", bbox_inches="tight", dpi=600)
    print(f"[saved] {args.out}.pdf and {args.out}.png")


if __name__ == "__main__":
    main()
