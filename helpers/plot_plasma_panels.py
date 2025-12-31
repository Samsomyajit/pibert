#!/usr/bin/env python3
"""
Quick comparison panels for plasma snapshots:
  - Ground truth
  - PINN prediction + abs error
  - PIBERT prediction + abs error

Inputs are sample_XXX.npz files saved by runner (contain gt and pred).

Example:
python plot_plasma_panels.py \
  --pinn results_plasma_quick/PINN_seed42/sample_000.npz \
  --pibert results_plasma_quick/PIBERT_seed42/sample_000.npz \
  --out plasma_compare.png
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_sample(path):
    d = np.load(path)
    gt = d["gt"]  # (C,H,W)
    pr = d["pred"]
    return gt.astype(np.float32), pr.astype(np.float32)


def field_mag(arr):
    if arr.shape[0] >= 2:
        return np.sqrt(arr[0] ** 2 + arr[1] ** 2)
    return arr[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinn", required=True, help="sample_XXX.npz for PINN")
    ap.add_argument("--pibert", required=True, help="sample_XXX.npz for PIBERT")
    ap.add_argument("--out", default="plasma_compare.png")
    args = ap.parse_args()

    sns.set_theme(style="white", context="talk")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
    })

    gt_pinn, pr_pinn = load_sample(args.pinn)
    gt_pib, pr_pib = load_sample(args.pibert)

    # assume GT identical; use first
    gt = gt_pinn
    mag_gt = field_mag(gt)
    mag_pinn = field_mag(pr_pinn)
    mag_pib = field_mag(pr_pib)

    err_pinn = np.abs(mag_pinn - mag_gt)
    err_pib = np.abs(mag_pib - mag_gt)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), dpi=300)
    cm_field = "jet"
    cm_err = "magma"

    panels = [
        (mag_gt, "Ground Truth", cm_field),
        (mag_pinn, "PINN | Veloc", cm_field),
        (err_pinn, "PINN | Abs Err", cm_err),
        (mag_gt, "Ground Truth", cm_field),
        (mag_pib, "PIBERT | Veloc", cm_field),
        (err_pib, "PIBERT | Abs Err", cm_err),
    ]

    for ax, (im, title, cmap) in zip(axes.flat, panels):
        h = ax.imshow(im, cmap=cmap, origin="lower", extent=(0, 1, 0, 1))
        ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
        ax.tick_params(labelsize=10, width=0.9, colors="black")
        ax.set_xlabel("X", fontsize=12); ax.set_ylabel("Y", fontsize=12)
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
        cbar = fig.colorbar(h, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(title, fontsize=11)
        cbar.ax.tick_params(labelsize=9, width=0.8, colors="black")

    fig.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print("Saved", args.out)


if __name__ == "__main__":
    main()
