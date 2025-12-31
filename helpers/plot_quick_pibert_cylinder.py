import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def vorticity_from_uv(u, v, dx=1.0, dy=1.0):
    """
    Compute scalar vorticity (dz component) omega = dv/dx - du/dy using
    central differences with simple one-sided boundaries.
    """
    dudx = np.zeros_like(u)
    dudy = np.zeros_like(u)
    dvdx = np.zeros_like(v)
    dvdy = np.zeros_like(v)

    dudx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
    dudy[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dy)
    dvdx[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / (2 * dx)
    dvdy[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2 * dy)

    # one-sided boundaries
    dudx[:, 0] = (u[:, 1] - u[:, 0]) / dx
    dudx[:, -1] = (u[:, -1] - u[:, -2]) / dx
    dudy[0, :] = (u[1, :] - u[0, :]) / dy
    dudy[-1, :] = (u[-1, :] - u[-2, :]) / dy

    dvdx[:, 0] = (v[:, 1] - v[:, 0]) / dx
    dvdx[:, -1] = (v[:, -1] - v[:, -2]) / dx
    dvdy[0, :] = (v[1, :] - v[0, :]) / dy
    dvdy[-1, :] = (v[-1, :] - v[-2, :]) / dy

    return dvdx - dudy


def load_sample(root: Path, split: str, idx: int):
    npz_path = root / f"sample_{idx:03d}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} not found")
    data = np.load(npz_path)
    # expected keys: 'gt' (C,H,W), 'pred' (C,H,W)
    gt = data["gt"]
    pred = data["pred"]
    return gt, pred


def plot_velocity(gt, pred, out_path):
    u_gt, v_gt = gt[0], gt[1]
    u_pr, v_pr = pred[0], pred[1]
    eps = 1e-8
    rel_u = np.abs(u_pr - u_gt) / (np.abs(u_gt) + eps)
    rel_v = np.abs(v_pr - v_gt) / (np.abs(v_gt) + eps)

    fig, axes = plt.subplots(
        2, 3, figsize=(12, 8), constrained_layout=True, sharex=True, sharey=True
    )
    titles = ["u GT", "u Pred", r"$|u-\hat{u}|/|u_{gt}|$", "v GT", "v Pred", r"$|v-\hat{v}|/|v_{gt}|$"]
    fields = [u_gt, u_pr, rel_u, v_gt, v_pr, rel_v]
    cmaps = ["coolwarm", "coolwarm", "magma", "coolwarm", "coolwarm", "magma"]

    for ax, fld, title, cmap in zip(axes.flat, fields, titles, cmaps):
        im = ax.imshow(fld, cmap=cmap, origin="lower")
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        cax = ax.inset_axes([1.02, 0.05, 0.04, 0.9])
        plt.colorbar(im, cax=cax)

    fig.suptitle("Velocity (u,v) — GT vs Pred vs RelErr", fontsize=14)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_vorticity(gt, pred, out_path, clip_percentile=None):
    u_gt, v_gt = gt[0], gt[1]
    u_pr, v_pr = pred[0], pred[1]
    omega_gt = vorticity_from_uv(u_gt, v_gt)
    omega_pr = vorticity_from_uv(u_pr, v_pr)
    eps = 1e-8
    rel = np.abs(omega_pr - omega_gt) / (np.abs(omega_gt) + eps)

    if clip_percentile:
        lo = np.percentile(omega_gt, 100 - clip_percentile)
        hi = np.percentile(omega_gt, clip_percentile)
    else:
        lo = hi = None

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True, sharex=True, sharey=True)
    titles = [r"$\omega_{gt}$", r"$\hat{\omega}$", r"$|\omega-\hat{\omega}|/|\omega_{gt}|$"]
    fields = [omega_gt, omega_pr, rel]
    cmaps = ["coolwarm", "coolwarm", "magma"]

    for ax, fld, title, cmap in zip(axes, fields, titles, cmaps):
        vmin = lo if (lo is not None and ax is axes[0]) else None
        vmax = hi if (hi is not None and ax is axes[0]) else None
        im = ax.imshow(fld, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        cax = ax.inset_axes([1.02, 0.05, 0.04, 0.9])
        plt.colorbar(im, cax=cax)

    fig.suptitle("Vorticity — GT vs Pred vs RelErr", fontsize=14)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Quick PIBERT cylinder plots (velocity & vorticity)")
    parser.add_argument("--root", type=str, default="results_cylinder_pibert_200/PIBERT_seed42",
                        help="Path containing sample_XXX.npz")
    parser.add_argument("--sample", type=int, default=0, help="Sample index to plot")
    parser.add_argument("--outdir", type=str, default="Full_PIBERT_analysis/quick_plots",
                        help="Output directory for figures")
    parser.add_argument("--clip_percentile", type=float, default=None,
                        help="If set (e.g., 99), clip omega_gt to [100-p, p] percentiles for display")
    args = parser.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gt, pred = load_sample(root, split="val", idx=args.sample)

    vel_path = outdir / f"velocity_sample{args.sample:03d}.png"
    vort_path = outdir / f"vorticity_sample{args.sample:03d}.png"
    plot_velocity(gt, pred, vel_path)
    plot_vorticity(gt, pred, vort_path, clip_percentile=args.clip_percentile)
    print(f"Saved: {vel_path}")
    print(f"Saved: {vort_path}")


if __name__ == "__main__":
    main()
