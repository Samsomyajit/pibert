#!/usr/bin/env python3
"""
Visualize encoder exports (ff/wv/fuse/tok) alongside GT/Pred from embed_XXX.npz.
"""
import argparse, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import font_manager
print(font_manager.findfont("DejaVu Serif"))


plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
})
sns.set_theme(style="white", context="talk")


def field_mag(arr):
    """Return magnitude map from (C,H,W) array."""
    arr = np.asarray(arr, float)
    if arr.ndim != 3:
        raise ValueError("expected (C,H,W)")
    if arr.shape[0] >= 2:
        return np.sqrt((arr[:2] ** 2).sum(axis=0))
    return arr[0]


def chan_norm(arr):
    """L2 over channels for (C,H,W)."""
    arr = np.asarray(arr, float)
    if arr.ndim != 3:
        raise ValueError("expected (C,H,W)")
    return np.sqrt((arr ** 2).sum(axis=0))


def _robust_minmax(im, lo=2.0, hi=98.0):
    vals = im[np.isfinite(im)].ravel()
    if vals.size == 0:
        return None, None
    return float(np.percentile(vals, lo)), float(np.percentile(vals, hi))


def plot_embed(npz_path, out_path=None, relerr_eps="p99", clamp_relerr=99.0):
    d = np.load(npz_path)
    gt = d["gt"]
    pr = d["pred"]
    ff = d["ff"]
    wv = d["wv"]
    fuse = d["fuse"]
    tok_up = d["tok_up"]

    mag_gt = field_mag(gt)
    mag_pr = field_mag(pr)

    ff_n = chan_norm(ff)
    wv_n = chan_norm(wv)
    fuse_n = chan_norm(fuse)
    tok_n = chan_norm(tok_up)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=300)
    cm_field = "rainbow_r"      # brighter rainbow for GT/Pred
    cm_feat = "coolwarm"  # features/error-style maps

    panels = [
        {"im": mag_gt,  "label": r"Velocity (GT)",   "cmap": cm_field, "kind": "field"},
        {"im": mag_pr,  "label": r"Velocity (Pred)", "cmap": cm_field, "kind": "field"},
        {"im": ff_n,    "label": r"‖Fourier‖",       "cmap": cm_feat,  "kind": "feat"},
        {"im": wv_n,    "label": r"‖Wavelet‖",       "cmap": cm_feat,  "kind": "feat"},
        {"im": fuse_n,  "label": r"‖Fuse‖",          "cmap": cm_feat,  "kind": "feat"},
        {"im": tok_n,   "label": r"‖Tokens↑‖",       "cmap": cm_feat,  "kind": "feat"},
    ]
    panel_tags = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    for ax, meta, tag in zip(axes.flat, panels, panel_tags):
        im = meta["im"]; cmap = meta["cmap"]
        vmin, vmax = _robust_minmax(im, 2.0, 98.0)
        h = ax.imshow(im, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax, extent=(0, 1, 0, 1))
        ax.set_xticks([0.0, 0.5, 1.0]); ax.set_yticks([0.0, 0.5, 1.0])
        ax.tick_params(labelsize=11, width=0.9, colors="black")
        ax.set_xlabel("X", fontsize=12); ax.set_ylabel("Y", fontsize=12)
        # keep borders light
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
        ax.text(-0.12, 1.04, tag, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=13, fontweight="normal",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.0, pad=0.0))
        cbar = fig.colorbar(h, ax=ax, fraction=0.04, pad=0.02, shrink=0.65)
        cbar.set_label(meta["label"], fontsize=11)
        cbar.ax.tick_params(labelsize=9, width=0.7, colors="black")

    fig.tight_layout()
    out = out_path or (os.path.splitext(npz_path)[0] + "_viz.png")
    plt.savefig(out, bbox_inches="tight")
    print("Saved", out)


def main():
    ap = argparse.ArgumentParser(description="Visualize one embed_XXX.npz export")
    ap.add_argument("--npz", required=True, help="Path to embed_XXX.npz produced by export_encoder_embeddings.py")
    ap.add_argument("--out", default=None, help="Output image path (defaults to *_viz.png)")
    ap.add_argument("--relerr-eps", default="p99",
                    help="Epsilon for relative error: float or 'pXX' to use XXth percentile of |GT| times 1e-3.")
    ap.add_argument("--relerr-clamp", type=float, default=99.0,
                    help="Percentile to clamp relerr for colormap (0 to disable).")
    args = ap.parse_args()
    plot_embed(args.npz, args.out, relerr_eps=args.relerr_eps, clamp_relerr=args.relerr_clamp)


if __name__ == "__main__":
    main()
