#!/usr/bin/env python3
"""
PhysGTO-style strip panel (like your ref 'd'), but showing predictions instead of abs errors:

Row1: Ground Truth
Row2: PINN Prediction
Row3: PIBERT Prediction

- Grey arrow band header with timestep labels
- Optional geometry mask -> white cut-out region
- One shared colorbar for the field
- Vector + high-dpi export

Usage example:
python plot_strip_preds.py --config config_eagle_spline_quick.json \
  --out eagle_preds_panel_d \
  --cols 1 100 150 200 250 \
  --mask_channel 0 --mask_threshold 0.5 --mask_from_x
"""

import argparse, json, os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

from src.data import make_loaders
from src.runner import build_model, pick_device


def pick_field(arr, channel=0, use_mag=False):
    """
    arr: (N,C,H,W) or (C,H,W)
    Returns: (N,H,W) if N exists else (H,W)
    """
    if arr.ndim == 4:
        if use_mag and arr.shape[1] >= 2:
            return np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2)
        return arr[:, channel]
    if arr.ndim == 3:
        if use_mag and arr.shape[0] >= 2:
            return np.sqrt(arr[0] ** 2 + arr[1] ** 2)
        return arr[channel]
    raise ValueError(f"Unexpected shape {arr.shape}")


def collect_preds_and_x(model, loader, norm, device):
    model.eval()
    y_mean = torch.tensor(norm["y_mean"], device=device, dtype=torch.float32)
    y_std  = torch.tensor(norm["y_std"],  device=device, dtype=torch.float32)

    preds, gts, xs = [], [], []
    with torch.no_grad():
        for x, y, xyt in loader:
            x, y, xyt = x.to(device), y.to(device), xyt.to(device)
            p = model(xyt, x).float()

            yhat = (p * y_std + y_mean).cpu().numpy()
            ygt  = (y * y_std + y_mean).cpu().numpy()

            preds.append(yhat[0])  # (C,H,W)
            gts.append(ygt[0])     # (C,H,W)
            xs.append(x.cpu().numpy()[0])  # (Cin,H,W)

    return np.stack(preds, 0), np.stack(gts, 0), np.stack(xs, 0)


def load_models(cfg_path, models=("PINN", "PIBERT"), device_pref="auto"):
    cfg = json.load(open(cfg_path, "r"))
    device = pick_device(cfg.get("train", {}).get("device", device_pref))

    _, _, test_loader, norm, shapes = make_loaders(
        cfg["data"]["root"],
        fmt=cfg["data"].get("format", "npz"),
        batch_size=1,
        normalize=True,
    )

    preds, gts, xs = {}, None, None
    for name in models:
        model = build_model(name, shapes["Cin"], shapes["Cout"], cfg.get("model_cfg", {})).to(device)

        seed = cfg["train"].get("seed", 42)
        ckpt_dir = os.path.join(cfg["eval"]["outdir"], f"{name}_seed{seed}")
        ckpt = os.path.join(ckpt_dir, "last.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Missing checkpoint {ckpt}")

        raw_state = torch.load(ckpt, map_location=device)["model"]
        current = model.state_dict()
        state = {k: v for k, v in raw_state.items() if k in current and current[k].shape == v.shape}
        model.load_state_dict(state, strict=False)

        p, gt_arr, x_arr = collect_preds_and_x(model, test_loader, norm, device)
        preds[name] = p
        if gts is None:
            gts = gt_arr
        if xs is None:
            xs = x_arr

    return preds, gts, xs


def add_arrow_band(fig, left, right, y, h, labels, title="Timestep"):
    """
    Draws the grey arrow band like the reference figure.
    labels: list[str] per column (e.g. ["+1","+100",...])
    """
    axb = fig.add_axes([left, y, right - left, h])
    axb.set_axis_off()

    # Background rectangle + arrow head
    rect = Rectangle((0, 0), 0.96, 1.0, transform=axb.transAxes, facecolor="0.7", edgecolor="none")
    tri  = Polygon([[0.96, 0], [1.0, 0.5], [0.96, 1.0]],
                   transform=axb.transAxes, closed=True, facecolor="0.7", edgecolor="none")
    axb.add_patch(rect)
    axb.add_patch(tri)

    # Title on left
    axb.text(0.02, 0.5, title, va="center", ha="left",
             fontsize=16, fontweight="bold", color="white", transform=axb.transAxes)

    # Column labels
    K = len(labels)
    for i, lab in enumerate(labels):
        x = 0.18 + (0.76) * (i + 0.5) / K
        axb.text(x, 0.5, lab, va="center", ha="center",
                 fontsize=16, fontweight="bold", color="white", transform=axb.transAxes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_eagle_spline_quick.json")
    ap.add_argument("--out", default="panel_preds")
    ap.add_argument("--device", default="auto")

    # which frames/indices to show
    ap.add_argument("--cols", type=int, nargs="+", default=[1, 100, 150, 200, 250],
                    help="Indices (timesteps / sample indices) to display as columns.")
    ap.add_argument("--col_labels", type=str, nargs="+", default=None,
                    help="Optional labels printed in the grey band. If omitted, uses '+{idx}'.")

    # field options
    ap.add_argument("--channel", type=int, default=0, help="Which output channel to visualize.")
    ap.add_argument("--use_mag", action="store_true", help="Use magnitude if vector field.")
    ap.add_argument("--cmap", default="coolwarm")
    ap.add_argument("--symmetric", action="store_true",
                    help="Use symmetric vmin/vmax around 0 (good for signed pressure).")

    # mask options (to get the white cut-out like your ref)
    ap.add_argument("--mask_from_x", action="store_true",
                    help="Build mask from input x channels (recommended if geometry mask exists).")
    ap.add_argument("--mask_channel", type=int, default=0,
                    help="Which x channel is the geometry mask (if --mask_from_x).")
    ap.add_argument("--mask_threshold", type=float, default=0.5,
                    help="Mask threshold: values < threshold become white (masked).")
    args = ap.parse_args()

    preds, gts, xs = load_models(args.config, device_pref=args.device)

    # Build displayed arrays
    gt = pick_field(gts, channel=args.channel, use_mag=args.use_mag)              # (N,H,W)
    pinn = pick_field(preds["PINN"], channel=args.channel, use_mag=args.use_mag)  # (N,H,W)
    pib  = pick_field(preds["PIBERT"], channel=args.channel, use_mag=args.use_mag)

    N = gt.shape[0]
    idxs = [int(i) for i in args.cols if 0 <= int(i) < N]
    if len(idxs) == 0:
        raise ValueError(f"All --cols are out of range. N={N}")

    # Mask (optional)
    mask = None
    if args.mask_from_x:
        m = xs[:, args.mask_channel]  # (N,H,W)
        mask = (m < args.mask_threshold)
    else:
        # fallback: if GT has NaNs, mask them
        if np.isnan(gt).any():
            mask = ~np.isfinite(gt)

    # Shared color limits (GT + preds combined for fair comparison)
    stack = np.concatenate([gt[idxs].reshape(len(idxs), -1),
                            pinn[idxs].reshape(len(idxs), -1),
                            pib[idxs].reshape(len(idxs), -1)], axis=1).reshape(-1)
    stack = stack[np.isfinite(stack)]
    v1, v99 = np.percentile(stack, [1, 99])

    if args.symmetric:
        vmax = max(abs(v1), abs(v99))
        vmin = -vmax
    else:
        vmin, vmax = v1, v99

    # Colormap with white for masked region
    cmap = plt.get_cmap(args.cmap).copy()
    cmap.set_bad(color="white")

    # --- Figure layout like panel d ---
    K = len(idxs)
    fig = plt.figure(figsize=(3.2*K + 3.2, 6.2), dpi=300, facecolor="white")
    gs = fig.add_gridspec(nrows=3, ncols=K, hspace=0.08, wspace=0.04)

    # panel letter
    fig.text(0.01, 0.95, "c", fontsize=22, fontweight="bold", va="top")

    # grey arrow band
    labels = args.col_labels if args.col_labels is not None else [f"+{i}" for i in idxs]
    left = 0.08
    right = 0.88
    add_arrow_band(fig, left=left, right=right, y=0.89, h=0.07, labels=labels, title="Timestep")

    # Row labels (left)
    fig.text(0.02, 0.70, "Ground\nTruth", ha="left", va="center", fontsize=16, fontweight="bold")
    fig.text(0.02, 0.44, "PINN\nPrediction", ha="left", va="center", fontsize=16, fontweight="bold")
    fig.text(0.02, 0.18, "PIBERT\nPrediction", ha="left", va="center", fontsize=16, fontweight="bold")

    # Plot images
    im_ref = None
    for c, idx in enumerate(idxs):
        # prepare masked arrays
        def M(a):
            if mask is None:
                return a
            return np.ma.array(a, mask=mask[idx])

        ax = fig.add_subplot(gs[0, c])
        im_ref = ax.imshow(M(gt[idx]), cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(1.2); sp.set_color("0.2")

        ax = fig.add_subplot(gs[1, c])
        ax.imshow(M(pinn[idx]), cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(1.2); sp.set_color("0.2")

        ax = fig.add_subplot(gs[2, c])
        ax.imshow(M(pib[idx]), cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(1.2); sp.set_color("0.2")

    # One shared colorbar on the right (like a single scale for all rows)
    cax = fig.add_axes([0.90, 0.20, 0.015, 0.62])
    cb = fig.colorbar(im_ref, cax=cax)
    cb.ax.tick_params(labelsize=11, width=1.0)
    cb.set_label("Field", fontsize=13)

    # # Title (optional, if you want like “EAGLE Pressure …”)
    # fig.suptitle("EAGLE Pressure: Ground Truth and Predictions", y=0.995,
    #              fontsize=18, fontweight="bold")

    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    fig.savefig(args.out + ".png", bbox_inches="tight", dpi=600)
    print(f"[saved] {args.out}.pdf and {args.out}.png")


if __name__ == "__main__":
    main()
