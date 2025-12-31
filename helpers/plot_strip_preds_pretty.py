#!/usr/bin/env python3
"""
PhysGTO-style strip panel showing predictions (not abs error):

Row1: Ground Truth
Row2: PINN Prediction
Row3: PIBERT Prediction

Key fixes:
- Robustly handles weird dataset shapes like (F,1,1,1,C,H,W)
- Flattens any leading dims into frames => always (F,C,H,W)
- --cols selects frame indices; cylinder has only 5 frames => valid 0..4
- Optional --timesteps to auto-pick evenly spaced frames
- --no_mask disables any white cutouts

Example (Cylinder, 5 frames only):
python plot_strip_preds_pretty.py --config config_cylinder_vort300.json \
  --out cylinder_preds_physgto \
  --cols 0 1 2 3 4 \
  --col_labels +0 +7 +15 +23 +31 \
  --cmap rainbow_r --symmetric --no_mask
"""

import argparse, json, os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

from src.data import make_loaders
from src.runner import build_model, pick_device


# -------------------------- shape handling (CRITICAL) --------------------------

def squeeze_to_CHW(a: np.ndarray) -> np.ndarray:
    """
    Remove singleton dims except the last 3 dims which are assumed (C,H,W).
    Works for shapes like (1,1,1,C,H,W), (F,1,1,1,C,H,W), etc.
    """
    a = np.asarray(a)
    if a.ndim <= 3:
        return a
    axes = tuple(i for i in range(a.ndim - 3) if a.shape[i] == 1)
    return np.squeeze(a, axis=axes) if axes else a


def flatten_to_frames(a: np.ndarray) -> np.ndarray:
    """
    Force array into (F,C,H,W) where last 3 dims are (C,H,W).

    - (C,H,W) -> (1,C,H,W)
    - (F,C,H,W) -> unchanged
    - (...,C,H,W) -> flatten leading dims to F
    """
    a = squeeze_to_CHW(a)

    if a.ndim == 3:          # (C,H,W)
        return a[None, ...]  # (1,C,H,W)
    if a.ndim == 4:          # (F,C,H,W)
        return a
    if a.ndim >= 5:          # (...,C,H,W)
        lead = int(np.prod(a.shape[:-3]))
        return a.reshape(lead, a.shape[-3], a.shape[-2], a.shape[-1])
    raise ValueError(f"Cannot flatten shape {a.shape}")


def pick_field(frames_FCHW: np.ndarray, channel=0, use_mag=False) -> np.ndarray:
    """
    frames_FCHW: any shape that can be flattened to (F,C,H,W)
    returns: (F,H,W)
    """
    frames = flatten_to_frames(frames_FCHW)  # (F,C,H,W)
    if use_mag and frames.shape[1] >= 2:
        return np.sqrt(frames[:, 0] ** 2 + frames[:, 1] ** 2)
    return frames[:, channel]


# -------------------------- data collection --------------------------

def collect_gt_x(loader, norm):
    """
    Collect GT and X from loader, denormalize GT using norm.
    Returns:
      gts: (F,C,H,W)
      xs:  (F,Cin,H,W)
    """
    y_mean = torch.tensor(norm["y_mean"], dtype=torch.float32)
    y_std  = torch.tensor(norm["y_std"],  dtype=torch.float32)

    gts, xs = [], []
    for x, y, xyt in loader:
        ygt = (y * y_std + y_mean).cpu().numpy()
        xnp = x.cpu().numpy()

        gts.append(squeeze_to_CHW(ygt[0]))  # -> (C,H,W)
        xs.append(squeeze_to_CHW(xnp[0]))   # -> (Cin,H,W)

    gts = flatten_to_frames(np.stack(gts, 0))
    xs  = flatten_to_frames(np.stack(xs, 0))
    return gts, xs


def collect_preds(model, loader, norm, device):
    """
    Collect predictions from model, denormalize using norm.
    Returns preds: (F,C,H,W)
    """
    model.eval()
    y_mean = torch.tensor(norm["y_mean"], device=device, dtype=torch.float32)
    y_std  = torch.tensor(norm["y_std"],  device=device, dtype=torch.float32)

    preds = []
    with torch.no_grad():
        for x, y, xyt in loader:
            x, xyt = x.to(device), xyt.to(device)
            p = model(xyt, x).float()
            yhat = (p * y_std + y_mean).cpu().numpy()
            preds.append(squeeze_to_CHW(yhat[0]))  # -> (C,H,W)

    return flatten_to_frames(np.stack(preds, 0))


def load_models(cfg_path, models=("PINN", "PIBERT"), device_pref="auto"):
    cfg = json.load(open(cfg_path, "r"))
    device = pick_device(cfg.get("train", {}).get("device", device_pref))

    _, _, test_loader, norm, shapes = make_loaders(
        cfg["data"]["root"],
        fmt=cfg["data"].get("format", "npz"),
        batch_size=1,
        normalize=True,
    )

    # Collect GT/X once
    gts, xs = collect_gt_x(test_loader, norm)

    preds = {}
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

        # IMPORTANT: re-iterate loader (DataLoader is re-iterable)
        _, _, test_loader2, _, _ = make_loaders(
            cfg["data"]["root"],
            fmt=cfg["data"].get("format", "npz"),
            batch_size=1,
            normalize=True,
        )
        preds[name] = collect_preds(model, test_loader2, norm, device)

    return preds, gts, xs


# -------------------------- aesthetics helpers --------------------------

def add_arrow_band(fig, left, right, y, h, labels, title="Timestep"):
    axb = fig.add_axes([left, y, right-left, h])
    axb.set_axis_off()

    rect = Rectangle((0, 0), 0.96, 1.0, transform=axb.transAxes, facecolor="0.7", edgecolor="none")
    tri  = Polygon([[0.96, 0], [1.0, 0.5], [0.96, 1.0]],
                   transform=axb.transAxes, closed=True, facecolor="0.7", edgecolor="none")
    axb.add_patch(rect); axb.add_patch(tri)

    axb.text(0.03, 0.5, title, va="center", ha="left",
             fontsize=22, fontweight="bold", color="white", transform=axb.transAxes)

    # push labels right so "Timestep" never overlaps
    K = len(labels)
    x0, span = 0.38, 0.58
    for i, lab in enumerate(labels):
        x = x0 + span * (i + 0.5) / K
        axb.text(x, 0.5, lab, va="center", ha="center",
                 fontsize=22, fontweight="bold", color="white", transform=axb.transAxes)


def style_im_ax(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(1.2)
        sp.set_color("0.2")


# -------------------------- main --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_cylinder_vort300.json")
    ap.add_argument("--out", default="panel_preds_physgto")
    ap.add_argument("--device", default="auto")

    # columns: either explicit indices or auto selection
    ap.add_argument("--cols", type=int, nargs="+", default=None,
                    help="Frame indices to show (0-based). For cylinder you only have 5 frames => 0..4.")
    ap.add_argument("--timesteps", type=int, default=None,
                    help="If --cols is not provided, show this many evenly spaced frames.")
    ap.add_argument("--col_labels", type=str, nargs="+", default=None)

    # field options
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--use_mag", action="store_true")
    ap.add_argument("--cmap", default="coolwarm")
    ap.add_argument("--symmetric", action="store_true")
    ap.add_argument("--interp", default="bilinear", choices=["nearest", "bilinear"])

    # mask options
    ap.add_argument("--no_mask", action="store_true")
    ap.add_argument("--mask_from_x", action="store_true")
    ap.add_argument("--mask_channel", type=int, default=0)
    ap.add_argument("--mask_threshold", type=float, default=0.5)
    args = ap.parse_args()

    preds, gts, xs = load_models(args.config, device_pref=args.device)

    # flatten/standardize
    gts_FCHW = flatten_to_frames(gts)
    pinn_FCHW = flatten_to_frames(preds["PINN"])
    pib_FCHW  = flatten_to_frames(preds["PIBERT"])
    F = gts_FCHW.shape[0]
    print("frames =", F, "gts shape =", gts_FCHW.shape)

    # choose columns
    if args.cols is None:
        k = args.timesteps if args.timesteps is not None else min(5, F)
        idxs = np.linspace(0, F - 1, k, dtype=int).tolist()
    else:
        idxs = [int(i) for i in args.cols if 0 <= int(i) < F]
        bad = [int(i) for i in args.cols if not (0 <= int(i) < F)]
        if len(bad) > 0:
            print(f"[warn] dropped out-of-range cols {bad} (valid: 0..{F-1})")

    if len(idxs) == 0:
        raise ValueError(f"No valid frames selected. F={F}")

    K = len(idxs)

    # labels
    if args.col_labels is None:
        labels = [f"+{i}" for i in idxs]
    else:
        labels = list(args.col_labels)
        if len(labels) < K:
            labels += [f"+{i}" for i in idxs[len(labels):]]
            print(f"[warn] col_labels too short; padded to {K}")
        if len(labels) > K:
            labels = labels[:K]
            print(f"[warn] col_labels too long; truncated to {K}")

    # extract scalar field: (F,H,W)
    gt   = pick_field(gts_FCHW, channel=args.channel, use_mag=args.use_mag)
    pinn = pick_field(pinn_FCHW, channel=args.channel, use_mag=args.use_mag)
    pib  = pick_field(pib_FCHW,  channel=args.channel, use_mag=args.use_mag)

    # mask (optional)
    mask = None
    if not args.no_mask:
        if args.mask_from_x:
            xs_FCinHW = flatten_to_frames(xs)
            m = xs_FCinHW[:, args.mask_channel]  # (F,H,W)
            mask = (m < args.mask_threshold)
        else:
            if np.isnan(gt).any() or (~np.isfinite(gt)).any():
                mask = ~np.isfinite(gt)

    # shared color limits across GT + preds
    stack = np.concatenate([
        gt[idxs].reshape(K, -1),
        pinn[idxs].reshape(K, -1),
        pib[idxs].reshape(K, -1),
    ], axis=1).reshape(-1)
    stack = stack[np.isfinite(stack)]
    v1, v99 = np.percentile(stack, [1, 99])

    if args.symmetric:
        vmax = float(max(abs(v1), abs(v99)))
        vmin = -vmax
    else:
        vmin, vmax = float(v1), float(v99)

    cmap = plt.get_cmap(args.cmap).copy()
    cmap.set_bad(color="white")  # only relevant if mask is used

    # figure sizing similar to PhysGTO panel
    fig = plt.figure(figsize=(3.05 * K + 5.4, 6.6), dpi=300, facecolor="white")
    gs = fig.add_gridspec(
        3, K,
        left=0.14, right=0.885,
        bottom=0.08, top=0.86,
        hspace=0.10, wspace=0.14
    )

    # panel letter
    fig.text(0.02, 0.92, "d", fontsize=28, fontweight="bold", va="top")

    # arrow band
    add_arrow_band(fig, left=0.14, right=0.885, y=0.87, h=0.07, labels=labels, title="Timestep")

    # row labels (big, bold)
    fig.text(0.04, 0.64, "Ground\nTruth",     ha="left", va="center", fontsize=26, fontweight="bold")
    fig.text(0.04, 0.37, "PINN\nPrediction",  ha="left", va="center", fontsize=26, fontweight="bold")
    fig.text(0.04, 0.11, "PIBERT\nPrediction",ha="left", va="center", fontsize=26, fontweight="bold")

    def M(a, frame_idx):
        if mask is None:
            return a
        return np.ma.array(a, mask=mask[frame_idx])

    im_ref = None
    for c, idx in enumerate(idxs):
        ax = fig.add_subplot(gs[0, c])
        im_ref = ax.imshow(M(gt[idx], idx), cmap=cmap, origin="lower",
                           vmin=vmin, vmax=vmax, interpolation=args.interp)
        style_im_ax(ax)

        ax = fig.add_subplot(gs[1, c])
        ax.imshow(M(pinn[idx], idx), cmap=cmap, origin="lower",
                  vmin=vmin, vmax=vmax, interpolation=args.interp)
        style_im_ax(ax)

        ax = fig.add_subplot(gs[2, c])
        ax.imshow(M(pib[idx], idx), cmap=cmap, origin="lower",
                  vmin=vmin, vmax=vmax, interpolation=args.interp)
        style_im_ax(ax)

    # single shared colorbar (right)
    cax = fig.add_axes([0.905, 0.14, 0.018, 0.62])
    cb = fig.colorbar(im_ref, cax=cax)
    cb.ax.tick_params(labelsize=16, width=1.0, length=5)
    cb.set_label("Field", fontsize=22)

    fig.savefig(args.out + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(args.out + ".png", bbox_inches="tight", pad_inches=0.02, dpi=600)
    print(f"[saved] {args.out}.pdf and {args.out}.png")


if __name__ == "__main__":
    main()
