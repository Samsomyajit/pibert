#!/usr/bin/env python3
"""
PhysGTO-style panel d with a *smooth clipped boundary* (spline-like),
using a geometry mask -> contour -> Chaikin smoothing -> clip_path.

Rows (mode=errors):
  1) Ground Truth
  2) PINN Abs Error
  3) PIBERT Abs Error

Rows (mode=preds):
  1) Ground Truth
  2) PINN Prediction
  3) PIBERT Prediction

Example (EAGLE):
python plot_panel_d_physgto_clip.py \
  --config config_eagle_spline_quick.json \
  --out eagle_panel_d_like \
  --cols 1 100 150 200 250 \
  --col_labels +1 +100 +150 +200 +250 \
  --mode errors \
  --mask_from_x --mask_channel 0 --mask_threshold 0.5 \
  --cmap_field coolwarm --cmap_err magma \
  --clip_smooth 3 --dpi 800
"""

import argparse, json, os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.path import Path

from src.data import make_loaders
from src.runner import build_model, pick_device


# ---------------------- shape utilities ----------------------

def squeeze_keep_last3(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim <= 3:
        return a
    axes = tuple(i for i in range(a.ndim - 3) if a.shape[i] == 1)
    return np.squeeze(a, axis=axes) if axes else a

def to_FCHW(a: np.ndarray) -> np.ndarray:
    """Force any array to (F,C,H,W) by flattening leading dims and squeezing singleton dims."""
    a = squeeze_keep_last3(a)
    if a.ndim == 3:       # (C,H,W)
        return a[None, ...]
    if a.ndim == 4:       # (F,C,H,W)
        return a
    if a.ndim >= 5:       # (...,C,H,W)
        lead = int(np.prod(a.shape[:-3]))
        return a.reshape(lead, a.shape[-3], a.shape[-2], a.shape[-1])
    raise ValueError(f"Unexpected shape {a.shape}")

def pick_field_FHW(frames_FCHW: np.ndarray, channel=0, use_mag=False) -> np.ndarray:
    frames = to_FCHW(frames_FCHW)
    if use_mag and frames.shape[1] >= 2:
        return np.sqrt(frames[:, 0]**2 + frames[:, 1]**2)
    return frames[:, channel]


# ---------------------- model IO ----------------------

def collect_gt_x(loader, norm):
    y_mean = torch.tensor(norm["y_mean"], dtype=torch.float32)
    y_std  = torch.tensor(norm["y_std"],  dtype=torch.float32)

    gts, xs = [], []
    for x, y, xyt in loader:
        ygt = (y * y_std + y_mean).cpu().numpy()
        gts.append(squeeze_keep_last3(ygt[0]))  # (C,H,W) or weird -> fixed

        xnp = x.cpu().numpy()
        xs.append(squeeze_keep_last3(xnp[0]))   # (Cin,H,W)
    return to_FCHW(np.stack(gts, 0)), to_FCHW(np.stack(xs, 0))

def collect_preds(model, loader, norm, device):
    model.eval()
    y_mean = torch.tensor(norm["y_mean"], device=device, dtype=torch.float32)
    y_std  = torch.tensor(norm["y_std"],  device=device, dtype=torch.float32)

    preds = []
    with torch.no_grad():
        for x, y, xyt in loader:
            x, xyt = x.to(device), xyt.to(device)
            p = model(xyt, x).float()
            yhat = (p * y_std + y_mean).cpu().numpy()
            preds.append(squeeze_keep_last3(yhat[0]))
    return to_FCHW(np.stack(preds, 0))

def load_models(cfg_path, models=("PINN", "PIBERT"), device_pref="auto"):
    cfg = json.load(open(cfg_path, "r"))
    device = pick_device(cfg.get("train", {}).get("device", device_pref))

    _, _, test_loader, norm, shapes = make_loaders(
        cfg["data"]["root"],
        fmt=cfg["data"].get("format", "npz"),
        batch_size=1,
        normalize=True,
    )

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

        # re-create loader to re-iterate cleanly
        _, _, test_loader2, norm2, _ = make_loaders(
            cfg["data"]["root"],
            fmt=cfg["data"].get("format", "npz"),
            batch_size=1,
            normalize=True,
        )
        preds[name] = collect_preds(model, test_loader2, norm2, device)

    return preds, gts, xs


# ---------------------- clip path (smooth boundary) ----------------------

def chaikin(p, n_iter=3, closed=True):
    p = np.asarray(p, dtype=float)

    # guard: empty or too short
    if p.ndim != 2 or p.shape[0] < 3:
        return p

    # ensure closed loop if requested
    if closed:
        if not np.allclose(p[0], p[-1]):
            p = np.vstack([p, p[0]])

    for _ in range(n_iter):
        if p.shape[0] < 3:
            break
        q = 0.75 * p[:-1] + 0.25 * p[1:]
        r = 0.25 * p[:-1] + 0.75 * p[1:]
        p = np.vstack([np.column_stack([q[:, 0], q[:, 1]]),
                       np.column_stack([r[:, 0], r[:, 1]])]).reshape(-1, 2)

        if closed:
            # close again
            if not np.allclose(p[0], p[-1]):
                p = np.vstack([p, p[0]])

    return p


def mask_to_clip_vertices(valid_mask: np.ndarray, smooth_iters=3):
    # If mask is trivial, there is no boundary to clip
    if valid_mask.all() or (~valid_mask).all():
        return None

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.set_axis_off()

    cs = ax.contour(valid_mask.astype(float), levels=[0.5], origin="lower")

    segs = []
    if hasattr(cs, "allsegs") and len(cs.allsegs) and len(cs.allsegs[0]):
        segs = cs.allsegs[0]

    plt.close(fig)

    # keep only non-empty segments
    segs = [s for s in segs if s is not None and np.asarray(s).shape[0] >= 3]
    if not segs:
        return None

    verts = max(segs, key=lambda s: s.shape[0])

    if smooth_iters and smooth_iters > 0:
        verts = chaikin(verts, n_iter=smooth_iters, closed=True)

    # final guard
    if verts is None or np.asarray(verts).shape[0] < 3:
        return None

    return verts




# ---------------------- aesthetics ----------------------

def add_arrow_band(fig, left, right, y, h, labels, title="Timestep"):
    axb = fig.add_axes([left, y, right-left, h])
    axb.set_axis_off()

    rect = Rectangle((0, 0), 0.96, 1.0, transform=axb.transAxes, facecolor="0.70", edgecolor="none")
    tri  = Polygon([[0.96, 0], [1.0, 0.5], [0.96, 1.0]],
                   transform=axb.transAxes, closed=True, facecolor="0.70", edgecolor="none")
    axb.add_patch(rect); axb.add_patch(tri)

    axb.text(0.03, 0.5, title, va="center", ha="left",
             fontsize=18, fontweight="bold", color="white", transform=axb.transAxes)

    K = len(labels)
    x0, span = 0.32, 0.64
    for i, lab in enumerate(labels):
        x = x0 + span * (i + 0.5) / K
        axb.text(x, 0.5, lab, va="center", ha="center",
                 fontsize=18, fontweight="bold", color="white", transform=axb.transAxes)

def style_ax(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


# ---------------------- main ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_eagle_spline_quick.json")
    ap.add_argument("--out", default="panel_d_like")
    ap.add_argument("--device", default="auto")

    ap.add_argument("--cols", type=int, nargs="+", default=[1, 100, 150, 200, 250])
    ap.add_argument("--col_labels", type=str, nargs="+", default=None)

    ap.add_argument("--mode", choices=["errors", "preds"], default="errors")

    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--use_mag", action="store_true")

    ap.add_argument("--cmap_field", default="coolwarm")
    ap.add_argument("--cmap_err", default="magma")
    ap.add_argument("--symmetric_field", action="store_true")
    ap.add_argument("--interp", default="bicubic", choices=["nearest", "bilinear", "bicubic"])

    ap.add_argument("--mask_from_x", action="store_true")
    ap.add_argument("--mask_channel", type=int, default=0)
    ap.add_argument("--mask_threshold", type=float, default=0.5)
    ap.add_argument("--no_mask", action="store_true")

    ap.add_argument("--clip_smooth", type=int, default=3, help="Chaikin smoothing iterations for boundary.")
    ap.add_argument("--dpi", type=int, default=800)
    args = ap.parse_args()

    preds, gts, xs = load_models(args.config, device_pref=args.device)

    gts = to_FCHW(gts)
    xs  = to_FCHW(xs)
    pinn = to_FCHW(preds["PINN"])
    pib  = to_FCHW(preds["PIBERT"])

    F = gts.shape[0]
    idxs = [i for i in args.cols if 0 <= i < F]
    if len(idxs) == 0:
        raise ValueError(f"All --cols out of range. F={F}. Requested={args.cols}")

    K = len(idxs)
    labels = args.col_labels if args.col_labels is not None else [f"+{i}" for i in idxs]
    if len(labels) != K:
        labels = (labels + [f"+{i}" for i in idxs])[:K]
        print("[warn] col_labels length mismatch; auto-fixed.")

    print(f"[info] total frames F={F}, requested cols={args.cols}, using idxs={idxs}")


    # fields (F,H,W)
    gtF = pick_field_FHW(gts, channel=args.channel, use_mag=args.use_mag)
    p1F = pick_field_FHW(pinn, channel=args.channel, use_mag=args.use_mag)
    p2F = pick_field_FHW(pib,  channel=args.channel, use_mag=args.use_mag)

    # mask -> clip path (use first frame, usually constant over time)
    clip_verts = None
    if (not args.no_mask) and args.mask_from_x:
        m = xs[0, args.mask_channel]  # (H,W) from first frame
        valid = (m >= args.mask_threshold)
        clip_verts = mask_to_clip_vertices(valid, smooth_iters=args.clip_smooth)

    # color limits
    def pct(a, lo=1, hi=99):
        v = a[np.isfinite(a)]
        return np.percentile(v, [lo, hi])

    f_lo, f_hi = pct(np.concatenate([gtF[idxs], p1F[idxs], p2F[idxs]], axis=0))
    if args.symmetric_field:
        vmax = float(max(abs(f_lo), abs(f_hi))); vmin = -vmax
    else:
        vmin, vmax = float(f_lo), float(f_hi)

    if args.mode == "errors":
        e1 = np.abs(p1F - gtF)
        e2 = np.abs(p2F - gtF)
        emax = float(np.percentile(np.concatenate([e1[idxs], e2[idxs]]).ravel(), 99))
        emin = 0.0

    # figure layout close to the paper
    fig = plt.figure(figsize=(3.0*K + 2.6, 4.8), dpi=300, facecolor="white")
    gs = fig.add_gridspec(
        3, K,
        left=0.11, right=0.90,
        bottom=0.10, top=0.82,
        hspace=0.10, wspace=0.04
    )

    fig.text(0.03, 0.87, "d", fontsize=22, fontweight="bold", va="top")
    add_arrow_band(fig, left=0.11, right=0.90, y=0.83, h=0.07, labels=labels, title="Timestep")

    # row labels
    fig.text(0.05, 0.66, "Ground\nTruth", ha="left", va="center", fontsize=12, fontweight="bold")
    if args.mode == "errors":
        fig.text(0.05, 0.41, "PINN\nAbs Error", ha="left", va="center", fontsize=12, fontweight="bold")
        fig.text(0.05, 0.16, "PIBERT\nAbs Error", ha="left", va="center", fontsize=12, fontweight="bold")
    else:
        fig.text(0.05, 0.41, "PINN\nPrediction", ha="left", va="center", fontsize=12, fontweight="bold")
        fig.text(0.05, 0.16, "PIBERT\nPrediction", ha="left", va="center", fontsize=12, fontweight="bold")

    im_field_ref = None
    im_err_ref = None

    for c, idx in enumerate(idxs):
        # GT
        ax = fig.add_subplot(gs[0, c])
        im = ax.imshow(gtF[idx], cmap=args.cmap_field, origin="lower",
                       vmin=vmin, vmax=vmax, interpolation=args.interp)
        style_ax(ax)
        if clip_verts is not None:
            im.set_clip_path(Path(clip_verts), ax.transData)
            print("[warn] Could not extract a boundary from mask (mask may be inverted or threshold wrong). "
          "Falling back to normal imshow (no clipping).")
        im_field_ref = im

        # row2
        ax = fig.add_subplot(gs[1, c])
        if args.mode == "errors":
            im = ax.imshow(e1[idx], cmap=args.cmap_err, origin="lower",
                           vmin=emin, vmax=emax, interpolation=args.interp)
            im_err_ref = im
        else:
            im = ax.imshow(p1F[idx], cmap=args.cmap_field, origin="lower",
                           vmin=vmin, vmax=vmax, interpolation=args.interp)
        style_ax(ax)
        if clip_verts is not None:
            im.set_clip_path(Path(clip_verts), ax.transData)

        # row3
        ax = fig.add_subplot(gs[2, c])
        if args.mode == "errors":
            im = ax.imshow(e2[idx], cmap=args.cmap_err, origin="lower",
                           vmin=emin, vmax=emax, interpolation=args.interp)
        else:
            im = ax.imshow(p2F[idx], cmap=args.cmap_field, origin="lower",
                           vmin=vmin, vmax=vmax, interpolation=args.interp)
        style_ax(ax)
        if clip_verts is not None:
            im.set_clip_path(Path(clip_verts), ax.transData)

    # colorbars (match paper: field bar near top row, error bar near lower rows)
    cax_field = fig.add_axes([0.905, 0.55, 0.010, 0.25])
    cb1 = fig.colorbar(im_field_ref, cax=cax_field)
    cb1.ax.tick_params(labelsize=7, width=0.8, length=3)

    if args.mode == "errors":
        cax_err = fig.add_axes([0.905, 0.17, 0.010, 0.25])
        cb2 = fig.colorbar(im_err_ref, cax=cax_err)
        cb2.ax.tick_params(labelsize=7, width=0.8, length=3)

    fig.savefig(args.out + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(args.out + ".png", bbox_inches="tight", pad_inches=0.02, dpi=args.dpi)
    print(f"[saved] {args.out}.pdf and {args.out}.png")


if __name__ == "__main__":
    
    main()
