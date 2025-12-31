#!/usr/bin/env python3
import argparse, json, os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

from src.data import make_loaders
from src.runner import build_model, pick_device


def pick_field(arr, channel=0, use_mag=False):
    # arr: (..., C, H, W) or (..., H, W)
    if arr.ndim >= 3 and arr.shape[-3] >= 1:  # has C
        if use_mag and arr.shape[-3] >= 2:
            a0 = arr[..., 0, :, :]
            a1 = arr[..., 1, :, :]
            return np.sqrt(a0*a0 + a1*a1)
        return arr[..., channel, :, :]
    return arr


def smart_stack(lst):
    # lst elements could be (C,H,W) OR (T,C,H,W)
    a0 = lst[0]
    if a0.ndim == 3:
        return np.stack(lst, 0)          # (N,C,H,W)
    if a0.ndim == 4:
        return np.stack(lst, 0)          # (N,T,C,H,W)
    raise ValueError("Unexpected element shape in list:", a0.shape)


def collect_preds(model, loader, norm, device):
    model.eval()
    y_mean = torch.tensor(norm["y_mean"], device=device, dtype=torch.float32)
    y_std  = torch.tensor(norm["y_std"],  device=device, dtype=torch.float32)

    preds, gts = [], []
    with torch.no_grad():
        for x, y, xyt in loader:
            x, y, xyt = x.to(device), y.to(device), xyt.to(device)
            p = model(xyt, x).float()
            yhat = (p * y_std + y_mean).cpu().numpy()
            ygt  = (y * y_std + y_mean).cpu().numpy()

            preds.append(yhat[0])  # could be (C,H,W) or (T,C,H,W)
            gts.append(ygt[0])
    return smart_stack(preds), smart_stack(gts)


def load_models(cfg_path, models=("PINN", "PIBERT"), device_pref="auto"):
    cfg = json.load(open(cfg_path, "r"))
    device = pick_device(cfg.get("train", {}).get("device", device_pref))

    _, _, test_loader, norm, shapes = make_loaders(
        cfg["data"]["root"],
        fmt=cfg["data"].get("format", "npz"),
        batch_size=1,
        normalize=True,
    )

    preds, gts = {}, None
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

        p, gt_arr = collect_preds(model, test_loader, norm, device)
        preds[name] = p
        if gts is None:
            gts = gt_arr

    return preds, gts


def add_arrow_band(fig, left, right, y, h, labels, title="Timestep"):
    axb = fig.add_axes([left, y, right-left, h])
    axb.set_axis_off()

    rect = Rectangle((0, 0), 0.96, 1.0, transform=axb.transAxes, facecolor="0.7", edgecolor="none")
    tri  = Polygon([[0.96, 0], [1.0, 0.5], [0.96, 1.0]],
                   transform=axb.transAxes, closed=True, facecolor="0.7", edgecolor="none")
    axb.add_patch(rect); axb.add_patch(tri)

    axb.text(0.02, 0.5, title, va="center", ha="left",
             fontsize=18, fontweight="bold", color="white", transform=axb.transAxes)

    # start labels further right so "Timestep" never overlaps
    K = len(labels)
    x0, span = 0.34, 0.60
    for i, lab in enumerate(labels):
        x = x0 + span * (i + 0.5) / K
        axb.text(x, 0.5, lab, va="center", ha="center",
                 fontsize=18, fontweight="bold", color="white", transform=axb.transAxes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_cylinder_vort300.json")
    ap.add_argument("--out", default="cylinder_panel_d_preds")
    ap.add_argument("--device", default="auto")

    ap.add_argument("--cols", type=int, nargs="+", default=[0, 7, 15, 23, 31, 39])
    ap.add_argument("--col_labels", type=str, nargs="+", default=None)

    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--use_mag", action="store_true")
    ap.add_argument("--cmap", default="coolwarm")
    ap.add_argument("--symmetric", action="store_true")
    ap.add_argument("--interp", default="bilinear", choices=["nearest", "bilinear"])
    args = ap.parse_args()

    preds, gts = load_models(args.config, device_pref=args.device)
    print("gts shape =", gts.shape)  # <--- important

    # Decide whether time is sample-axis or inside-sample
    # Case A: gts = (N,C,H,W)  -> frames = N
    # Case B: gts = (N,T,C,H,W)-> frames = T (usually N==1)
    if gts.ndim == 4:
        # frames live on axis 0
        frames_gt   = gts                      # (N,C,H,W)
        frames_pinn = preds["PINN"]
        frames_pib  = preds["PIBERT"]
        total = frames_gt.shape[0]
        idxs = [i for i in args.cols if 0 <= i < total]
    elif gts.ndim == 5:
        # pick first sample, frames live on axis 0 of T
        frames_gt   = gts[0]                   # (T,C,H,W)
        frames_pinn = preds["PINN"][0]
        frames_pib  = preds["PIBERT"][0]
        total = frames_gt.shape[0]
        idxs = [i for i in args.cols if 0 <= i < total]
    else:
        raise ValueError("Unexpected gts ndim:", gts.ndim)

    if len(idxs) == 0:
        raise ValueError(f"No valid cols. total frames={total}, requested={args.cols}")

    # IMPORTANT: force labels length to match plotted columns
    labels = args.col_labels if args.col_labels is not None else [f"+{i}" for i in idxs]
    labels = labels[:len(idxs)]
    if args.col_labels is not None and len(args.col_labels) != len(idxs):
        print(f"[warn] You provided {len(args.col_labels)} labels but only {len(idxs)} valid frames. "
              f"Using first {len(idxs)} labels.")

    # Extract the plotted scalar field
    gt   = pick_field(frames_gt,   channel=args.channel, use_mag=args.use_mag)   # (F,H,W)
    pinn = pick_field(frames_pinn, channel=args.channel, use_mag=args.use_mag)
    pib  = pick_field(frames_pib,  channel=args.channel, use_mag=args.use_mag)

    # Shared vmin/vmax across GT+preds
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

    K = len(idxs)
    fig = plt.figure(figsize=(2.9*K + 5.2, 6.4), dpi=300, facecolor="white")
    gs = fig.add_gridspec(3, K, left=0.14, right=0.88, bottom=0.08, top=0.86,
                          hspace=0.08, wspace=0.04)

    fig.text(0.02, 0.92, "d", fontsize=24, fontweight="bold", va="top")
    add_arrow_band(fig, left=0.14, right=0.88, y=0.87, h=0.07, labels=labels, title="Timestep")

    # Row labels
    fig.text(0.04, 0.64, "Ground\nTruth", ha="left", va="center", fontsize=22, fontweight="bold")
    fig.text(0.04, 0.37, "PINN\nPrediction", ha="left", va="center", fontsize=22, fontweight="bold")
    fig.text(0.04, 0.11, "PIBERT\nPrediction", ha="left", va="center", fontsize=22, fontweight="bold")

    im_ref = None
    for c, idx in enumerate(idxs):
        ax = fig.add_subplot(gs[0, c])
        im_ref = ax.imshow(gt[idx], cmap=args.cmap, origin="lower",
                           vmin=vmin, vmax=vmax, interpolation=args.interp)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(1.2); sp.set_color("0.2")

        ax = fig.add_subplot(gs[1, c])
        ax.imshow(pinn[idx], cmap=args.cmap, origin="lower",
                  vmin=vmin, vmax=vmax, interpolation=args.interp)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(1.2); sp.set_color("0.2")

        ax = fig.add_subplot(gs[2, c])
        ax.imshow(pib[idx], cmap=args.cmap, origin="lower",
                  vmin=vmin, vmax=vmax, interpolation=args.interp)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(1.2); sp.set_color("0.2")

    # Colorbar
    cax = fig.add_axes([0.90, 0.14, 0.018, 0.62])
    cb = fig.colorbar(im_ref, cax=cax)
    cb.ax.tick_params(labelsize=14, width=1.0)
    cb.set_label("Field", fontsize=18)

    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    fig.savefig(args.out + ".png", bbox_inches="tight", dpi=600)
    print(f"[saved] {args.out}.pdf and {args.out}.png")


if __name__ == "__main__":
    main()
