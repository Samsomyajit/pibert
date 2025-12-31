#!/usr/bin/env python3
"""
Token similarity heatmaps (cosine) for PINN/PIBERT (and other supported models)
using the embedder hooks in runner._get_embedder. Plots a PhysGTO-style matrix:
tokens x tokens with cosine similarity, centered at 0.

Example:
  python plot_token_similarity.py --config config_cylinder.json --models PINN PIBERT --idx 0 --max_tokens 64 --out token_sim_cyl.png
"""
import argparse, json, os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from src.data import make_loaders
from src.runner import build_model, pick_device, _get_embedder


def _cosine_matrix(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.float32)
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    return z @ z.T


def _get_tokens(model, loader, device, max_tokens=None):
    embed = _get_embedder(model)
    if embed is None:
        return None
    for x, y, xyt in loader:
        x, xyt = x.to(device), xyt.to(device)
        with torch.no_grad():
            tok = embed(xyt, x)  # (B, N, D)
        tok = tok[0].cpu().numpy()
        if max_tokens and tok.shape[0] > max_tokens:
            tok = tok[:max_tokens]
        return tok
    return None


def plot_matrix(ax, mat, title, xticks=True, yticks=True, vmax=1.0):
    im = ax.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax, origin="upper", interpolation="none")
    ax.set_title(title, fontsize=12, fontweight="normal")
    if not xticks:
        ax.set_xticks([])
    if not yticks:
        ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(1.2)
    return im


def main():
    ap = argparse.ArgumentParser(description="Token similarity heatmaps via embedder hooks.")
    ap.add_argument("--config", required=True, help="Config JSON for the dataset/model shapes.")
    ap.add_argument("--models", nargs="+", default=["PINN", "PIBERT"])
    ap.add_argument("--idx", type=int, default=0, help="Sample index from test set.")
    ap.add_argument("--max_tokens", type=int, default=64, help="Optional token cap for readability.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="token_similarity.png")
    ap.add_argument("--center", action="store_true", help="Mean-center tokens before cosine.")
    ap.add_argument("--zscore", action="store_true", help="Z-score tokens per feature before cosine (implies centering).")
    args = ap.parse_args()

    cfg = json.load(open(args.config, "r"))
    device = pick_device(cfg.get("train", {}).get("device", args.device))

    # Load data once; then reseed the loader to jump to idx
    _, _, test_loader, _, shapes = make_loaders(
        cfg["data"]["root"],
        fmt=cfg["data"].get("format", "npz"),
        batch_size=1,
        normalize=True,
    )

    # grab idx-th sample
    test_list = list(test_loader)
    if args.idx < 0 or args.idx >= len(test_list):
        raise SystemExit(f"idx {args.idx} out of range for test set of size {len(test_list)}")

    x_one, y_one, xyt_one = test_list[args.idx]

    rows = []
    for name in args.models:
        model = build_model(name, shapes["Cin"], shapes["Cout"], cfg.get("model_cfg", {})).to(device)
        ckpt_dir = os.path.join(cfg["eval"]["outdir"], f"{name}_seed{cfg['train'].get('seed', 42)}")
        ckpt = os.path.join(ckpt_dir, "last.pt")
        if not os.path.exists(ckpt):
            print(f"[skip] missing checkpoint {ckpt}")
            continue
        raw_state = torch.load(ckpt, map_location=device)["model"]
        current = model.state_dict()
        state = {k: v for k, v in raw_state.items() if k in current and current[k].shape == v.shape}
        model.load_state_dict(state, strict=False)

        # use a tiny loader with the selected sample to avoid reusing the whole list
        from torch.utils.data import DataLoader, TensorDataset
        ds = TensorDataset(x_one, y_one, xyt_one)
        dl = DataLoader(ds, batch_size=1, shuffle=False)

        tok = _get_tokens(model, dl, device, max_tokens=args.max_tokens)
        if tok is None:
            print(f"[skip] embedder not available for {name}")
            continue
        if args.zscore:
            tok = tok - tok.mean(axis=0, keepdims=True)
            tok = tok / (tok.std(axis=0, keepdims=True) + 1e-6)
        elif args.center:
            tok = tok - tok.mean(axis=0, keepdims=True)
        sim = _cosine_matrix(tok)
        rows.append((name, sim))

    if not rows:
        raise SystemExit("No models produced embeddings; nothing to plot.")

    sns.set_theme(style="white")
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"]})

    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), dpi=300)
    if n == 1:
        axes = [axes]

    vmax = max(np.abs(sim).max() for _, sim in rows)
    vmax = max(vmax, 1.0)

    ims = []
    for ax, (name, sim) in zip(axes, rows):
        ims.append(plot_matrix(ax, sim, title=name, xticks=True, yticks=True, vmax=vmax))

    # Single colorbar outside the right edge; widen the figure slightly and disable tight_layout
    fig.subplots_adjust(right=0.88, top=0.92, bottom=0.08, left=0.08, wspace=0.25)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    cbar = fig.colorbar(ims[0], cax=cax)
    cbar.set_label("Cosine Similarity", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    plt.savefig(args.out, bbox_inches="tight")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
