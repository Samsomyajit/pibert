#!/usr/bin/env python3
"""
Grid of token-similarity heatmaps (cosine) for multiple datasets (rows) and models (columns).
Adds panel labels (a, b, c, d, ...).

Example:
python plot_token_similarity_grid.py \
  --configs config_cylinder_vort300.json config_plasma_quick.json config_eagle_spline_quick.json \
  --models PINN PIBERT \
  --row_labels Cylinder Plasma Eagle \
  --idx 0 --max_tokens 64 --zscore --out token_sim_grid.png
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


def compute_sim(cfg_path, model_name, idx, device, max_tokens, center, zscore):
    cfg = json.load(open(cfg_path, "r"))
    device = pick_device(cfg.get("train", {}).get("device", device))
    _, _, test_loader, _, shapes = make_loaders(
        cfg["data"]["root"],
        fmt=cfg["data"].get("format", "npz"),
        batch_size=1,
        normalize=True,
    )
    test_list = list(test_loader)
    if idx < 0 or idx >= len(test_list):
        raise SystemExit(f"idx {idx} out of range for test set of size {len(test_list)} (config {cfg_path})")
    x_one, y_one, xyt_one = test_list[idx]

    model = build_model(model_name, shapes["Cin"], shapes["Cout"], cfg.get("model_cfg", {})).to(device)
    seed = cfg["train"].get("seed", 42)
    ckpt_dir = os.path.join(cfg["eval"]["outdir"], f"{model_name}_seed{seed}")
    ckpt = os.path.join(ckpt_dir, "last.pt")
    if not os.path.exists(ckpt):
        print(f"[skip] missing checkpoint {ckpt}")
        return None
    raw_state = torch.load(ckpt, map_location=device)["model"]
    current = model.state_dict()
    state = {k: v for k, v in raw_state.items() if k in current and current[k].shape == v.shape}
    model.load_state_dict(state, strict=False)

    from torch.utils.data import DataLoader, TensorDataset
    dl = DataLoader(TensorDataset(x_one, y_one, xyt_one), batch_size=1, shuffle=False)
    tok = _get_tokens(model, dl, device, max_tokens=max_tokens)
    if tok is None:
        print(f"[skip] embedder not available for {model_name} ({cfg_path})")
        return None
    if zscore:
        tok = tok - tok.mean(axis=0, keepdims=True)
        tok = tok / (tok.std(axis=0, keepdims=True) + 1e-6)
    elif center:
        tok = tok - tok.mean(axis=0, keepdims=True)
    sim = _cosine_matrix(tok)
    return sim


def main():
    ap = argparse.ArgumentParser(description="Grid of token similarity heatmaps with panel labels.")
    ap.add_argument("--configs", nargs="+", required=True, help="List of config JSONs (one row per dataset).")
    ap.add_argument("--models", nargs="+", default=["PINN", "PIBERT"], help="Model names per column.")
    ap.add_argument("--row_labels", nargs="+", default=None, help="Optional labels per row; defaults to config basenames.")
    ap.add_argument("--idx", type=int, default=0, help="Test sample index to visualize.")
    ap.add_argument("--max_tokens", type=int, default=64, help="Token cap for readability.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="token_sim_grid.png")
    ap.add_argument("--center", action="store_true", help="Mean-center tokens before cosine.")
    ap.add_argument("--zscore", action="store_true", help="Z-score tokens per feature before cosine (implies centering).")
    args = ap.parse_args()

    n_rows = len(args.configs)
    n_cols = len(args.models)
    row_labels = args.row_labels if args.row_labels and len(args.row_labels) == n_rows else [
        os.path.splitext(os.path.basename(c))[0].replace("config_", "").replace("_quick", "").replace("_vort300", "").replace("_vort", "").replace("_spline", "").replace("_npz", "").strip("_") for c in args.configs
    ]

    sims = [[None for _ in range(n_cols)] for _ in range(n_rows)]
    vmax = 1.0
    for i, cfg_path in enumerate(args.configs):
        for j, m in enumerate(args.models):
            sim = compute_sim(cfg_path, m, args.idx, args.device, args.max_tokens, args.center, args.zscore)
            sims[i][j] = sim
            if sim is not None:
                vmax = max(vmax, np.abs(sim).max())

    sns.set_theme(style="white")
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"]})

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols + 1, 4.2 * n_rows), dpi=300)
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    ims = []
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]
            sim = sims[i][j]
            if sim is None:
                ax.axis("off")
                continue
            im = ax.imshow(sim, cmap="coolwarm", vmin=-vmax, vmax=vmax, origin="upper", interpolation="none")
            ax.set_xticks([]); ax.set_yticks([])
            title = args.models[j]
            ax.set_title(title, fontsize=12, fontweight="normal")
            # row label on the leftmost column
            if j == 0:
                ax.set_ylabel(row_labels[i], fontsize=12, fontweight="bold", rotation=90, labelpad=12)
            ims.append(im)

    # panel labels (a, b, c, ...) placed just outside top-left of each axis
    letters = "abcdefghijklmnopqrstuvwxyz"
    for idx, ax in enumerate(axes.flat):
        ax.text(-0.07, 1.02,  "(" + letters[idx] + ")", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=11, fontweight="normal")

    # separate colorbar for each axis, placed on its right
    fig.subplots_adjust(top=0.96, bottom=0.06, left=0.12, right=0.92, wspace=0.35, hspace=0.35)
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]
            sim = sims[i][j]
            if sim is None:
                continue
            bbox = ax.get_position()
            height = bbox.height * 0.6  # shorten cbar to 60% of subplot height
            cax = fig.add_axes([bbox.x1 + 0.01, bbox.y0 + (bbox.height - height) / 2, 0.02, height])
            cbar = fig.colorbar(ax.images[0], cax=cax)
            cbar.set_label("Cosine Similarity", fontsize=11)
            cbar.ax.tick_params(labelsize=9)

    plt.savefig(args.out, bbox_inches="tight")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
