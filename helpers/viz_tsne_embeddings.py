# viz_tsne_embeddings.py
# Standalone: load checkpoints, collect output embeddings, run PCA + t-SNE, plot dense 2D grids.

import argparse, os, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.data import make_loaders
from src.runner import build_model, pick_device
from src.models import _ddx, _ddy, _grid_spacing


@torch.no_grad()
def physical_scalars(y_src, xyt):
    u, v = y_src[:, 0:1], y_src[:, 1:2]
    dx, dy = _grid_spacing(xyt)
    vort = _ddx(v, dx) - _ddy(u, dy)
    div = _ddx(u, dx) + _ddy(v, dy)
    speed = (u * u + v * v).sqrt()
    return {
        "speed": speed.flatten(1).cpu().numpy(),
        "vorticity": vort.flatten(1).cpu().numpy(),
        "divergence": div.flatten(1).cpu().numpy(),
    }


@torch.no_grad()
def collect_outputs(model, loader, device, unnorm, color_by="speed", max_slices=0, source="pred"):
    y_mean = torch.tensor(unnorm["y_mean"], device=device, dtype=torch.float32)
    y_std = torch.tensor(unnorm["y_std"], device=device, dtype=torch.float32)
    zs, cols = [], []
    taken = 0
    for x, y, xyt in loader:
        x, y, xyt = x.to(device), y.to(device), xyt.to(device)
        if source == "gt":
            y_src = y.float() * y_std + y_mean
        else:
            y_src = model(xyt, x).float() * y_std + y_mean
        z_np = y_src.cpu().numpy().reshape(y_src.size(0), -1)
        zs.append(z_np)
        scal = physical_scalars(y_src, xyt)[color_by]
        # pool scalar to match one row per sample
        scal_mean = scal.mean(axis=1, keepdims=True)
        cols.append(scal_mean)
        taken += 1
        if max_slices and taken >= max_slices:
            break
    Z = np.concatenate(zs, axis=0)
    C = np.concatenate(cols, axis=0)
    return Z, C


def make_tsne_grid(panels, color_by, out_png, cmap="rainbow"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    allc = np.concatenate([p["color"] for p in panels])
    vmin = np.percentile(allc, 1)
    vmax = np.percentile(allc, 99)
    for ax, p in zip(axes, panels):
        ax.scatter(p["Z"][:, 0], p["Z"][:, 1], c=p["color"],
                   s=6, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
        ax.set_title(f"{p['name']} (R^2={p['r2']:.3f})")
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.grid(False)
    for j in range(len(panels), 4):
        axes[j].set_visible(False)
    cax = fig.add_axes([0.1, 0.05, 0.8, 0.025])
    sm = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label(color_by)
    fig.tight_layout(rect=[0.02, 0.08, 0.98, 0.98])
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"[saved] {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="config JSON")
    ap.add_argument("--models", nargs="+", default=["PINN", "FNO2d", "DeepONet2d", "PIBERT"])
    ap.add_argument("--color_by", default="speed", choices=["speed", "vorticity", "divergence"])
    ap.add_argument("--color_source", default="pred", choices=["pred", "gt"])
    ap.add_argument("--slices", type=int, default=200, help="batches to sample (0=all)")
    ap.add_argument("--max_points", type=int, default=2000, help="subsample points per model")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--tsne_iter", type=int, default=500)
    ap.add_argument("--pca_dim", type=int, default=50, help="pre-PCA dim before t-SNE")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir_root", default=None, help="checkpoint root override")
    ap.add_argument("--out", required=True, help="output directory for figures/CSV")
    ap.add_argument("--ckpt_fallback", nargs="*", default=["results"])
    ap.add_argument("--device", default="cpu", help="cpu|cuda|mps (tsne runs on CPU regardless)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    out_root = args.outdir_root or cfg["eval"]["outdir"]
    os.makedirs(args.out, exist_ok=True)

    device = torch.device(args.device)
    train_loader, val_loader, test_loader, norm, shapes = make_loaders(
        cfg["data"]["root"], fmt=cfg["data"].get("format", "npz"),
        batch_size=1, normalize=True,
    )
    # use train+val+test if slices==0 to maximize points
    all_batches = list(train_loader) + list(val_loader) + list(test_loader)

    panels = []
    rows = []
    for mname in args.models[:4]:
        ckpt = None
        for root in [out_root] + [r for r in args.ckpt_fallback if r != out_root]:
            path = os.path.join(root, f"{mname}_seed{args.seed}", "last.pt")
            if os.path.exists(path):
                ckpt = path; break
        if ckpt is None:
            print(f"[skip] no checkpoint for {mname}")
            continue
        model = build_model(mname, shapes["Cin"], shapes["Cout"], cfg.get("model_cfg", {})).to(device)
        state = torch.load(ckpt, map_location=device)["model"]
        model.load_state_dict(state)
        print(f"[load] {ckpt}")
        model.eval()

        Z, C = collect_outputs(model, all_batches if args.slices == 0 else all_batches[:args.slices],
                               device, norm, color_by=args.color_by, max_slices=args.slices,
                               source=args.color_source)
        # subsample points
        if args.max_points and Z.shape[0] > args.max_points:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(Z.shape[0], size=args.max_points, replace=False)
            Z = Z[idx]; C = C[idx]
        # pre-PCA
        d = min(args.pca_dim, Z.shape[1])
        Zp = PCA(n_components=d, random_state=args.seed).fit_transform(Z)
        # t-SNE (safe fallback to PCA)
        try:
            ts = TSNE(
                n_components=2,
                perplexity=min(args.perplexity, max(5, Zp.shape[0] // 3)),
                max_iter=args.tsne_iter,
                learning_rate="auto",
                init="pca",
                random_state=args.seed,
                method="barnes_hut",
                n_jobs=1,
            ).fit_transform(Zp.astype(np.float64, copy=False))
            coords = ts
        except Exception as e:
            print(f"[warn] TSNE failed for {mname} ({e}); using 2D PCA fallback.")
            coords = PCA(n_components=2, random_state=args.seed).fit_transform(Zp)
        C1d = C.reshape(-1)
        # simple EPA R2 via linear probe on 2D coords
        if coords.shape[0] >= 4 and np.std(C1d) > 1e-8:
            from sklearn.linear_model import Ridge
            reg = Ridge(alpha=1e-3)
            reg.fit(coords, C1d)
            pred = reg.predict(coords)
            ss_res = ((C1d - pred) ** 2).sum()
            ss_tot = ((C1d - C1d.mean()) ** 2).sum() + 1e-12
            r2 = 1.0 - ss_res / ss_tot
        else:
            r2 = float("nan")
        panels.append({"name": mname, "Z": coords, "color": C1d, "r2": r2})
        rows.append({"Model": mname, f"EPA_R2_{args.color_by}": r2, "path": ckpt})

    if panels:
        make_tsne_grid(panels, args.color_by, os.path.join(args.out, f"tsne_grid_{args.color_by}_{args.color_source}.png"))
    if rows:
        import pandas as pd
        pd.DataFrame(rows).to_csv(os.path.join(args.out, f"embedding_metrics_{args.color_by}_{args.color_source}.csv"), index=False)


if __name__ == "__main__":
    main()
