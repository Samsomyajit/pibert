# viz_logged_embeddings.py
# Visualize embeddings saved during training (npz per epoch/model) with PCA and EPA R^2.

import argparse, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge


def zscore(Z):
    m = Z.mean(axis=0, keepdims=True)
    s = Z.std(axis=0, keepdims=True) + 1e-8
    return (Z - m) / s


def epa_r2_split(Z, y, test_frac=0.2, seed=0, alpha=1e-3):
    if Z.shape[0] < 4 or np.std(y) < 1e-8:
        return float("nan")
    n = Z.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    te = idx[: max(1, int(test_frac * n))]
    tr = idx[max(1, int(test_frac * n)):]
    Zm, Zs = Z[tr].mean(axis=0), Z[tr].std(axis=0) + 1e-8
    ym, ys = y[tr].mean(), y[tr].std() + 1e-8
    Zt = (Z - Zm) / Zs
    yt = (y - ym) / ys
    reg = Ridge(alpha=alpha)
    reg.fit(Zt[tr], yt[tr])
    pred = reg.predict(Zt[te])
    ss_res = ((yt[te] - pred) ** 2).sum()
    ss_tot = ((yt[te] - yt[te].mean()) ** 2).sum() + 1e-12
    return 1.0 - ss_res / ss_tot


def load_embeds(root, model, seed, epoch=None):
    pat = os.path.join(root, f"{model}_seed{seed}", "embeds_epoch*.npz")
    files = sorted(glob.glob(pat))
    if not files:
        return None
    if epoch is None:
        path = files[-1]
    else:
        path = os.path.join(root, f"{model}_seed{seed}", f"embeds_epoch{epoch:04d}.npz")
        if not os.path.exists(path):
            return None
    data = np.load(path)
    Z = data["Z"]
    phys = {k: data[k] for k in ["speed", "vorticity", "divergence"] if k in data}
    ep = int(data["epoch"]) if "epoch" in data else None
    return Z, phys, ep, path


def make_pca_grid(panels, color_by, out_path, cmap="rainbow"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    vmin = min([np.percentile(p["color"], 1) for p in panels])
    vmax = max([np.percentile(p["color"], 99) for p in panels])
    for ax, p in zip(axes, panels):
        pts = ax.scatter(p["Z"][:, 0], p["Z"][:, 1], c=p["color"],
                         s=4, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
        ax.set_title(f"{p['name']} (epoch {p['epoch']})")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.subplots_adjust(bottom=0.1, right=0.85)
    cax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(pts, cax=cax, label=color_by)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="results root (e.g., results_tube_quick_refit)")
    ap.add_argument("--models", nargs="+", default=["PINN", "FNO2d", "DeepONet2d", "PIBERT"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epoch", type=int, default=None, help="epoch to load (default: latest)")
    ap.add_argument("--color_by", default="vorticity", choices=["speed", "vorticity", "divergence"])
    ap.add_argument("--pca_dim", type=int, default=8)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--ridge_alpha", type=float, default=1e-3)
    ap.add_argument("--out", required=True, help="output directory for figures/CSVs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    panels = []
    rows = []
    for m in args.models[:4]:
        loaded = load_embeds(args.root, m, args.seed, args.epoch)
        if loaded is None:
            print(f"[skip] no embeds for {m}")
            continue
        Z, phys, ep, path = loaded
        Zt = zscore(Z)
        d = min(max(2, args.pca_dim), Zt.shape[1], max(2, Zt.shape[0] - 1))
        if d < 2:
            print(f"[skip] {m} has insufficient samples (Z shape {Zt.shape})")
            continue
        Zp = PCA(n_components=d, random_state=args.seed).fit_transform(Zt)
        color = phys[args.color_by]
        panels.append({"name": m, "Z": Zp[:, :2], "color": color, "epoch": ep if ep else "?"})

        r2 = epa_r2_split(Zp, color, test_frac=args.test_frac, seed=args.seed, alpha=args.ridge_alpha)
        rows.append({"Model": m, f"EPA_R2_{args.color_by}": r2, "epoch": ep, "path": path})
        print(f"[{m}] EPA_R2_{args.color_by}={r2:.3f} (epoch {ep}) from {path}")

    if panels:
        make_pca_grid(panels, args.color_by, os.path.join(args.out, f"pca_grid_{args.color_by}.png"))
    if rows:
        import pandas as pd
        pd.DataFrame(rows).to_csv(os.path.join(args.out, f"embedding_metrics_{args.color_by}.csv"), index=False)


if __name__ == "__main__":
    main()
