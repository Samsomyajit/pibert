# plot_compare_pub.py
"""
Publication figure: rows = models, columns = [GT | Pred | RelErr]
- Better colormaps
- Shared colorbars
- Bigger fonts
- Robust relative error with epsilon to avoid blow-ups
"""
import os, argparse, numpy as np, seaborn as sns, matplotlib.pyplot as plt

def load_trip(path, mode="mag"):
    d = np.load(path)
    gt, pr, re = d["gt"].astype(np.float32), d["pred"].astype(np.float32), d["relerr"].astype(np.float32)
    if mode == "mag":
        GT = np.sqrt(gt[0]**2 + gt[1]**2)
        PR = np.sqrt(pr[0]**2 + pr[1]**2)
        # recompute relative error robustly to ensure shared epsilon
        eps = 1e-3 * np.percentile(np.abs(GT), 95)
        RE = np.abs(PR-GT) / (np.abs(GT) + eps)
    elif mode in ("u","v"):
        idx = 0 if mode=="u" else 1
        GT = gt[idx]; PR = pr[idx]
        eps = 1e-3 * np.percentile(np.abs(GT), 95)
        RE = np.abs(PR-GT) / (np.abs(GT) + eps)
    else:
        raise ValueError("mode must be 'mag', 'u', or 'v'")
    return GT, PR, RE

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="model dir names under results/")
    ap.add_argument("--idx", type=int, default=0, help="sample index to visualize")
    ap.add_argument("--mode", default="mag", choices=["mag","u","v"])
    ap.add_argument("--results_root", default="results")
    ap.add_argument("--out", default="figure_compare.png")
    args = ap.parse_args()

    sns.set_context("talk")
    sns.set_style("white")
    plt.rcParams.update({"axes.titlesize":18, "axes.labelsize":16, "xtick.labelsize":12, "ytick.labelsize":12})

    trips = []
    for m in args.models:
        path = os.path.join(args.results_root, m, f"sample_{args.idx:03d}.npz")
        GT, PR, RE = load_trip(path, mode=args.mode)
        trips.append((m, GT, PR, RE))

    # field colormap (perceptually uniform), error colormap (sequential)
    cmap_field = "viridis"
    cmap_err = "rocket"

    # shared vmin/vmax
    vmin = min(GT.min() for _,GT,_,_ in trips)
    vmax = max(GT.max() for _,GT,_,_ in trips)

    n = len(trips)
    fig, axes = plt.subplots(n, 3, figsize=(3*4.5, n*3.8), constrained_layout=True)
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    ims_field, ims_err = [], []
    for i, (name, GT, PR, RE) in enumerate(trips):
        ax0, ax1, ax2 = axes[i]
        im0 = sns.heatmap(GT, ax=ax0, vmin=vmin, vmax=vmax, cmap=cmap_field, cbar=False, square=False)
        im1 = sns.heatmap(PR, ax=ax1, vmin=vmin, vmax=vmax, cmap=cmap_field, cbar=False, square=False)
        # Clamp extreme RE to 99th percentile for legibility
        rmax = np.percentile(RE, 99.5)
        im2 = sns.heatmap(np.clip(RE, 0, rmax), ax=ax2, cmap=cmap_err, cbar=False, square=False)
        ims_field.append(im0.collections[0]); ims_err.append(im2.collections[0])

        ax0.set_title(f"{name} — GT"); ax1.set_title("Pred"); ax2.set_title("RelErr")
        for ax in (ax0, ax1, ax2):
            ax.set_xlabel("x"); ax.set_ylabel("y")
            ax.tick_params(length=0)

    # shared colorbars
    cax1 = fig.add_axes([0.92, 0.55, 0.018, 0.35])
    cax2 = fig.add_axes([0.92, 0.12, 0.018, 0.35])
    fig.colorbar(ims_field[0], cax=cax1, label="Field")
    fig.colorbar(ims_err[0], cax=cax2, label="RelErr")

    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print("Saved", args.out)

if __name__ == "__main__":
    main()
