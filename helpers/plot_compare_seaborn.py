# plot_compare_seaborn.py
import numpy as np, seaborn as sns, matplotlib.pyplot as plt, argparse, os

def load_trip(path):
    d = np.load(path)
    gt, pr, re = d["gt"], d["pred"], d["relerr"]
    GT = np.sqrt(gt[0]**2 + gt[1]**2); PR = np.sqrt(pr[0]**2 + pr[1]**2); RE = re.mean(0)
    return GT, PR, RE

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="model dir names under results/")
    ap.add_argument("--idx", type=int, default=0, help="sample index")
    ap.add_argument("--out", default="compare.png")
    args = ap.parse_args()

    trips = []
    for m in args.models:
        path = os.path.join("results", m, f"sample_{args.idx:03d}.npz")
        GT, PR, RE = load_trip(path)
        trips.append((m, GT, PR, RE))

    # shared vmin/vmax across all GT/PR
    vmin = min(t[1].min() for t in trips + [(None, trips[0][1], None, None)])
    vmax = max(t[2].max() for t in trips + [(None, None, trips[0][2], None)])

    n = len(trips)
    fig, axes = plt.subplots(n, 3, figsize=(3*3.6, n*3.2), constrained_layout=True)
    if n == 1: axes = np.expand_dims(axes,0)
    cmap_main, cmap_err = "viridis", "magma"

    # set up shared colorbars
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ims0, ims2 = [], []
    for i, (name, GT, PR, RE) in enumerate(trips):
        ax0, ax1, ax2 = axes[i]
        im0 = sns.heatmap(GT, ax=ax0, vmin=vmin, vmax=vmax, cmap=cmap_main, cbar=False)
        im1 = sns.heatmap(PR, ax=ax1, vmin=vmin, vmax=vmax, cmap=cmap_main, cbar=False)
        im2 = sns.heatmap(RE, ax=ax2, cmap=cmap_err, cbar=False)
        ims0.append(im0.collections[0]); ims2.append(im2.collections[0])
        ax0.set_title(f"{name} — GT"); ax1.set_title("Pred"); ax2.set_title("RelErr")
        for ax in (ax0, ax1, ax2): ax.tick_params(labelsize=11)

    # shared colorbars
    cax0 = fig.add_axes([0.92, 0.55, 0.015, 0.3]); cax2 = fig.add_axes([0.92, 0.15, 0.015, 0.3])
    fig.colorbar(ims0[0], cax=cax0, label="Field"); fig.colorbar(ims2[0], cax=cax2, label="RelErr")

    plt.savefig(args.out, dpi=300, bbox_inches="tight"); print("Saved", args.out)
