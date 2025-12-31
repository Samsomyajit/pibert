# plot_seaborn_example.py (optional; for local use)
import numpy as np, seaborn as sns, matplotlib.pyplot as plt, argparse

def compute_uvmag(arr):
    if arr.shape[0] >= 2:
        return np.sqrt(arr[0]**2 + arr[1]**2)
    return arr[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--field", default="uvmag", choices=["uvmag","u","v","p"])
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    data = np.load(args.npz)
    gt = data["gt"]; pred = data["pred"]; rel = data["relerr"]

    if args.field == "uvmag":
        GT = compute_uvmag(gt); PR = compute_uvmag(pred); RE = rel.mean(0)
    elif args.field == "u": GT,PR,RE = gt[0], pred[0], rel[0]
    elif args.field == "v": GT,PR,RE = gt[1], pred[1], rel[1]
    else: GT,PR,RE = gt[0], pred[0], rel.mean(0)

    fig, axes = plt.subplots(1,3,figsize=(12,4))
    vmin = min(GT.min(), PR.min()); vmax = max(GT.max(), PR.max())
    cmap = "viridis"
    sns.heatmap(GT, ax=axes[0], vmin=vmin, vmax=vmax, cmap=cmap, cbar_kws={"label": "Field"})
    axes[0].set_title("Ground Truth")
    sns.heatmap(PR, ax=axes[1], vmin=vmin, vmax=vmax, cmap=cmap, cbar=False)
    axes[1].set_title("Prediction")
    sns.heatmap(RE, ax=axes[2], cmap="magma", cbar_kws={"label": "RelErr"})
    axes[2].set_title("Relative Error")
    for ax in axes: ax.tick_params(labelsize=10)
    if args.save: plt.savefig(args.save, dpi=300, bbox_inches="tight")
    else: plt.show()

if __name__ == "__main__":
    main()
# plot_seaborn_example.py placeholder — please re-download the earlier pack if missing.
