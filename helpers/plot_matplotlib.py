# plot_matplotlib.py
import numpy as np
import matplotlib.pyplot as plt
import argparse

def compute_uvmag(arr):
    if arr.shape[0] >= 2:
        return np.sqrt(arr[0]**2 + arr[1]**2)
    return arr[0]

def plot_grid(gt, pred, rel, field="uvmag", save=None, title=None):
    if field == "uvmag":
        GT = compute_uvmag(gt); PR = compute_uvmag(pred); RE = rel.mean(0)
    elif field == "u": GT,PR,RE = gt[0], pred[0], rel[0]
    elif field == "v": GT,PR,RE = gt[1], pred[1], rel[1]
    else: GT,PR,RE = gt[0], pred[0], rel.mean(0)

    fig, axes = plt.subplots(1, 3, figsize=(12,4), constrained_layout=True)
    vmin = min(GT.min(), PR.min()); vmax = max(GT.max(), PR.max())
    im0 = axes[0].imshow(GT, origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground Truth"); axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    im1 = axes[1].imshow(PR, origin="lower", vmin=vmin, vmax=vmax)
    axes[1].set_title("Prediction"); axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    im2 = axes[2].imshow(RE, origin="lower")
    axes[2].set_title("Relative Error"); axes[2].set_xlabel("x"); axes[2].set_ylabel("y")

    cbar0 = fig.colorbar(im0, ax=axes[:2], fraction=0.046, pad=0.04)
    cbar0.ax.set_title("Field")
    cbar1 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cbar1.ax.set_title("RelErr")

    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=10)

    if title: fig.suptitle(title, fontsize=14)
    if save: plt.savefig(save, dpi=300, bbox_inches="tight")
    else: plt.show()
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="sample_n.npz path with GT, Pred, RelErr arrays")
    ap.add_argument("--field", default="uvmag", choices=["uvmag","u","v","p"])
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    data = np.load(args.npz)
    gt = data["gt"]; pred = data["pred"]; rel = data["relerr"]
    plot_grid(gt, pred, rel, field=args.field, save=args.save, title=args.npz)

if __name__ == "__main__":
    main()
