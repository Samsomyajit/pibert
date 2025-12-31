#!/usr/bin/env python3
"""
Trade-off plots (L2 error vs FLOPs) for Cylinder, Cavity, and Tube datasets.
Matches the style of the provided reference: dual y-axes, solid line for L2, dashed for FLOPs.

To use:
  1) Edit the `DATA` dict below with your measured values.
  2) Run: python plot_tradeoff_cylinder_cavity_tube.py
  3) Figure is saved to tradeoff_cyl_cav_tube.png
"""
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Fill these with your numbers.
# x_labels: tick labels on the x axis
# x_vals  : numeric positions (same length as x_labels)
# err     : L2 errors (same length)
# flops   : FLOPs in G (same length)
# ---------------------------------------------------------------------------
DATA = {
    "mesh_sampling": {
        "xlabel": "Mesh Sampling Rate",
        "x_labels": ["1x", "2x", "4x", "10x", "20x", "100x"],
        "x_vals":   [1, 2, 4, 10, 20, 100],
        # Replace the example values below with your measurements
        "Cylinder": {"err": [0.012, 0.014, 0.015, 0.017, 0.018, 0.019],
                     "flops": [35, 25, 20, 15, 12, 10]},
        "Cavity":   {"err": [0.010, 0.012, 0.014, 0.016, 0.017, 0.019],
                     "flops": [40, 30, 22, 16, 13, 11]},
        "Tube":     {"err": [0.011, 0.013, 0.015, 0.016, 0.018, 0.020],
                     "flops": [32, 24, 18, 14, 11, 9]},
    },
    "latent_dim": {
        "xlabel": "Latent Dimension",
        "x_labels": ["32", "64", "128", "256"],
        "x_vals":   [32, 64, 128, 256],
        # Replace the example values below with your measurements
        "Cylinder": {"err": [0.0155, 0.0100, 0.0085, 0.0062],
                     "flops": [12, 18, 30, 75]},
        "Cavity":   {"err": [0.0130, 0.0110, 0.0090, 0.0070],
                     "flops": [14, 20, 33, 80]},
        "Tube":     {"err": [0.0145, 0.0120, 0.0100, 0.0080],
                     "flops": [16, 22, 36, 85]},
    },
}

COLORS = {"err": "#7b68ee", "flops": "#5cb85c"}  # purple, green


def plot_panel(ax, x_vals, x_labels, err, flops, title, xlabel, ylim_err=None, ylim_flops=None):
    ax2 = ax.twinx()

    ax.plot(x_vals, err, marker="o", color=COLORS["err"], lw=2.5, label="L2 Error")
    ax2.plot(x_vals, flops, marker="s", color=COLORS["flops"], lw=2.5, ls="--", label="FLOPs")

    ax.set_xticks(x_vals)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.tick_params(axis="y", labelsize=10, width=1.4, length=5, color="0.2")
    ax2.tick_params(axis="y", labelsize=10, width=1.4, length=5, color="0.2")
    for sp in ax.spines.values(): sp.set_linewidth(1.8)
    for sp in ax2.spines.values(): sp.set_linewidth(1.8)

    ax.grid(True, ls="--", lw=0.7, alpha=0.5, color="0.5")
    ax.set_ylabel(r"$L_2$ Errors", fontsize=12, fontweight="bold")
    ax2.set_ylabel("FLOPs (G)", fontsize=12, fontweight="bold", color="0.25")
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", y=1.02)

    if ylim_err: ax.set_ylim(*ylim_err)
    if ylim_flops: ax2.set_ylim(*ylim_flops)

    # Single legend combining both axes
    lines = ax.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=10, frameon=False, loc="upper right")


def main():
    sns.set_theme(style="white")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
    })

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=300)

    datasets = ["Cylinder", "Cavity", "Tube"]
    # Top row: mesh sampling
    ms = DATA["mesh_sampling"]
    for j, ds in enumerate(datasets):
        plot_panel(axes[0, j], ms["x_vals"], ms["x_labels"],
                   ms[ds]["err"], ms[ds]["flops"],
                   title=ds, xlabel=ms["xlabel"])

    # Bottom row: latent dim
    ld = DATA["latent_dim"]
    for j, ds in enumerate(datasets):
        plot_panel(axes[1, j], ld["x_vals"], ld["x_labels"],
                   ld[ds]["err"], ld[ds]["flops"],
                   title=ds, xlabel=ld["xlabel"])

    fig.tight_layout()
    out = "tradeoff_cyl_cav_tube.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"[saved] {out}. Edit DATA in the script to use your own numbers.")


if __name__ == "__main__":
    main()
