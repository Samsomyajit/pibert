# plot_pressure_jcp_style.py  (Field cbar: left-bottom | Error cbar: right-bottom + LEFT model names)
import os, argparse, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# -------------------- Metrics --------------------
def robust_relerr(gt, pr, percentile=95, eps_scale=1e-3):
    """
    Relative error with robust stabilization:
      |pr-gt| / (|gt| + eps),  eps = eps_scale * percentile(|gt|, p)
    """
    scale = np.percentile(np.abs(gt), percentile)
    eps = max(eps_scale * scale, 1e-8)
    return np.abs(pr - gt) / (np.abs(gt) + eps)

# -------------------- IO -------------------------
def load_trip_pressure(path):
    """
    Expects npz with arrays:
      - gt   : (1, H, W)
      - pred : (1, H, W)
    Returns (GT, PR, RE) where RE is relative error.
    """
    d = np.load(path)
    gt = d["gt"].astype(np.float32)    # (1,H,W)
    pr = d["pred"].astype(np.float32)  # (1,H,W)
    GT = gt[0]
    PR = pr[0]
    RE = robust_relerr(GT, PR)
    return GT, PR, RE

# -------------------- Style helpers -------------------------
def set_ax_clean(ax):
    sns.despine(ax=ax, top=True, right=True)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(width=1.0, length=4)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))

# -------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="List of model directory names under --results_root (optionally with --labels).")
    ap.add_argument("--idx", type=int, default=0,
                    help="Which sample_{idx:03d}.npz to load.")
    ap.add_argument("--results_root", default="results_dam_p",
                    help="Root where runner saved <model>[/_seedXX]/sample_*.npz")
    ap.add_argument("--out", default="figure_pressure.png")
    ap.add_argument("--cmap_field", default="coolwarm")
    ap.add_argument("--cmap_err", default="magma")
    ap.add_argument("--symmetric", type=str, default="false",
                    help="True/False: symmetric color limits for the field (pressure). Default False.")

    # error visualization
    ap.add_argument("--err-style", dest="err_style",
                    choices=["heat","contour","log","mask"], default="heat")
    ap.add_argument("--err-levels", type=int, default=7)
    ap.add_argument("--err-min", type=float, default=None)
    ap.add_argument("--err-max", type=float, default=None)
    ap.add_argument("--mask-thresh", type=float, default=5.0,
                    help="percent threshold for --err-style mask")

    # bottom colorbar sizing/placement (figure fractions)
    ap.add_argument("--cbar-frac", type=float, default=0.36,
                    help="width of EACH bottom colorbar as fraction of figure width")
    ap.add_argument("--cbar-height", type=float, default=0.013,
                    help="height of each bottom colorbar (figure fraction)")
    ap.add_argument("--cbar-bottom", type=float, default=0.010,
                    help="bottom coord for bottom colorbars (0..1)")
    ap.add_argument("--cbar-padx", type=float, default=0.08,
                    help="inner horizontal padding from left/right figure edges")

    # space & style for model labels on the far left
    ap.add_argument("--left-margin", type=float, default=0.12,
                    help="figure left margin to make room for model names (0..1)")
    ap.add_argument("--model-label-offset", type=float, default=0.018,
                    help="how far LEFT of the GT panel the name is placed (figure fraction)")
    ap.add_argument("--model-label-size", type=float, default=16)
    ap.add_argument("--model-label-weight", default="bold")

    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--seed", type=str, default=None,
                    help="If set, load from <model>_seed<seed>. Use an int like 42.")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Optional labels to display (one per model).")
    # --- add these args (near the other argparse options) ---
    ap.add_argument("--col-gap", type=float, default=0.30,
                    help="Horizontal whitespace between columns (matplotlib wspace).")
    ap.add_argument("--row-gap", type=float, default=0.24,
                    help="Vertical whitespace between rows (matplotlib hspace).")

    args = ap.parse_args()

    sns.set_theme(style="white", context="talk")

    symmetric = args.symmetric.lower() == "true"

    # Gather trips
    trips = []
    dirnames = []
    for m in args.models:
        dname = f"{m}_seed{args.seed}" if args.seed is not None else m
        dirnames.append(dname)
        p = os.path.join(args.results_root, dname, f"sample_{args.idx:03d}.npz")
        if not os.path.exists(p):
            raise SystemExit(f"Missing {p}. Run runner.py first to create it.")
        GT, PR, RE = load_trip_pressure(p)
        trips.append((m, GT, PR, RE))

    # Field limits (pressure)
    if symmetric:
        vmax = max(np.max(np.abs(GT)) for _, GT, _, _ in trips)
        vmin = -vmax
    else:
        vmin = min(GT.min() for _, GT, _, _ in trips)
        vmax = max(GT.max() for _, GT, _, _ in trips)

    # Relative error limits
    all_re = np.concatenate([RE.ravel() for *_, RE in trips])
    rmin = 0.0 if args.err_min is None else args.err_min
    rmax = np.percentile(all_re, 99.5) if args.err_max is None else args.err_max
    rmax = max(rmax, rmin + 1e-6)

    # Figure/grid
    n = len(trips)
    fig_w, fig_h = 16.5, 3.6 * n + 1.6
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=args.dpi)
    gs = fig.add_gridspec(nrows=n, ncols=3, hspace=0.20, wspace=0.20)

    # column headers
    for j, txt in enumerate(["Ground Truth p", "Prediction p", "Relative Error"]):
        fig.text((j + 0.5)/3, 0.985, txt, ha="center", va="top",
                 fontsize=22, weight="bold")

    ims_field, ims_err = [], []
    row_ax0 = []  # keep GT axes to position labels later

    for i, (name, GT, PR, RE) in enumerate(trips):
        ax0 = fig.add_subplot(gs[i, 0]); ax1 = fig.add_subplot(gs[i, 1]); ax2 = fig.add_subplot(gs[i, 2])
        row_ax0.append((ax0, name))

        im0 = ax0.imshow(GT, origin="lower", cmap=args.cmap_field,
                         vmin=vmin, vmax=vmax, interpolation="bilinear")
        im1 = ax1.imshow(PR, origin="lower", cmap=args.cmap_field,
                         vmin=vmin, vmax=vmax, interpolation="bilinear")

        # error styles
        if args.err_style == "heat":
            im2 = ax2.imshow(np.clip(RE, rmin, rmax), origin="lower",
                             cmap=args.cmap_err, vmin=rmin, vmax=rmax, interpolation="bilinear")
        elif args.err_style == "contour":
            levels = np.geomspace(max(1e-6, rmax/400), rmax, args.err_levels)
            im2 = ax2.contourf(RE, levels=levels, cmap=args.cmap_err)
            ax2.contour(RE, levels=levels, colors="k", linewidths=0.6, alpha=0.7)
        elif args.err_style == "log":
            logv = np.log10(np.clip(RE, 1e-6, None))
            v2min, v2max = np.log10(max(1e-6, rmin)), np.log10(max(1e-6, rmax))
            im2 = ax2.imshow(logv, origin="lower", cmap=args.cmap_err,
                             vmin=v2min, vmax=v2max, interpolation="bilinear")
        else:  # mask
            thr = args.mask_thresh/100.0
            mask = (RE >= thr).astype(float)
            im2 = ax2.imshow(mask, origin="lower", cmap="gray_r", vmin=0, vmax=1,
                             interpolation="nearest")

        for ax in (ax0, ax1, ax2):
            ax.set_xlabel("x", fontsize=14)
            ax.set_ylabel("y", fontsize=14)
            set_ax_clean(ax)

        ims_field.append(im0); ims_err.append(im2)

    # Make room at the bottom + left for labels and colorbars
    need_bottom = args.cbar_bottom + args.cbar_height + 0.012
    plt.subplots_adjust(left=args.left_margin, bottom=max(need_bottom, 0.12), top=0.95)

    # display names
    if args.labels and len(args.labels) == len(trips):
        display_names = args.labels
    else:
        display_names = [name.split("_seed")[0] for name, *_ in trips]

    # ---- Put model names on the FAR LEFT (outside axes) ----
    for i, (ax0, name) in enumerate(row_ax0):
        pos = ax0.get_position()
        x = max(0.005, pos.x0 - args.model_label_offset)
        y = pos.y0 + pos.height / 2.0
        plt.gcf().text(x, y, display_names[i], ha="right", va="center",
                       fontsize=args.model_label_size, weight=args.model_label_weight)

    # --- Bottom colorbars ---
    # left bar (Field)
    left_field = args.cbar_padx
    width_field = args.cbar_frac
    cax_field = fig.add_axes([left_field, args.cbar_bottom, width_field, args.cbar_height])

    # right bar (Error)
    width_err = args.cbar_frac
    left_err = 1.0 - args.cbar_padx - width_err
    cax_err = fig.add_axes([left_err, args.cbar_bottom, width_err, args.cbar_height])

    cb0 = fig.colorbar(ims_field[0], cax=cax_field, orientation="horizontal")
    cb0.set_label("Pressure p", fontsize=14)
    cb0.ax.tick_params(labelsize=12, length=3, width=0.8)

    err_label = {"log": "Relative Error (log10)",
                 "mask": f"Error ≥ {args.mask_thresh:.1f}% (0/1)",
                 "contour": "Relative Error",
                 "heat": "Relative Error"}[args.err_style]
    cb2 = fig.colorbar(ims_err[0], cax=cax_err, orientation="horizontal")
    cb2.set_label(err_label, fontsize=14)
    cb2.ax.tick_params(labelsize=12, length=3, width=0.8)

    fig.savefig(args.out, bbox_inches="tight")
    print("Saved", args.out)

if __name__ == "__main__":
    main()
