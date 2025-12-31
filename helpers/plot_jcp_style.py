# plot_jcp_style.py  (Field cbar: left-bottom | Error cbar: right-bottom + LEFT model names)
import os, argparse, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

def robust_relerr(gt, pr, percentile=95, eps_scale=1e-3):
    scale = np.percentile(np.abs(gt), percentile)
    eps = max(eps_scale * scale, 1e-8)
    return np.abs(pr - gt) / (np.abs(gt) + eps)

def load_trip(path, mode="mag"):
    d = np.load(path)
    gt, pr = d["gt"].astype(np.float32), d["pred"].astype(np.float32)
    if mode == "mag":
        GT = np.sqrt(gt[0]**2 + gt[1]**2)
        PR = np.sqrt(pr[0]**2 + pr[1]**2)
    elif mode == "u":
        GT, PR = gt[0], pr[0]
    elif mode == "v":
        GT, PR = gt[1], pr[1]
    else:
        raise ValueError("mode must be 'mag','u','v'")
    RE = robust_relerr(GT, PR)
    return GT, PR, RE

def set_ax_clean(ax):
    sns.despine(ax=ax, top=True, right=True)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(width=1.0, length=4)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--mode", choices=["mag","u","v"], default="mag")
    ap.add_argument("--results_root", default="results")
    ap.add_argument("--out", default="figure_jcp.png")
    ap.add_argument("--cmap_field", default="RdBu_r")
    ap.add_argument("--cmap_err", default="magma")
    ap.add_argument("--symmetric", type=str, default="auto",
                    help="True/False/auto; if auto: symmetric for u,v; not for mag")

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

    # NEW: space & style for model labels on the far left
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

    args = ap.parse_args()

    sns.set_theme(style="white", context="talk")

    symmetric = (args.mode in ("u","v")) if args.symmetric.lower() == "auto" \
               else (args.symmetric.lower() == "true")

    trips = []
    dirnames = []
    for m in args.models:
        dname = f"{m}_seed{args.seed}" if args.seed is not None else m
        dirnames.append(dname)
        p = os.path.join(args.results_root, dname, f"sample_{args.idx:03d}.npz")
        if not os.path.exists(p):
            raise SystemExit(f"Missing {p}. Run runner.py first to create it.")
        GT, PR, RE = load_trip(p, args.mode)
        trips.append((m, GT, PR, RE))


    # field limits
    if symmetric:
        vmax = max(np.max(np.abs(GT)) for _, GT, _, _ in trips); vmin = -vmax
    else:
        vmin = min(GT.min() for _, GT, _, _ in trips)
        vmax = max(GT.max() for _, GT, _, _ in trips)

    # relative error limits
    all_re = np.concatenate([RE.ravel() for *_, RE in trips])
    rmin = 0.0 if args.err_min is None else args.err_min
    rmax = (np.percentile(all_re, 99.5) if args.err_max is None else args.err_max)

    # figure/grid
    n = len(trips)
    fig_w, fig_h = 13.2, 3.6 * n + 1.6
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=args.dpi)
    gs = fig.add_gridspec(nrows=n, ncols=3, hspace=0.20, wspace=0.20)

    # column headers
    for j, txt in enumerate(["Ground Truth", "Prediction", "Relative Error"]):
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
            levels = np.geomspace(max(1e-3, rmax/400), rmax, args.err_levels)
            im2 = ax2.contourf(RE, levels=levels, cmap=args.cmap_err)
            ax2.contour(RE, levels=levels, colors="k", linewidths=0.6, alpha=0.7)
        elif args.err_style == "log":
            logv = np.log10(np.clip(RE, 1e-3, None))
            v2min, v2max = np.log10(max(1e-3, rmin)), np.log10(max(1e-3, rmax))
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

    if args.labels and len(args.labels) == len(trips):
        display_names = args.labels
    else:
        # default: strip a trailing “_seedNN” if present
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
    left_field = args.cbar_padx           # was args.cbar_padax
    width_field = args.cbar_frac
    cax_field = fig.add_axes([left_field, args.cbar_bottom, width_field, args.cbar_height])

    # right bar (Error)
    width_err = args.cbar_frac
    left_err = 1.0 - args.cbar_padx - width_err   # was args.cbar_padax
    cax_err = fig.add_axes([left_err, args.cbar_bottom, width_err, args.cbar_height])


    cb0 = fig.colorbar(ims_field[0], cax=cax_field, orientation="horizontal")
    cb0.set_label("Field", fontsize=14)
    cb0.ax.tick_params(labelsize=12, length=3, width=0.8)

    err_label = {"log": "Relative Error (log10)",
                 "mask": f"Error ≥ {args.mask_thresh:.1f}% (0/1)",
                 "contour": "Relative Error (%)",
                 "heat": "Relative Error (%)"}[args.err_style]
    cb2 = fig.colorbar(ims_err[0], cax=cax_err, orientation="horizontal")
    cb2.set_label(err_label, fontsize=14)
    cb2.ax.tick_params(labelsize=12, length=3, width=0.8)

    fig.savefig(args.out, bbox_inches="tight")
    print("Saved", args.out)

if __name__ == "__main__":
    main()
