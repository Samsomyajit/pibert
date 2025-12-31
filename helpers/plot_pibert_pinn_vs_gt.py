#!/usr/bin/env python3
import os, glob, argparse, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

sns.set_theme(style="white", context="talk")

# ------------------------- style -------------------------
def use_jcp_style(serif=True, dpi=300):
    plt.rcParams.update({
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0, "ytick.major.width": 1.0,
        "xtick.major.size": 6, "ytick.major.size": 6,
        "xtick.minor.size": 4, "ytick.minor.size": 4,
        "legend.frameon": False, "figure.dpi": dpi,
    })
    if serif:
        plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})

def set_ax_clean(ax):
    for s in ("top","right","left","bottom"):
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(width=1.0, length=4)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))

# ------------------------- math helpers -------------------------
def robust_relerr(gt, pr, percentile=95, eps_scale=1e-3):
    scale = np.percentile(np.abs(gt), percentile)
    eps = max(eps_scale * scale, 1e-8)
    return np.abs(pr - gt) / (np.abs(gt) + eps)

def vector_mag(arr2):
    return np.sqrt(arr2[0]**2 + arr2[1]**2)

def extract_mode(d, mode):
    gt = d["gt"].astype(np.float32)
    pr = d["pred"].astype(np.float32)
    if mode == "mag":
        GT = vector_mag(gt); PR = vector_mag(pr)
    elif mode == "u":
        GT = gt[0]; PR = pr[0]
    elif mode == "v":
        GT = gt[1]; PR = pr[1]
    else:
        # scalar / pressure
        GT = gt[0]; PR = pr[0]
    if "relerr" in d and d["relerr"].ndim in (2,3):
        RE_saved = d["relerr"].astype(np.float32)
        RE = RE_saved[0] if RE_saved.ndim == 3 else RE_saved
    else:
        RE = robust_relerr(GT, PR)
    return GT, PR, RE

# ------------------------- IO -------------------------
def find_sample(results_root, model, idx, seed=None):
    """Return (path, seed_str) to sample_{idx:03d}.npz for a given model."""
    name = f"sample_{idx:03d}.npz"

    # explicit seed
    if seed is not None:
        seed_str = str(int(seed)) if str(seed).isdigit() else str(seed)
        p = os.path.join(results_root, f"{model}_seed{seed_str}", name)
        if os.path.exists(p):
            return p, seed_str

    # auto-detect a seed dir
    globbed = sorted(glob.glob(os.path.join(results_root, f"{model}_seed*", name)))
    if globbed:
        p = globbed[0]
        parent = os.path.basename(os.path.dirname(p))
        seed_str = parent.split("_seed")[-1] if "_seed" in parent else None
        return p, seed_str

    # fallback old layout
    fallback = os.path.join(results_root, model, name)
    if os.path.exists(fallback):
        return fallback, None

    return None, None

# ------------------------- plotting -------------------------
def main():
    ap = argparse.ArgumentParser(description="PINN/PIBERT vs CFD (JCP-style)")
    ap.add_argument("--results_root", default="results_sup")
    ap.add_argument("--models", nargs="+", required=True, help="e.g., PINN PIBERT")
    ap.add_argument("--idx", type=int, default=0, help="sample index saved by runner")
    ap.add_argument("--mode", choices=["mag","u","v","p"], default="mag")
    ap.add_argument("--second", choices=["both","pred","error"], default="pred",
                    help="columns: both=GT|Pred|Error, pred=GT|Pred, error=GT|Error")
    ap.add_argument("--seed", default=None, help="seed to load from (applies to all models)")
    ap.add_argument("--show-seed-label", action="store_true",
                    help="include '(seed N)' in the left row labels (off by default)")

    # colormaps & styles
    ap.add_argument("--cmap_field", default="RdBu_r")
    ap.add_argument("--cmap_err", default="magma")
    ap.add_argument("--err-style", choices=["heat","log","contour","mask"], default="log")
    ap.add_argument("--err-levels", type=int, default=7)
    ap.add_argument("--err-min", type=float, default=None)
    ap.add_argument("--err-max", type=float, default=None)
    ap.add_argument("--mask-thresh", type=float, default=5.0)

    # layout / labels
    ap.add_argument("--left-margin", type=float, default=0.16, help="space for model labels")
    ap.add_argument("--model-label-offset", type=float, default=0.055)
    ap.add_argument("--model-label-size", type=float, default=16)
    ap.add_argument("--model-label-weight", default="bold")

    # bottom colorbars
    ap.add_argument("--cbar-frac", type=float, default=0.34)
    ap.add_argument("--cbar-height", type=float, default=0.013)
    ap.add_argument("--cbar-bottom", type=float, default=0.010)
    ap.add_argument("--cbar-padx", type=float, default=0.08)

    # general
    ap.add_argument("--no-serif", action="store_true")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out", default="fig_pibert_pinn_jcp.png")
    args = ap.parse_args()

    use_jcp_style(serif=not args.no_serif, dpi=args.dpi)

    # which columns
    if args.second == "both":
        cols = ["CFD Ground Truth", "Prediction", "Relative Error"]
    elif args.second == "pred":
        cols = ["CFD Ground Truth", "Prediction"]
    else:
        cols = ["CFD Ground Truth", "Relative Error"]

    # load rows
    rows = []  # (row_label, GT, PR, RE)
    for m in args.models:
        p, used_seed = find_sample(args.results_root, m, args.idx, seed=args.seed)
        if p is None:
            print(f"[WARN] missing sample for {m} idx={args.idx}")
            continue
        d = np.load(p)
        GT, PR, RE = extract_mode(d, args.mode)
        if args.show_seed_label and used_seed is not None:
            label = f"{m} (seed {used_seed})"
        else:
            label = m
        rows.append((label, GT, PR, RE))

    if not rows:
        raise SystemExit("No samples found. Make sure runner saved sample_*.npz under results_root.")

    # field limits
    symmetric = (args.mode in ("u","v"))
    if symmetric:
        vmax = max(np.max(np.abs(GT)) for _, GT, _, _ in rows)
        vmin = -vmax
    else:
        vmin = min(GT.min() for _, GT, _, _ in rows)
        vmax = max(GT.max() for _, GT, _, _ in rows)

    # error limits
    all_re = np.concatenate([RE.ravel() for *_, RE in rows])
    rmin = 0.0 if args.err_min is None else args.err_min
    rmax = (np.percentile(all_re, 99.5) if args.err_max is None else args.err_max)
    rmax = max(rmax, 1e-3)

    # figure
    n = len(rows)
    ncols = len(cols)
    base_w = 4.0
    fig_w = max(9.5, base_w * ncols + 2.0)
    fig_h = 3.7 * n + 1.4
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=args.dpi)
    gs = fig.add_gridspec(nrows=n, ncols=ncols, hspace=0.22, wspace=0.22)

    # column headers
    for j, txt in enumerate(cols):
        fig.text((j + 0.5)/ncols, 0.985, txt, ha="center", va="top",
                 fontsize=22, weight="bold")

    ims_field, ims_err = [], []
    row_ax0 = []

    for i, (label, GT, PR, RE) in enumerate(rows):
        axes = [fig.add_subplot(gs[i, j]) for j in range(ncols)]
        ax0 = axes[0]; row_ax0.append((ax0, label))
        # GT
        im0 = ax0.imshow(GT, origin="lower", cmap=args.cmap_field,
                         vmin=vmin, vmax=vmax, interpolation="bilinear")
        ims_field.append(im0)
        set_ax_clean(ax0); ax0.set_xlabel("x"); ax0.set_ylabel("y")

        # Prediction panel
        def plot_pred(ax):
            im = ax.imshow(PR, origin="lower", cmap=args.cmap_field,
                           vmin=vmin, vmax=vmax, interpolation="bilinear")
            set_ax_clean(ax); ax.set_xlabel("x"); ax.set_ylabel("y")
            return im

        # Error panel (optional)
        def plot_err(ax):
            if args.err_style == "heat":
                dat = np.clip(RE, rmin, rmax)
                im = ax.imshow(dat, origin="lower", cmap=args.cmap_err,
                               vmin=rmin, vmax=rmax, interpolation="bilinear")
            elif args.err_style == "log":
                dat = np.log10(np.clip(RE, 1e-3, None))
                v2min, v2max = np.log10(max(1e-3, rmin)), np.log10(max(1e-3, rmax))
                im = ax.imshow(dat, origin="lower", cmap=args.cmap_err,
                               vmin=v2min, vmax=v2max, interpolation="bilinear")
            elif args.err_style == "contour":
                levels = np.geomspace(max(1e-3, rmax/400), rmax, args.err_levels)
                im = ax.contourf(RE, levels=levels, cmap=args.cmap_err)
                ax.contour(RE, levels=levels, colors="k", linewidths=0.6, alpha=0.7)
            else:  # mask
                thr = args.mask_thresh/100.0
                mask = (RE >= thr).astype(float)
                im = ax.imshow(mask, origin="lower", cmap="gray_r", vmin=0, vmax=1,
                               interpolation="nearest")
            set_ax_clean(ax); ax.set_xlabel("x"); ax.set_ylabel("y")
            return im

        if args.second == "both":
            plot_pred(axes[1]); ims_err.append(plot_err(axes[2]))
        elif args.second == "pred":
            plot_pred(axes[1])
        else:
            ims_err.append(plot_err(axes[1]))

    # make room for colorbars + left labels
    need_bottom = args.cbar_bottom + args.cbar_height + 0.012
    plt.subplots_adjust(left=args.left_margin, bottom=max(need_bottom, 0.12), top=0.95)

    # model labels on the far-left (seed omitted unless --show-seed-label)
    for ax0, label in row_ax0:
        pos = ax0.get_position()
        x = max(0.005, pos.x0 - args.model_label_offset)
        y = pos.y0 + pos.height/2.0
        plt.gcf().text(x, y, label, ha="right", va="center",
                       fontsize=args.model_label_size, weight=args.model_label_weight)

    # bottom colorbar for the field
    cax_field = fig.add_axes([args.cbar_padx, args.cbar_bottom, args.cbar_frac, args.cbar_height])
    cb0 = fig.colorbar(ims_field[0], cax=cax_field, orientation="horizontal")
    cb0.set_label("Field", fontsize=14)
    cb0.ax.tick_params(labelsize=12, length=3, width=0.8)

    # (optional) bottom colorbar for error
    if ims_err:
        left_err = 1.0 - args.cbar_padx - args.cbar_frac
        cax_err = fig.add_axes([left_err, args.cbar_bottom, args.cbar_frac, args.cbar_height])
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
