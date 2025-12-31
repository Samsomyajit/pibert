#!/usr/bin/env python3
import os, glob, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator, LogLocator

sns.set_theme(style="white", context="talk")

# ------------------------- style -------------------------
def use_jcp_style(serif=True, box_lw=1.2):
    plt.rcParams.update({
        "axes.linewidth": box_lw,
        "xtick.major.width": 1.0, "ytick.major.width": 1.0,
        "xtick.minor.width": 0.8, "ytick.minor.width": 0.8,
        "xtick.major.size": 6, "ytick.major.size": 6,
        "xtick.minor.size": 4, "ytick.minor.size": 4,
        "legend.frameon": False, "figure.dpi": 300,
        "axes.edgecolor": "black", "axes.labelcolor": "black",
        "text.color": "black", "xtick.color": "black", "ytick.color": "black",
    })
    if serif:
        plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})

def ema(x, a):
    if a <= 0:
        return np.asarray(x, float)
    x = np.asarray(x, float)
    if x.size == 0:
        return x
    y = np.empty_like(x); acc = x[0]
    for i, v in enumerate(x):
        acc = a * v + (1 - a) * (acc if i else v)
        y[i] = acc
    return y

# ------------------------- palettes -------------------------
def get_palette(name, n):
    name = (name or "okabe_ito").lower()
    if name in ("okabe", "okabe_ito", "oi", "colorblind"):
        base = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
                "#CC79A7", "#56B4E9", "#F0E442", "#999999"]
        if n <= len(base): return base[:n]
        extra = sns.color_palette("deep", n - len(base))
        return base + [sns.utils.rgb2hex(c) for c in extra]
    try:
        return [sns.utils.rgb2hex(c) for c in sns.color_palette(name, n_colors=n)]
    except Exception:
        return [sns.utils.rgb2hex(c) for c in sns.color_palette("deep", n_colors=n)]

# ------------------------- IO helpers -------------------------
def find_histories(results_root, model):
    pat = os.path.join(results_root, f"{model}_seed*", "history.csv")
    files = sorted(glob.glob(pat))
    out = []
    for p in files:
        seed = os.path.basename(os.path.dirname(p)).split("_seed")[-1]
        try:
            df = pd.read_csv(p); out.append((seed, df))
        except Exception:
            pass
    if not out:
        old = os.path.join(results_root, model, "history.csv")
        if os.path.exists(old):
            out.append(("single", pd.read_csv(old)))
    return out

def aggregate_across_seeds(dfs, metric, smooth=0.0, reducer="median", ci=0.95):
    all_epochs = sorted(set(int(e) for _, df in dfs for e in df["epoch"].dropna()))
    if not all_epochs: return np.array([]), np.array([]), None, None
    rows = []
    for _, df in dfs:
        if metric not in df.columns: continue
        d = df[["epoch", metric]].dropna().copy()
        series = dict(zip(d["epoch"].astype(int), ema(d[metric].to_numpy(), smooth)))
        rows.append([series.get(ep, np.nan) for ep in all_epochs])
    if not rows: return np.array([]), np.array([]), None, None
    A = np.array(rows, dtype=np.float64)
    valid = np.isfinite(A).any(axis=0)
    epochs = np.array(all_epochs)[valid]; A = A[:, valid]
    if A.shape[1] == 0: return np.array([]), np.array([]), None, None
    center = np.nanmedian(A, axis=0) if reducer == "median" else np.nanmean(A, axis=0)
    lo = hi = None
    if ci and ci > 0 and A.shape[0] > 1:
        q = (1 - ci) / 2.0
        lo = np.nanquantile(A, q, axis=0)
        hi = np.nanquantile(A, 1 - q, axis=0)
    return epochs, center, lo, hi

# ------------------------- plotting -------------------------
def style_axis(ax, ylabel="", ylog=True):
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
    if ylog:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=6))
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.grid(True, which="both", ls="--", lw=0.7, alpha=0.45, color="#888888")
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_linewidth(1.2)

def plot_panel(ax, models, results_root, metric, ylabel, colors,
               smooth=0.0, aggregate="median", ci=0.95, show_seeds=False):
    markers_list = ["x", "o", "s", "^"]
    seed_alpha = 0.35
    marker_size = 4.0
    z_band = 1.5
    z_line = 3.0
    single_color = "black" if len(models) == 1 else None
    for i, m in enumerate(models):
        items = find_histories(results_root, m)
        if not items:
            print(f"[WARN] no history files for {m}"); continue
        color = single_color or colors[i % len(colors)]
        mk    = markers_list[i % len(markers_list)]
        if show_seeds or aggregate == "none":
            for seed, df in items:
                if "epoch" not in df or metric not in df: continue
                x = df["epoch"].to_numpy()
                y = ema(df[metric].to_numpy(), smooth)
                if y.size == 0: continue
                ax.plot(x, y, lw=1.0, alpha=seed_alpha, color=color,
                        marker=None, zorder=1.0,
                        label=None if aggregate!="none" else f"{m} (seed {seed})")
        if aggregate != "none":
            x, center, lo, hi = aggregate_across_seeds(items, metric, smooth, aggregate, ci)
            if center.size:
                if lo is not None and hi is not None and np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
                    ax.fill_between(x, lo, hi, color=color, alpha=0.14, linewidth=0, zorder=z_band)
                ax.plot(x, center, lw=2.6, color=color, label=None, zorder=z_line,
                        marker=None)
    style_axis(ax, ylabel=ylabel, ylog=True)

# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser(description="Two-panel MPP/ECP loss curves (no legend, labels outside)")
    ap.add_argument("--results_root", default="results_cylinder_pibert_200")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--smooth", type=float, default=0.0)
    ap.add_argument("--aggregate", choices=["median", "mean", "none"], default="median")
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--show-seeds", action="store_true")
    ap.add_argument("--palette", default="okabe_ito")
    ap.add_argument("--fig-w", type=float, default=13.8)
    ap.add_argument("--fig-h", type=float, default=4.6)
    ap.add_argument("--wspace", type=float, default=0.35, help="horizontal gap between the two panels")
    ap.add_argument("--no-serif", action="store_true")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out", default="fig_mpp_ecp.png")
    ap.add_argument("--ecp-smooth", type=float, default=0.2,
                    help="EMA smoothing factor for ECP curve (0 = none).")
    ap.add_argument("--ecp-clip", type=float, default=99.0,
                    help="Percentile clamp for ECP weighted loss to suppress spikes (0 = off).")
    args = ap.parse_args()

    use_jcp_style(serif=not args.no_serif, box_lw=1.2)

    fig = plt.figure(figsize=(args.fig_w, args.fig_h), dpi=args.dpi)
    gs  = fig.add_gridspec(nrows=1, ncols=2, wspace=args.wspace, hspace=0.0)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])

    colors = get_palette(args.palette, len(args.models))

    plot_panel(axL, args.models, args.results_root,
               metric="train_mpp_w", ylabel=r"$\lambda_{\mathrm{mpp}}\cdot L_{\mathrm{mpp}}$",
               colors=colors, smooth=args.smooth, aggregate=args.aggregate,
               ci=args.ci, show_seeds=args.show_seeds)

    # ECP: clamp outlier spikes and apply additional smoothing for readability
    # Temporarily adjust smoothing and inject a clipped column into the histories.
    orig_smooth = args.smooth
    ecp_smooth = max(0.0, float(args.ecp_smooth))
    def _prepare_metric(hist_path, metric, clip_p):
        df = pd.read_csv(hist_path)
        if metric not in df:
            return df
        if clip_p and clip_p > 0:
            vals = pd.to_numeric(df[metric], errors="coerce")
            hi = np.nanpercentile(vals, clip_p)
            df[metric] = vals.clip(upper=hi)
        return df

    # monkey-patch find_histories to return clipped ECP metric
    def find_histories_clipped(results_root, model, metric, clip_p):
        pat = os.path.join(results_root, f"{model}_seed*", "history.csv")
        files = sorted(glob.glob(pat))
        out = []
        for p in files:
            seed = os.path.basename(os.path.dirname(p)).split("_seed")[-1]
            try:
                df = _prepare_metric(p, metric, clip_p)
                out.append((seed, df))
            except Exception:
                pass
        if not out:
            old = os.path.join(results_root, model, "history.csv")
            if os.path.exists(old):
                out.append(("single", _prepare_metric(old, metric, clip_p)))
        return out

    # temporarily override global finder
    global find_histories
    _orig_fh = find_histories
    find_histories = lambda root, model: find_histories_clipped(root, model, "train_ecp_w", args.ecp_clip)

    plot_panel(axR, args.models, args.results_root,
               metric="train_ecp_w", ylabel=r"$\lambda_{\mathrm{ecp}}\cdot L_{\mathrm{ecp}}$",
               colors=colors, smooth=ecp_smooth if ecp_smooth > 0 else orig_smooth,
               aggregate=args.aggregate, ci=args.ci, show_seeds=args.show_seeds)

    # restore
    find_histories = _orig_fh

    # Panel labels outside the axes boxes
    axL.text(-0.12, 1.02, "(a)", transform=axL.transAxes, ha="left", va="bottom", fontsize=16)
    axR.text(-0.12, 1.02, "(b)", transform=axR.transAxes, ha="left", va="bottom", fontsize=16)

    fig.subplots_adjust(left=0.085, right=0.985, top=0.96, bottom=0.14)
    plt.savefig(args.out, bbox_inches="tight")
    print("Saved", args.out)

if __name__ == "__main__":
    main()
