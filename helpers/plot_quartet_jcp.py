#!/usr/bin/env python3
import os, glob, argparse, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import LogLocator, MaxNLocator

sns.set_theme(style="white", context="talk")

# ---------- style ----------
def use_jcp_style(serif=True):
    plt.rcParams.update({
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0, "ytick.major.width": 1.0,
        "xtick.major.size": 6, "ytick.major.size": 6,
        "legend.frameon": False, "figure.dpi": 300,
        "axes.spines.top": True, "axes.spines.right": True,
    })
    if serif:
        plt.rcParams.update({"font.family":"serif","mathtext.fontset":"dejavuserif"})

def get_palette(name, n):
    # a few nice, color-blind safe choices
    presets = {
        "okabe_ito": ["#0072B2","#E69F00","#009E73","#D55E00","#CC79A7","#56B4E9","#F0E442","#000000"],
        "tol_bright": ["#4477AA","#EE6677","#228833","#CCBB44","#66CCEE","#AA3377","#BBBBBB"],
        "tab10": sns.color_palette("tab10", 10)
    }
    if name in presets:
        pal = presets[name]
        return pal[:max(n,1)] if isinstance(pal,list) else pal
    return sns.color_palette(name, n_colors=n)

# ---------- helpers ----------
def robust_relerr(gt, pr, percentile=95, eps_scale=1e-3):
    scale = np.percentile(np.abs(gt), percentile)
    eps = max(eps_scale * scale, 1e-8)
    return np.abs(pr - gt) / (np.abs(gt) + eps)

def load_panel_trip(npz_path, mode="mag"):
    d = np.load(npz_path)
    gt, pr = d["gt"].astype(np.float32), d["pred"].astype(np.float32)
    if mode == "mag":
        GT = np.sqrt(gt[0]**2 + gt[1]**2)
        PR = np.sqrt(pr[0]**2 + pr[1]**2)
    elif mode == "u":
        GT, PR = gt[0], pr[0]
    else:
        GT, PR = gt[1], pr[1]
    return GT, PR

def radial_spectrum_ratio(gt, pr, max_pixels=120_000, bins=24):
    # optional random downsample for speed
    H, W = gt.shape
    if H*W > max_pixels:
        stride = int(math.ceil(math.sqrt((H*W)/max_pixels)))
        gt = gt[::stride, ::stride]; pr = pr[::stride, ::stride]
        H, W = gt.shape
    # FFT magnitudes
    G = np.fft.fftshift(np.abs(np.fft.fft2(gt)))
    P = np.fft.fftshift(np.abs(np.fft.fft2(pr)))
    # radial bins
    y, x = np.indices((H, W))
    cy, cx = (H-1)/2.0, (W-1)/2.0
    r = np.sqrt((y-cy)**2 + (x-cx)**2)
    r = r / r.max()
    edges = np.linspace(0, 1.0, bins+1)
    kc = 0.5*(edges[:-1]+edges[1:])
    ratio = np.zeros(bins, dtype=float)
    for i in range(bins):
        m = (r>=edges[i]) & (r<edges[i+1])
        if np.any(m):
            g = G[m].mean(); p = P[m].mean()
            ratio[i] = (p+1e-12)/(g+1e-12)
        else:
            ratio[i] = np.nan
    # drop empty bins
    mask = np.isfinite(ratio) & (ratio>0)
    return kc[mask], ratio[mask]

# ---------- panels ----------
def plot_pareto(ax, metrics_csv, models, color_map, metric="NMSE", use_median=True):
    df = pd.read_csv(metrics_csv)
    base = f"{metric}"
    lat_col = "Latency(ms)"
    center = f"{base}_median" if use_median else f"{base}_mean"
    lo     = f"{base}_median_lo" if use_median else f"{base}_mean_lo"
    hi     = f"{base}_median_hi" if use_median else f"{base}_mean_hi"
    # may not have latency (then skip x-err)
    lat_c  = f"{lat_col}_median" if use_median else f"{lat_col}_mean"

    xs=[]; ys=[]; ss=[]; cs=[]
    labels=[]
    for m in models:
        g = df[df["Model"]==m]
        if g.empty: continue
        xs.append(float(g[lat_c].iloc[0]) if lat_c in g else np.nan)
        ys.append(float(g[center].iloc[0]))
        # use params to scale dot size if available
        ss.append(float(g["Param(M)"].iloc[0]) * 50.0 + 60.0)
        cs.append(color_map[m])
        labels.append(m)

    ax.scatter(xs, ys, s=ss, c=cs, edgecolor="k", linewidth=0.6, zorder=3)
    for x,y,l in zip(xs,ys,labels):
        ax.annotate(l, (x,y), xytext=(4,4), textcoords="offset points", fontsize=10)

    # rough frontier hint (dashed)
    if len(xs)>=2:
        x2 = np.array(xs); y2 = np.array(ys)
        sel = np.argsort(x2)
        ax.plot(x2[sel], y2[sel], ls="--", c="0.5", lw=1.0, zorder=1)

    ax.set_xlabel("Latency (ms)"); ax.set_ylabel(f"{metric} (median)")
    if np.isfinite(xs).any() and np.nanmin(xs) > 0 and np.isfinite(ys).any() and np.nanmin(ys) > 0:
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.grid(True, which="both", ls=":", lw=0.7, alpha=0.6)
    for s in ("top","right","left","bottom"): ax.spines[s].set_linewidth(1.0)
    ax.set_title("(a) Pareto: NMSE vs Latency", loc="left", fontsize=14, pad=6)

def plot_spectral(ax, results_root, models, color_map, n_samples=3, max_pixels=120_000, bins=24):
    any_line = False
    min_x = np.inf
    min_y = np.inf
    for m in models:
        # pick first available sample_XXX.npz from that model dir (seed-independent layout was saved earlier)
        pattern = os.path.join(results_root, f"{m}", "sample_*.npz")
        files = sorted(glob.glob(pattern))
        if not files:
            # also try per-seed layout
            pattern = os.path.join(results_root, f"{m}_seed*", "sample_*.npz")
            files = sorted(glob.glob(pattern))
        files = files[:max(1,n_samples)]
        xs_acc=[]; ys_acc=[]
        for p in files:
            GT, PR = load_panel_trip(p, mode="mag")
            k, r = radial_spectrum_ratio(GT, PR, max_pixels=max_pixels, bins=bins)
            if k.size:
                xs_acc.append(k); ys_acc.append(r)
        if xs_acc:
            x = np.median(np.vstack(xs_acc), axis=0)
            y = np.median(np.vstack(ys_acc), axis=0)
            ax.plot(x, y, lw=2.0, color=color_map[m], label=m)
            any_line = True
            min_x = min(min_x, np.nanmin(x))
            min_y = min(min_y, np.nanmin(y))
    if any_line and min_x > 0 and min_y > 0:
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Wavenumber k"); ax.set_ylabel(r"$E_{\mathrm{pred}}(k)/E_{\mathrm{GT}}(k)$")
    ax.grid(True, which="both", ls=":", lw=0.7, alpha=0.6)
    ax.set_title("(b) Spectral Error Ratio", loc="left", fontsize=14, pad=6)
    for s in ("top","right","left","bottom"): ax.spines[s].set_linewidth(1.0)

def plot_forest(ax, metrics_csv, models, color_map, metric="NMSE", use_median=True, per_seed_csv=None):
    agg = pd.read_csv(metrics_csv)
    base = f"{metric}"
    c  = f"{base}_median"    if use_median else f"{base}_mean"
    lo = f"{base}_median_lo" if use_median else f"{base}_mean_lo"
    hi = f"{base}_median_hi" if use_median else f"{base}_mean_hi"

    rows=[]
    for m in models:
        g = agg[agg["Model"]==m]
        if g.empty or c not in g: continue
        center=float(g[c].iloc[0])
        clo=float(g[lo].iloc[0]) if lo in g else np.nan
        chi=float(g[hi].iloc[0]) if hi in g else np.nan
        rows.append((m, center, clo, chi))
    if not rows:
        ax.text(0.5,0.5,"No metrics.csv found", ha="center", va="center"); return

    # sort by center (best first)
    rows = sorted(rows, key=lambda t: t[1])
    ys = np.arange(len(rows))[::-1]  # top = best

    min_center = np.inf
    for idx, (m, cen, clo, chi) in enumerate(rows):
        y = ys[idx]
        lo_err = float(max(cen - clo, 0)) if np.isfinite(clo) else 0.0
        hi_err = float(max(chi - cen, 0)) if np.isfinite(chi) else 0.0
        if np.isfinite(cen):
            min_center = min(min_center, cen)
        ax.errorbar(
            cen, y,
            xerr=[[lo_err], [hi_err]],
            fmt="o", ms=7, color=color_map[m], mec="k", mew=0.6,
            capsize=3, capthick=0.9, elinewidth=1.2, zorder=3
        )
        if np.isfinite(clo) and np.isfinite(chi):
            ax.plot([clo, chi], [y, y], lw=2.2, color=color_map[m], alpha=0.35, zorder=2)

    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_xlabel(metric); ax.set_ylabel("")
    if np.isfinite(min_center) and min_center > 0:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.grid(True, which="both", ls=":", lw=0.7, alpha=0.6, axis="x")
    for s in ("top","right","left","bottom"): ax.spines[s].set_linewidth(1.0)
    ax.set_title(f"(d) Forest: {metric}", loc="left", fontsize=14, pad=6)

    # optional per-seed sprinkle (grey) for context
    if per_seed_csv and os.path.exists(per_seed_csv):
        df = pd.read_csv(per_seed_csv)
        if metric in df.columns:
            for y, m in zip(ys, [r[0] for r in rows]):
                vals = df[df["Model"]==m][metric].values
                if vals.size:
                    ax.scatter(vals, np.full_like(vals, y, dtype=float), s=10,
                               color="0.25", alpha=0.35, zorder=1, marker="|")

def plot_ecdf(ax, per_seed_csv, models, color_map, metric="NMSE"):
    if not os.path.exists(per_seed_csv):
        ax.text(0.5,0.5,"metrics_per_seed.csv missing", ha="center", va="center"); return
    df = pd.read_csv(per_seed_csv)
    for m in models:
        vs = pd.to_numeric(df[df["Model"]==m][metric], errors="coerce").dropna().values
        if vs.size == 0: continue
        x = np.sort(vs); y = np.arange(1, x.size+1)/x.size
        ax.step(x, y, where="post", color=color_map[m], lw=2.0, label=m)
    ax.set_xlabel(metric); ax.set_ylabel("ECDF")
    if np.isfinite(df[metric]).any() and df[metric].dropna().gt(0).any():
        ax.set_xscale("log")
    ax.set_ylim(0,1)
    ax.grid(True, which="both", ls=":", lw=0.7, alpha=0.6)
    ax.set_title(f"(d) ECDF: {metric}", loc="left", fontsize=14, pad=6)
    for s in ("top","right","left","bottom"): ax.spines[s].set_linewidth(1.0)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Quartet figure with switchable bottom-right panel")
    ap.add_argument("--results_root", default="results_sup")
    ap.add_argument("--models", nargs="+", required=True)

    ap.add_argument("--palette", default="okabe_ito")
    ap.add_argument("--fig-w", type=float, default=12.0)
    ap.add_argument("--fig-h", type=float, default=8.8)
    ap.add_argument("--wspace", type=float, default=0.28)
    ap.add_argument("--hspace", type=float, default=0.30)
    ap.add_argument("--serif", action="store_true")

    # spectral panel args
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--max-pixels", type=int, default=120000)
    ap.add_argument("--bins", type=int, default=24)

    # bottom-right choice
    ap.add_argument("--right-bottom", choices=["forest","raincloud","ecdf"], default="forest")
    ap.add_argument("--metric-bottom", default="NMSE")

    ap.add_argument("--out", default="fig_quartet.png")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    use_jcp_style(serif=args.serif)

    models = args.models
    colors = get_palette(args.palette, len(models))
    color_map = {m: colors[i%len(colors)] for i,m in enumerate(models)}

    metrics_csv = os.path.join(args.results_root, "metrics.csv")
    per_seed_csv = os.path.join(args.results_root, "metrics_per_seed.csv")

    fig = plt.figure(figsize=(args.fig_w, args.fig_h), dpi=args.dpi)
    gs = fig.add_gridspec(2, 2, width_ratios=[1,1], height_ratios=[1,1],
                          wspace=args.wspace, hspace=args.hspace)

    axA = fig.add_subplot(gs[0,0])
    axB = fig.add_subplot(gs[0,1])
    axC = fig.add_subplot(gs[1,0])
    axD = fig.add_subplot(gs[1,1])

    # (a) Pareto (kept)
    if os.path.exists(metrics_csv):
        plot_pareto(axA, metrics_csv, models, color_map, metric="NMSE", use_median=True)
    else:
        axA.text(0.5,0.5,"metrics.csv missing", ha="center", va="center")

    # (b) Spectral error (kept)
    plot_spectral(axB, args.results_root, models, color_map,
                  n_samples=args.n_samples, max_pixels=args.max_pixels, bins=args.bins)

    # (c) Keep your calibration by |v| or repurpose; here show NMSE vs epoch (median) as a compact line
    # If you'd rather keep your old calibration panel, replace this block with it.
    try:
        # quick median val_nmse across seeds (if present)
        items = []
        for m in models:
            pattern = os.path.join(args.results_root, f"{m}_seed*", "history.csv")
            files = sorted(glob.glob(pattern))
            curves = []
            epochs = set()
            for p in files:
                df = pd.read_csv(p)
                if "epoch" in df and "val_nmse" in df:
                    epochs |= set(df["epoch"].astype(int).tolist())
                    curves.append(dict(zip(df["epoch"].astype(int), df["val_nmse"].astype(float))))
            if curves and epochs:
                xs = sorted(epochs)
                mat = np.array([[c.get(e, np.nan) for e in xs] for c in curves], float)
                med = np.nanmedian(mat, axis=0)
                if np.isfinite(med).any():
                    axC.plot(xs, med, lw=2.0, color=color_map[m], label=m)
        axC.set_xlabel("Epoch"); axC.set_ylabel("NMSE (median)")
        if axC.lines and np.nanmin([l.get_ydata().min() for l in axC.lines if l.get_ydata().size]) > 0:
            axC.set_yscale("log")
        axC.grid(True, which="both", ls=":", lw=0.7, alpha=0.6)
        axC.set_title("(c) Convergence (val NMSE)", loc="left", fontsize=14, pad=6)
        for s in ("top","right","left","bottom"): axC.spines[s].set_linewidth(1.0)
    except Exception:
        axC.text(0.5,0.5,"history.csv missing", ha="center", va="center")

    # (d) Bottom-right alternatives
    if args.right_bottom == "forest":
        plot_forest(axD, metrics_csv, models, color_map,
                    metric=args.metric_bottom, use_median=True, per_seed_csv=per_seed_csv)
    elif args.right_bottom == "ecdf":
        plot_ecdf(axD, per_seed_csv, models, color_map, metric=args.metric_bottom)
    else:
        # fallback: simple raincloud look via violin+box+points
        if not os.path.exists(per_seed_csv):
            axD.text(0.5,0.5,"metrics_per_seed.csv missing", ha="center", va="center")
        else:
            df = pd.read_csv(per_seed_csv)
            df = df[df["Model"].isin(models)].copy()
            df[args.metric_bottom] = pd.to_numeric(df[args.metric_bottom], errors="coerce")
            order = df.groupby("Model")[args.metric_bottom].median().sort_values().index.tolist()
            pal = [color_map[m] for m in order]
            sns.violinplot(data=df, x="Model", y=args.metric_bottom, order=order, palette=pal,
                           cut=0, inner=None, linewidth=0, ax=axD)
            sns.boxplot(data=df, x="Model", y=args.metric_bottom, order=order,
                        width=0.32, color="white", showcaps=True, boxprops={"zorder":3},
                        whiskerprops={"linewidth":1.2}, medianprops={"color":"k","linewidth":1.4}, ax=axD)
            sns.stripplot(data=df, x="Model", y=args.metric_bottom, order=order,
                          color="0.25", alpha=0.45, size=3.0, jitter=0.12, ax=axD)
            if df[args.metric_bottom].gt(0).any():
                axD.set_yscale("log")
            axD.set_xlabel(""); axD.set_ylabel(args.metric_bottom)
            axD.grid(True, axis="y", ls=":", lw=0.7, alpha=0.6)
            axD.set_title(f"(d) Raincloud: {args.metric_bottom}", loc="left", fontsize=14, pad=6)
            for s in ("top","right","left","bottom"): axD.spines[s].set_linewidth(1.0)

    # shared legend at bottom (horizontal)
    handles = [plt.Line2D([0],[0], color=color_map[m], lw=3) for m in models]
    labels  = models
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(models)),
               bbox_to_anchor=(0.5, -0.02), frameon=False)

    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print("Saved", args.out)

if __name__ == "__main__":
    main()
