#!/usr/bin/env python3
import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="white", context="talk")

# ------------------------- style -------------------------
def use_jcp_style(serif=True):
    plt.rcParams.update({
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0, "ytick.major.width": 1.0,
        "xtick.major.size": 6, "ytick.major.size": 6,
        "legend.frameon": False, "figure.dpi": 300,
        # keep it boxed, no top/right spines on heatmap axes
        "axes.spines.top": False, "axes.spines.right": False,
    })
    if serif:
        plt.rcParams.update({"font.family": "serif",
                             "mathtext.fontset": "dejavuserif"})

# ------------------------- stats -------------------------
def bootstrap_ci(a, n_boot=2000, ci=0.95, reducer="median", rng=None):
    """Return (center, lo, hi) bootstrap CI; ignores NaNs."""
    x = np.asarray(a, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    if x.size == 1:
        return (float(x[0]), float(x[0]), float(x[0]))
    rng = np.random.default_rng(rng)
    if reducer == "mean":
        stat = np.mean
    else:
        stat = np.median
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = stat(x[idx], axis=1)
    lo = np.quantile(boots, (1-ci)/2.0)
    hi = np.quantile(boots, 1-(1-ci)/2.0)
    return float(stat(x)), float(lo), float(hi)

def fmt_value(v):
    if not np.isfinite(v):
        return "—"
    av = abs(v)
    if av == 0:
        return "0"
    if av < 1e-3 or av >= 9999:
        return f"{v:.2e}"
    if av < 0.1:
        return f"{v:.3f}"
    if av < 1:
        return f"{v:.3f}"
    if av < 10:
        return f"{v:.3f}"
    return f"{v:.3g}"

# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser(description="JCP-style heatmap of per-seed metrics (medians with CIs)")
    ap.add_argument("--results_csv", default="results_sup/metrics_per_seed.csv",
                    help="per-seed CSV produced by runner.py (metrics_per_seed.csv)")
    ap.add_argument("--metrics", nargs="+",
                    default=["MAE(1e-3)", "MSE", "NMSE"],
                    help="columns from metrics_per_seed.csv to include as heatmap columns")
    ap.add_argument("--log_metrics", nargs="*", default=[],
                    help="metrics to log10 before aggregating (e.g., MSE NMSE)")
    ap.add_argument("--palette", default="mako_r",
                    help="sequential palette for coloring (e.g., 'mako_r','rocket','viridis')")
    ap.add_argument("--coloring", choices=["value","rank"], default="rank",
                    help="heatmap colors by raw value (comparable units) or by per-metric rank (column-comparable)")
    ap.add_argument("--reducer", choices=["median","mean"], default="median",
                    help="aggregate across seeds using median (default) or mean")
    ap.add_argument("--ci", type=float, default=0.95, help="bootstrap CI level")
    ap.add_argument("--annot_ci", action="store_true",
                    help="annotate as value±half-CI; otherwise just value")
    ap.add_argument("--order_models_by", choices=["overall","alpha","custom"], default="overall",
                    help="row order: overall (by average rank), alpha (alphabetical), custom (as in CSV order)")
    ap.add_argument("--serif", action="store_true", help="use serif font (JCP look)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--fig_w", type=float, default=8.8)
    ap.add_argument("--fig_h", type=float, default=3.6)
    ap.add_argument("--out", default="fig_metrics_heatmap.png")
    args = ap.parse_args()

    use_jcp_style(serif=args.serif)

    if not os.path.exists(args.results_csv):
        raise SystemExit(f"Missing {args.results_csv}. Run runner.py first.")

    df = pd.read_csv(args.results_csv)
    need_cols = {"Model","Seed"} | set(args.metrics)
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in CSV: {missing}")

    # numeric metrics
    for m in args.metrics:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    # optional log10 transform (safer aggregation for heavy-tailed errors)
    df_trans = df.copy()
    for m in args.log_metrics:
        if m in df_trans.columns:
            df_trans[m] = np.log10(df_trans[m].clip(min=1e-12))

    # aggregate per model
    models = sorted(df_trans["Model"].unique().tolist())
    center = pd.DataFrame(index=models, columns=args.metrics, dtype=float)
    lo = center.copy(); hi = center.copy()

    for m in models:
        sub = df_trans[df_trans["Model"] == m]
        for met in args.metrics:
            c, l, h = bootstrap_ci(sub[met].values, ci=args.ci, reducer=args.reducer)
            center.loc[m, met] = c; lo.loc[m, met] = l; hi.loc[m, met] = h

    # choose coloring matrix
    if args.coloring == "value":
        color_mat = center.to_numpy(dtype=float)
        cbar_label = "Lower is better"
    else:
        # rank each column (lower value = better rank 1)
        ranks = center.rank(axis=0, method="min", ascending=True)
        # scale to [0,1] for consistent colorbar
        color_mat = (ranks - 1) / (len(models) - 1 if len(models) > 1 else 1)
        color_mat = color_mat.to_numpy(dtype=float)
        cbar_label = "Relative rank (0 = best)"

    # pick model order
    if args.order_models_by == "alpha":
        order = sorted(models)
    elif args.order_models_by == "overall" and args.coloring == "rank":
        avg_rank = pd.Series(color_mat.mean(axis=1), index=center.index)
        order = list(avg_rank.sort_values(ascending=True).index)
    else:
        order = models  # custom / CSV order

    center = center.loc[order]
    lo = lo.loc[order]
    hi = hi.loc[order]
    color_mat = color_mat[[models.index(o) for o in order], :]

    # build annotation strings (back-transform logs for display)
    def back(val, met):
        return (10**val) if met in args.log_metrics else val

    ann = []
    for i, mdl in enumerate(center.index):
        row = []
        for j, met in enumerate(center.columns):
            v = back(center.iloc[i, j], met)
            if args.annot_ci and np.isfinite(lo.iloc[i, j]) and np.isfinite(hi.iloc[i, j]):
                lo_v = back(lo.iloc[i, j], met)
                hi_v = back(hi.iloc[i, j], met)
                half = 0.5 * (hi_v - lo_v)
                row.append(f"{fmt_value(v)}\n±{fmt_value(half)}")
            else:
                row.append(f"{fmt_value(v)}")
        ann.append(row)
    ann = np.array(ann, dtype=object)

    # figure
    fig = plt.figure(figsize=(args.fig_w, args.fig_h), dpi=args.dpi)
    ax = fig.add_subplot(1,1,1)

    # seaborn heatmap
    cmap = sns.color_palette(args.palette, as_cmap=True)
    g = sns.heatmap(
        color_mat, ax=ax, cmap=cmap, cbar=True,
        linewidths=0.7, linecolor="#DDDDDD",
        annot=ann, fmt="", annot_kws={"fontsize":12},
        xticklabels=list(center.columns), yticklabels=list(center.index),
        vmin=0.0 if args.coloring=="rank" else None,
        vmax=1.0 if args.coloring=="rank" else None
    )

    # aesthetics
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=15)
    ax.tick_params(axis="y", rotation=0)
    # box look
    for s in ("left","bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(1.0)

    # move cbar to bottom, short and classy
    cbar = g.collections[0].colorbar
    cbar.ax.tick_params(length=3, width=0.8, labelsize=11)
    cbar.set_label(cbar_label, fontsize=12)
    # tighten
    plt.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print("Saved", args.out)

if __name__ == "__main__":
    main()
