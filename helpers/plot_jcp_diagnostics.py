#!/usr/bin/env python3
import os, argparse, csv, math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

sns.set_theme(style="white", context="talk")

# ------------------- JCP look -------------------
def use_jcp_style(serif=True):
    plt.rcParams.update({
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 4,
        "ytick.minor.size": 4,
        "legend.frameon": False,
        "figure.dpi": 300,
    })
    if serif:
        plt.rcParams.update({
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
        })

# ------------------------------- IO helpers -------------------------------
def load_field(npz_path, mode="mag"):
    d = np.load(npz_path)
    gt, pr = d["gt"].astype(np.float32), d["pred"].astype(np.float32)  # [2,H,W]
    if mode == "mag":
        GT = np.sqrt(gt[0]**2 + gt[1]**2)
        PR = np.sqrt(pr[0]**2 + pr[1]**2)
    elif mode == "u":
        GT, PR = gt[0], pr[0]
    elif mode == "v":
        GT, PR = gt[1], pr[1]
    else:
        raise ValueError("mode must be mag/u/v")
    return GT, PR  # [H,W]

def robust_relerr(gt, pr, percentile=95, eps_scale=1e-3):
    scale = np.percentile(np.abs(gt), percentile)
    eps = max(eps_scale * scale, 1e-8)
    return np.abs(pr - gt) / (np.abs(gt) + eps)

def ensure_range(id0, id1):
    if id1 < id0:
        id0, id1 = id1, id0
    return list(range(id0, id1+1))

def available_samples(root, model, idxs):
    out = []
    for k in idxs:
        p = os.path.join(root, model, f"sample_{k:03d}.npz")
        if os.path.exists(p):
            out.append(p)
    return out

# ------------------------------- Gradients / Vorticity / Q -------------------------------
def gradients(u, v, dx=1.0, dy=1.0):
    du_dy, du_dx = np.gradient(u, dy, dx, edge_order=2)
    dv_dy, dv_dx = np.gradient(v, dy, dx, edge_order=2)
    return du_dx, du_dy, dv_dx, dv_dy

def vorticity(u, v, dx=1.0, dy=1.0):
    du_dx, du_dy, dv_dx, dv_dy = gradients(u, v, dx, dy)
    return dv_dx - du_dy

def q_criterion(u, v, dx=1.0, dy=1.0):
    du_dx, du_dy, dv_dx, dv_dy = gradients(u, v, dx, dy)
    S11 = du_dx; S22 = dv_dy; S12 = 0.5*(du_dy + dv_dx)
    O12 = 0.5*(du_dy - dv_dx)       # O21 = -O12, O11=O22=0 (2D)
    S2 = S11**2 + S22**2 + 2.0*(S12**2)
    O2 = 2.0*(O12**2)
    return 0.5*(O2 - S2)

# ------------------------------- Spectra -------------------------------
def radial_spectrum_2d(u, v=None, window=True):
    F = u
    if window:
        wy = np.hanning(F.shape[0])[:, None]
        wx = np.hanning(F.shape[1])[None, :]
        w = wy * wx
    else:
        w = 1.0
    def spec(field):
        Fw = field * w
        S = np.fft.fftshift(np.fft.fft2(Fw))
        P = (np.abs(S)**2) / (field.size)
        return P
    P = spec(u)
    if v is not None:
        P += spec(v)
    H, W = P.shape
    ky = np.fft.fftshift(np.fft.fftfreq(H)) * H
    kx = np.fft.fftshift(np.fft.fftfreq(W)) * W
    KX, KY = np.meshgrid(kx, ky)
    KR = np.sqrt(KX**2 + KY**2)
    kmax = int(np.max(KR))
    bins = np.arange(0, kmax + 1.01, 1.0)
    which = np.digitize(KR.ravel(), bins) - 1
    Ek = np.zeros(len(bins)-1, dtype=np.float64)
    cnt = np.zeros_like(Ek)
    Pr = P.ravel()
    for i, wbin in enumerate(which):
        if 0 <= wbin < len(Ek):
            Ek[wbin] += Pr[i]; cnt[wbin] += 1
    cnt[cnt == 0] = 1
    Ek /= cnt
    kcenters = 0.5*(bins[:-1] + bins[1:])
    return kcenters, Ek

# ------------------------------- Plot utils -------------------------------
def set_spines(ax, boxed=True):
    """Either show a full box (all four spines) or despine top/right."""
    if boxed:
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(True)
            ax.spines[s].set_linewidth(1.0)
    else:
        sns.despine(ax=ax, top=True, right=True)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))

# ------------------------------- SUBCOMMAND: Spectra -------------------------------
def cmd_spectra(args):
    use_jcp_style(serif=not args.no_serif)

    idxs = ensure_range(args.idx_start, args.idx_end)
    fig, ax = plt.subplots(figsize=(8.6, 6.0), dpi=args.dpi)

    color_cycle = sns.color_palette("tab10", n_colors=max(3, len(args.models)))
    mcycle = ["x", "o", "s", "^", "D", "v"]

    for mi, model in enumerate(args.models):
        files = available_samples(args.results_root, model, idxs)
        if not files:
            print(f"[WARN] no samples for {model}")
            continue

        all_k, all_E, all_Egt = [], [], []
        for p in files:
            d = np.load(p)
            gt, pr = d["gt"].astype(np.float32), d["pred"].astype(np.float32)
            U = pr[0] if args.use_pred else gt[0]
            V = pr[1] if args.use_pred else gt[1]
            k, E = radial_spectrum_2d(U, V, window=True)
            all_k.append(k); all_E.append(E)
            if args.compare_gt and args.use_pred:
                kgt, Egt = radial_spectrum_2d(gt[0], gt[1], window=True)
                all_Egt.append(Egt)

        L = min(map(len, all_E))
        k = all_k[0][:L]
        E_stack = np.stack([e[:L] for e in all_E], axis=0)
        Em = E_stack.mean(0)
        q25, q75 = np.percentile(E_stack, [25, 75], axis=0)

        color = color_cycle[mi % len(color_cycle)]
        marker = mcycle[mi % len(mcycle)]

        if args.spectra_style == "ratio":
            if not all_Egt:
                print("[WARN] --spectra-style ratio requires --use-pred with --compare-gt.")
            else:
                Egtm = np.mean([e[:L] for e in all_Egt], axis=0)
                R = np.clip(Em / (Egtm + 1e-16), 1e-6, 1e6)
                ax.semilogx(k+1e-12, R, lw=2.2, marker=marker, markevery=args.marker_every,
                            ms=6, color=color, label=f"{model} (Pred/GT)")
                ax.axhline(1.0, color="k", lw=1.0, ls="--", alpha=0.7)
                ax.set_ylabel(r"$E_\mathrm{pred}(k)/E_\mathrm{GT}(k)$")
        else:
            if args.spectra_style == "band":
                ax.fill_between(k+1e-12, q25+1e-16, q75+1e-16, color=color, alpha=0.20, linewidth=0)
            if args.spectra_style == "area":
                ax.fill_between(k+1e-12, Em+1e-16, color=color, alpha=0.18, linewidth=0)
            ax.loglog(k+1e-12, Em+1e-16, lw=2.3, marker=marker,
                      markevery=args.marker_every, ms=6, color=color,
                      label=f"{model} {'Pred' if args.use_pred else 'GT'}")

            if args.compare_gt and args.use_pred and args.gt_outline:
                Egtm = np.mean([e[:L] for e in all_Egt], axis=0)
                ax.loglog(k+1e-12, Egtm+1e-16, lw=1.8, ls="--", color=color, alpha=0.85,
                          label=f"{model} GT")

    ax.set_xlabel(r"wavenumber $k$")
    if args.spectra_style != "ratio":
        ax.set_ylabel(r"energy spectrum $E(k)$")
    ax.grid(True, which="both", ls=":", lw=0.6, alpha=0.6)
    ax.legend(frameon=False, ncol=2)
    set_spines(ax, boxed=args.boxed)

    if args.y_format:
        ax.yaxis.set_major_formatter(FormatStrFormatter(args.y_format))
    if args.x_format:
        ax.xaxis.set_major_formatter(FormatStrFormatter(args.x_format))
    if args.xlim: ax.set_xlim(args.xlim)
    if args.ylim: ax.set_ylim(args.ylim)

    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print("Saved", args.out)

# ------------------------------- SUBCOMMAND: Vorticity/Q panel -------------------------------
def panel_field(ax, F, cmap, vmin, vmax, title=None, boxed=True):
    im = ax.imshow(F, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="bilinear")
    if title: ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    set_spines(ax, boxed=boxed)
    return im

def cmd_vortq(args):
    use_jcp_style(serif=not args.no_serif)

    trips = []
    for model in args.models:
        p = os.path.join(args.results_root, model, f"sample_{args.idx:03d}.npz")
        if not os.path.exists(p):
            print(f"[WARN] missing {p}")
            continue
        d = np.load(p)
        gt, pr = d["gt"].astype(np.float32), d["pred"].astype(np.float32)
        w_gt = vorticity(gt[0], gt[1]); w_pr = vorticity(pr[0], pr[1])
        Q_gt = q_criterion(gt[0], gt[1]); Q_pr = q_criterion(pr[0], pr[1])
        RE_w = robust_relerr(w_gt, w_pr); RE_Q = robust_relerr(Q_gt, Q_pr)
        trips.append((model, w_gt, w_pr, RE_w, Q_gt, Q_pr, RE_Q))

    if not trips:
        raise SystemExit("No samples to plot.")

    w_absmax = max(np.max(np.abs(t[1])) for t in trips)
    Q_absmax = max(np.max(np.abs(t[4])) for t in trips)
    w_vmin, w_vmax = -w_absmax, w_absmax
    Q_vmin, Q_vmax = -Q_absmax, Q_absmax
    rmax_w = np.percentile(np.concatenate([t[3].ravel() for t in trips]), 99.5)
    rmax_Q = np.percentile(np.concatenate([t[6].ravel() for t in trips]), 99.5)

    n = len(trips)
    # figure size scales with row count; gaps & margins are user-tunable
    fig = plt.figure(figsize=(args.fig_width, args.row_height * n + 1.6), dpi=args.dpi)
    gs = fig.add_gridspec(nrows=n, ncols=6, hspace=args.hspace, wspace=args.wspace)

    headers = [r"$\omega$ GT", r"$\omega$ Pred", r"$\omega$ RelErr",
               r"$Q$ GT", r"$Q$ Pred", r"$Q$ RelErr"]
    for j, txt in enumerate(headers):
        fig.text((j+0.5)/6, 0.985, txt, ha="center", va="top", fontsize=20, weight="bold")

    ims_field, ims_err = [], []
    for i, (name, wgt, wpr, rew, Qgt, Qpr, reQ) in enumerate(trips):
        ax0 = fig.add_subplot(gs[i, 0]); ax1 = fig.add_subplot(gs[i, 1]); ax2 = fig.add_subplot(gs[i, 2])
        ax3 = fig.add_subplot(gs[i, 3]); ax4 = fig.add_subplot(gs[i, 4]); ax5 = fig.add_subplot(gs[i, 5])

        im0 = panel_field(ax0, wgt, args.cmap_field, w_vmin, w_vmax, boxed=args.boxed)
        im1 = panel_field(ax1, wpr, args.cmap_field, w_vmin, w_vmax, boxed=args.boxed)
        im2 = panel_field(ax2, np.clip(rew, 0, rmax_w), args.cmap_err, 0, rmax_w, boxed=args.boxed)
        im3 = panel_field(ax3, Qgt, args.cmap_field, Q_vmin, Q_vmax, boxed=args.boxed)
        im4 = panel_field(ax4, Qpr, args.cmap_field, Q_vmin, Q_vmax, boxed=args.boxed)
        im5 = panel_field(ax5, np.clip(reQ, 0, rmax_Q), args.cmap_err, 0, rmax_Q, boxed=args.boxed)

        # model label on far left margin (user-tunable x)
        fig.text(args.model_label_x,
                 ax0.get_position().y0 + ax0.get_position().height/2,
                 name, rotation=90, va="center", ha="left", fontsize=14)

        ims_field.extend([im0, im3]); ims_err.extend([im2, im5])

    # margins + bottom colorbars
    plt.subplots_adjust(left=args.left, right=args.right, top=args.top, bottom=args.bottom)

    cax_field = fig.add_axes([args.cbar_padax, args.cbar_bottom, args.cbar_frac, args.cbar_height])
    cax_err   = fig.add_axes([1.0 - args.cbar_padax - args.cbar_frac, args.cbar_bottom,
                              args.cbar_frac, args.cbar_height])

    cb0 = fig.colorbar(ims_field[0], cax=cax_field, orientation="horizontal")
    cb0.set_label("Field value (ω or Q)", fontsize=13); cb0.ax.tick_params(labelsize=11)
    cb2 = fig.colorbar(ims_err[0], cax=cax_err, orientation="horizontal")
    cb2.set_label("Relative Error", fontsize=13); cb2.ax.tick_params(labelsize=11)

    fig.savefig(args.out, bbox_inches="tight")
    print("Saved", args.out)

# --- helper: pick the first column that exists (case-insensitive) ---
def _pick_col(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

# ------------------------------- SUBCOMMAND: Pareto -------------------------------
def cmd_pareto(args):
    import pandas as pd
    use_jcp_style(serif=not args.no_serif)

    df = pd.read_csv(args.metrics_csv)
    if args.show_cols:
        print("Columns in CSV:", list(df.columns))
        return

    if args.models:
        df = df[df["Model"].isin(args.models)]

    xcol = args.x_metric or _pick_col(df, [
        "Latency(ms)", "Latency_ms", "Latency", "Time(ms)", "TimeMs", "Inference(ms)",
        "Param(M)", "Params(M)", "Parameters(M)", "Params"
    ])
    if xcol is None:
        raise SystemExit("Could not find a suitable x-axis column. Use --show-cols then --x-metric <column>.")

    ycol = args.y_metric
    if ycol not in df.columns:
        raise SystemExit(f"y-metric '{ycol}' not found. Available: {list(df.columns)}")

    for c in (xcol, ycol):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[xcol, ycol])

    fig, ax = plt.subplots(figsize=(7.8, 5.6), dpi=args.dpi)
    colors = sns.color_palette("tab10", n_colors=len(df))
    markers = ["x", "o", "s", "^", "D", "v"]

    for i, (_, r) in enumerate(df.iterrows()):
        ax.scatter(r[xcol], r[ycol], s=80, edgecolor="k", linewidth=0.8,
                   marker=markers[i % len(markers)], color=colors[i % len(colors)])
        if args.annotate:
            ax.text(r[xcol], r[ycol], "  " + str(r["Model"]),
                    fontsize=12, va="center", ha="left")

    # Pareto front (lower-left)
    pts = df[[xcol, ycol]].to_numpy()
    order = np.argsort(pts[:, 0])
    best, cur = [], float("inf")
    for i in order:
        if pts[i, 1] < cur:
            best.append(i); cur = pts[i, 1]
    if len(best) >= 2:
        front = pts[best]
        ax.plot(front[:, 0], front[:, 1], "-o", lw=2.2, alpha=0.9, color="black",
                markersize=5, label="Pareto front")

    ax.set_xlabel(xcol); ax.set_ylabel(ycol)
    if args.xlog: ax.set_xscale("log")
    if args.ylog: ax.set_yscale("log")
    if args.x_format: ax.xaxis.set_major_formatter(FormatStrFormatter(args.x_format))
    if args.y_format: ax.yaxis.set_major_formatter(FormatStrFormatter(args.y_format))

    ax.grid(True, ls=":", lw=0.7, alpha=0.6)
    ax.legend(frameon=False, loc="best")
    set_spines(ax, boxed=args.boxed)
    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print("Saved", args.out)

# ------------------------------- Main / CLI -------------------------------
def main():
    p = argparse.ArgumentParser(description="JCP-friendly diagnostics: spectra, vorticity/Q panels, Pareto.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # spectra
    ps = sub.add_parser("spectra", help="Radial energy spectra E(k)")
    ps.add_argument("--models", nargs="+", required=True)
    ps.add_argument("--results_root", default="results_sup")
    ps.add_argument("--idx-start", type=int, default=0)
    ps.add_argument("--idx-end", type=int, default=199)
    ps.add_argument("--use-pred", action="store_true", help="Use predictions instead of GT")
    ps.add_argument("--compare-gt", action="store_true", help="Also plot GT spectra for the same samples")
    ps.add_argument("--gt-outline", action="store_true", help="GT as dashed outline when --compare-gt")
    ps.add_argument("--spectra-style", choices=["mean", "band", "area", "ratio"], default="band",
                    help="mean line; mean ± IQR band; filled area; or Pred/GT ratio")
    ps.add_argument("--marker-every", dest="marker_every", type=int, default=8)
    ps.add_argument("--xlim", type=float, nargs=2, default=None)
    ps.add_argument("--ylim", type=float, nargs=2, default=None)
    ps.add_argument("--x-format", default=None)
    ps.add_argument("--y-format", default=None)
    ps.add_argument("--boxed", action="store_true", default=True, help="draw full box around axes")
    ps.add_argument("--no-boxed", dest="boxed", action="store_false")
    ps.add_argument("--no-serif", action="store_true", help="use default sans font")
    ps.add_argument("--dpi", type=int, default=300)
    ps.add_argument("--out", default="fig_spectra.pdf")
    ps.set_defaults(func=cmd_spectra)

    # vorticity + Q
    pv = sub.add_parser("vortq", help="Vorticity & Q-criterion GT/Pred/RelErr panels")
    pv.add_argument("--models", nargs="+", required=True)
    pv.add_argument("--results_root", default="results_sup")
    pv.add_argument("--idx", type=int, default=0, help="sample index to plot")
    pv.add_argument("--cmap_field", default="coolwarm", help="diverging (for signed ω, Q)")
    pv.add_argument("--cmap_err", default="viridis", help="perceptual (for >=0 RelErr)")
    pv.add_argument("--boxed", action="store_true", default=True, help="draw full boxes around panels")
    pv.add_argument("--no-boxed", dest="boxed", action="store_false")
    pv.add_argument("--no-serif", action="store_true")
    # spacing & size controls
    pv.add_argument("--fig-width",   type=float, default=17.0)
    pv.add_argument("--row-height",  type=float, default=4.1, help="height per model row")
    pv.add_argument("--hspace",      type=float, default=0.38, help="vertical gap between rows")
    pv.add_argument("--wspace",      type=float, default=0.30, help="horizontal gap between columns")
    # margins
    pv.add_argument("--left",   type=float, default=0.08)
    pv.add_argument("--right",  type=float, default=0.985)
    pv.add_argument("--top",    type=float, default=0.965)
    pv.add_argument("--bottom", type=float, default=0.12)
    # model label position
    pv.add_argument("--model-label-x", dest="model_label_x", type=float, default=0.006,
                    help="figure fraction for model name (x), left margin")
    
    pv.add_argument("--model-label-dy", type=float, default=0.0,
                help="vertical offset for model name in figure fraction (positive = up)")

    # bottom colorbars
    pv.add_argument("--cbar-frac",   type=float, default=0.30, help="width of each bottom colorbar (figure fraction)")
    pv.add_argument("--cbar-height", type=float, default=0.012)
    pv.add_argument("--cbar-bottom", type=float, default=0.045)
    pv.add_argument("--cbar-padax",  type=float, default=0.07)
    pv.add_argument("--dpi", type=int, default=300)
    pv.add_argument("--out", default="fig_vortq.pdf")
    pv.set_defaults(func=cmd_vortq)

    # pareto
    pp = sub.add_parser("pareto", help="Error vs cost Pareto (from metrics.csv)")
    pp.add_argument("--metrics-csv", default="results/metrics.csv")
    pp.add_argument("--models", nargs="*", default=None)
    pp.add_argument("--y-metric", default="NMSE")
    pp.add_argument("--x-metric", default=None, help="e.g. 'Latency(ms)' or 'Param(M)'")
    pp.add_argument("--xlog", action="store_true")
    pp.add_argument("--ylog", action="store_true")
    pp.add_argument("--annotate", action="store_true", help="write model labels next to points")
    pp.add_argument("--x-format", default=None, help="e.g. '%.2f' to mimic JCP tick formatting")
    pp.add_argument("--y-format", default=None)
    pp.add_argument("--boxed", action="store_true", default=True, help="draw full box around axes")
    pp.add_argument("--no-boxed", dest="boxed", action="store_false")
    pp.add_argument("--no-serif", action="store_true")
    pp.add_argument("--dpi", type=int, default=300)
    pp.add_argument("--out", default="fig_pareto.pdf")
    pp.add_argument("--show-cols", action="store_true", help="print CSV columns and exit")
    pp.set_defaults(func=cmd_pareto)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
