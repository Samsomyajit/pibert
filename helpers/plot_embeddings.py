# plot_embeddings.py
import os, argparse, csv, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from glob import glob

from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity, pairwise_distances
from sklearn.manifold import SpectralEmbedding
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sns.set_theme(style="white", context="talk")

def apply_jcp_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.linewidth": 1.2,
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "figure.dpi": 300,
    })

# ---------- I/O & preprocessing ----------
def load_field(npz_path, mode="mag"):
    d = np.load(npz_path)
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
    return GT, PR

def vectorize_field(F):
    return F.astype(np.float32).ravel()

def fft_vector(F, keep=32):
    spec = np.fft.fftshift(np.fft.fft2(F))
    mag = np.abs(spec)
    h, w = mag.shape
    k = int(min(keep, h // 2, w // 2))
    cy, cx = h // 2, w // 2
    patch = mag[cy - k:cy + k, cx - k:cx + k]
    patch = (patch / (patch.max() + 1e-8)).astype(np.float32)
    return patch.ravel()

def zscore(X, eps=1e-8):
    m, s = X.mean(0, keepdims=True), X.std(0, keepdims=True)
    return (X - m) / (s + eps)

# ---------- Gamma selection ----------
def _parse_gamma_spec(X, spec, rng=np.random.RandomState(0)):
    if spec is None or (isinstance(spec, str) and spec.strip() == ""):
        return None
    if isinstance(spec, (int, float)):
        return float(spec)
    spec = spec.lower()
    if spec == "scale":
        var = float(X.var()) + 1e-12
        return 1.0 / (X.shape[1] * var)
    if spec == "auto":
        return 1.0 / X.shape[1]
    if spec == "median":
        m = min(2000, X.shape[0])
        idx = rng.choice(X.shape[0], m, replace=False)
        D = pairwise_distances(X[idx], metric="euclidean")
        med = np.median(D[D > 0])
        sigma2 = (med ** 2) + 1e-12
        return 1.0 / (2.0 * sigma2)
    try:
        return float(spec)
    except Exception:
        return None

# ---------- Affinity build ----------
def build_hybrid_affinity(X_feat, X_spec, alpha=0.6, gamma_spec="scale",
                          connect_eps=1e-3, rng=np.random.RandomState(0)):
    gamma = _parse_gamma_spec(X_feat, gamma_spec, rng=rng)
    Kf = rbf_kernel(X_feat, gamma=gamma)
    Ks = cosine_similarity(X_spec)
    Ks = (Ks + 1.0) * 0.5
    Kf /= (Kf.max() + 1e-12)
    Ks /= (Ks.max() + 1e-12)
    K = alpha * Kf + (1.0 - alpha) * Ks
    # ensure connectivity: mix a tiny uniform kernel
    if connect_eps > 0:
        n = K.shape[0]
        K = (1.0 - connect_eps) * K + connect_eps * (np.ones((n, n), dtype=K.dtype) / n)
    return K

# ---------- Plot helpers ----------
def add_axes_labels(ax):
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.grid(False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=12, width=1.0)

def add_side_colorbar(ax, mappable, label):
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="4%", pad=0.05)
    cbar = plt.colorbar(mappable, cax=cax)
    cbar.set_label(label, fontsize=12)
    return cbar

def style_colorbar(cbar, tick_labelsize=11, width=1.0):
    cbar.ax.tick_params(labelsize=tick_labelsize, width=width, length=4, pad=4)

def list_sample_indices(model_dir):
    files = glob(os.path.join(model_dir, "sample_*.npz"))
    idxs = []
    for f in files:
        try:
            stem = os.path.basename(f).split(".")[0]
            idx = int(stem.split("_")[1])
            idxs.append(idx)
        except Exception:
            continue
    return sorted(set(idxs))

def resolve_model_dir(results_root, model):
    """
    Pick a directory under results_root that matches the requested model.
    Prefer exact, then "<model>_seed*", then other "<model>_*", then any prefix.
    Returns (path, display_name) or (None, model) if not found.
    """
    exact = os.path.join(results_root, model)
    if os.path.isdir(exact):
        return exact, model
    try:
        entries = os.listdir(results_root)
    except FileNotFoundError:
        return None, model
    lower_model = model.lower()
    candidates = []
    for d in entries:
        full = os.path.join(results_root, d)
        if not os.path.isdir(full):
            continue
        dl = d.lower()
        if dl == lower_model:
            pr = 0
        elif dl.startswith(f"{lower_model}_seed"):
            pr = 1
        elif dl.startswith(f"{lower_model}_"):
            pr = 2
        elif dl.startswith(lower_model):
            pr = 3
        else:
            continue
        candidates.append((pr, d))
    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]))
        best = candidates[0][1]
        return os.path.join(results_root, best), best
    return None, model

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--results_root", default="results")
    ap.add_argument("--mode", choices=["mag", "u", "v"], default="mag")
    ap.add_argument("--idx-start", type=int, default=0)
    ap.add_argument("--idx-end", type=int, default=199)
    ap.add_argument("--use-pred", action="store_true",
                    help="Embed predictions instead of GT")
    ap.add_argument("--alpha", type=float, default=0.6,
                    help="Hybrid mix: features vs spectrum")
    ap.add_argument("--fft-keep", type=int, default=32,
                    help="Low-frequency square size for FFT")
    ap.add_argument("--rbf-gamma", default="scale",
                    help="RBF gamma for features: float, 'scale', 'auto', or 'median'")
    ap.add_argument("--connect-eps", type=float, default=1e-3,
                    help="Mix-in weight for uniform kernel to ensure graph connectivity")
    ap.add_argument("--label-csv", type=str, default=None,
                    help="Optional CSV with columns: idx,<label-col> (header required)")
    ap.add_argument("--label-col", type=str, default="label")
    ap.add_argument("--label-display", type=str, default=None,
                    help="Pretty name for colorbar/legend (defaults to --label-col).")
    ap.add_argument("--label-units", type=str, default=None,
                    help="Units text appended to the label (e.g., '1/s' for vorticity).")
    ap.add_argument("--dim-units", type=str, default=None,
                    help="Units for embedding axes (e.g., 'a.u.' or 'm/s'); appended in parentheses.")
    ap.add_argument("--pc-labels", nargs=3, default=["PC1", "PC2", "PC3"],
                    help="Labels for PCA axes (3D uses all 3).")
    ap.add_argument("--dim-labels", nargs=2, default=["dim 1", "dim 2"],
                    help="Labels for spectral embedding axes.")
    ap.add_argument("--no-titles", action="store_true",
                    help="Omit plot titles for a cleaner, publication-style layout.")
    ap.add_argument("--auto-samples", action="store_true", default=True,
                    help="Auto-detect available sample_*.npz indices from the model directory.")
    ap.add_argument("--out", default="fig_embeddings.pdf")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--pca-baseline", action="store_true",
                    help="Also plot PCA of feature vectors for comparison")
    ap.add_argument("--pca-3d", action="store_true",
                    help="Render PCA baseline as 3D (PC1/2/3) scatter")
    ap.add_argument("--cmap", default="rainbow",
                    help="Colormap for embedding points (default: rainbow)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (median heuristic, etc.)")
    ap.add_argument("--jcp-style", action="store_true",
                    help="Apply publication-style fonts/layout.")
    ap.add_argument("--point-size", type=float, default=36.0)
    ap.add_argument("--point-alpha", type=float, default=0.9)
    ap.add_argument("--edge-width", type=float, default=0.35)
    args = ap.parse_args()

    if args.pca_3d and not args.pca_baseline:
        print("[INFO] --pca-3d requested; enabling PCA baseline output.")
        args.pca_baseline = True

    rng = np.random.RandomState(args.seed)
    cmap = plt.get_cmap(args.cmap)
    label_text = args.label_display or args.label_col
    if args.label_units:
        label_text = f"{label_text} [{args.label_units}]"

    # labels (optional)
    labels = None
    if args.label_csv and os.path.exists(args.label_csv):
        lab = {}
        with open(args.label_csv) as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    idx = int(row["idx"])
                except Exception:
                    continue
                val = row.get(args.label_col, None)
                if val is None:
                    continue
                try:
                    val = float(val)
                except Exception:
                    val = str(val)
                lab[idx] = val
        labels = lab

    idxs = list(range(args.idx_start, args.idx_end + 1))
    n_models = len(args.models)
    ncols = 2 if args.pca_baseline else 1

    if args.jcp_style:
        apply_jcp_style()
        base_w, base_h = 8.4, 5.0
    else:
        base_w, base_h = 7.6, 4.6

    fig, axes = plt.subplots(n_models, ncols, figsize=(base_w * ncols, base_h * n_models),
                             dpi=args.dpi, squeeze=False)

    any_plotted = False

    for i, model in enumerate(args.models):
        model_dir, disp_name = resolve_model_dir(args.results_root, model)
        if model_dir is None:
            print(f"[WARN] Could not find directory for model '{model}' under {args.results_root}")
            continue
        if disp_name != model:
            print(f"[INFO] Using '{disp_name}' for model '{model}' (found in {args.results_root})")

        feats, specs, labs = [], [], []
        idxs_model = list_sample_indices(model_dir) if args.auto_samples else idxs
        if not idxs_model:
            idxs_model = idxs
        for idx in idxs_model:
            p = os.path.join(model_dir, f"sample_{idx:03d}.npz")
            if not os.path.exists(p):
                continue
            GT, PR = load_field(p, mode=args.mode)
            F = PR if args.use_pred else GT
            feats.append(vectorize_field(F))
            specs.append(fft_vector(F, keep=args.fft_keep))
            if labels is not None and idx in labels:
                labs.append(labels[idx])
            else:
                labs.append(idx)

        n = len(feats)
        if n < 2:
            print(f"[WARN] Too few samples for {model}, skipping")
            continue

        any_plotted = True

        Xf = zscore(np.vstack(feats))
        Xs = zscore(np.vstack(specs))
        K = build_hybrid_affinity(Xf, Xs, alpha=args.alpha,
                                  gamma_spec=args.rbf_gamma,
                                  connect_eps=args.connect_eps, rng=rng)

        # choose safe dimensionality
        n_components = 2 if n >= 3 else 1

        # try spectral embedding; fall back to PCA if it fails
        try:
            emb = SpectralEmbedding(n_components=n_components,
                                    affinity="precomputed").fit_transform(K)
        except Exception as e:
            print(f"[INFO] SpectralEmbedding failed for {model} ({e}); falling back to PCA.")
            emb = PCA(n_components=min(2, n)).fit_transform(Xf)

        # pad to 2D if needed
        if emb.shape[1] == 1:
            emb = np.column_stack([emb[:, 0], np.zeros_like(emb[:, 0]) + 1e-3 * rng.randn(n)])

        ax = axes[i, 0]
        hue = np.array(labs, dtype=object)

        # detect numeric vs categorical labels
        is_numeric = False
        try:
            vals = hue.astype(float)
            is_numeric = True
        except Exception:
            is_numeric = False

        if is_numeric:
            vals = hue.astype(float)
            sc = ax.scatter(
                emb[:, 0],
                emb[:, 1],
                s=args.point_size,
                c=vals,
                cmap=cmap,
                edgecolor="none" if args.edge_width <= 0 else "white",
                linewidth=args.edge_width,
                alpha=args.point_alpha,
            )
            cbar = add_side_colorbar(ax, sc, label_text)
            style_colorbar(cbar)
        else:
            uniq = np.unique(hue)
            pal = cmap(np.linspace(0.1, 0.9, len(uniq)))
            color_map = {u: pal[k] for k, u in enumerate(uniq)}
            colors = [color_map[x] for x in hue]
            ax.scatter(
                emb[:, 0],
                emb[:, 1],
                s=args.point_size,
                c=colors,
                edgecolor="none" if args.edge_width <= 0 else "white",
                linewidth=args.edge_width,
                alpha=args.point_alpha,
            )
            ax.legend(handles=[plt.Line2D([0], [0], marker='o', linestyle='',
                                          markerfacecolor=color_map[u], markeredgecolor="white",
                                          markersize=7, label=str(u))
                               for u in uniq],
                      title=label_text, frameon=False, ncol=4, fontsize=10)

        unit_suffix = f" ({args.dim_units})" if args.dim_units else ""
        ax.set_xlabel(f"{args.dim_labels[0]}{unit_suffix}")
        ax.set_ylabel(f"{args.dim_labels[1]}{unit_suffix}")
        add_axes_labels(ax)
        ax.tick_params(labelsize=12, width=1.0, length=4, pad=4)
        ax.tick_params(labelsize=12, width=1.0)

        # quick separability metric if categorical & enough classes
        if not is_numeric:
            try:
                lab_ids = np.unique(hue, return_inverse=True)[1]
                if len(np.unique(lab_ids)) >= 2 and emb.shape[0] >= 3:
                    sil = silhouette_score(emb, lab_ids, metric="euclidean")
                    ax.text(0.01, 0.98, f"silhouette = {sil:.2f}", transform=ax.transAxes,
                            ha="left", va="top", fontsize=11)
            except Exception:
                pass

        if args.pca_baseline:
            axp = axes[i, 1]
            if args.pca_3d:
                axp.remove()
                axp = fig.add_subplot(n_models, ncols, i * ncols + 2, projection="3d")
                axes[i, 1] = axp

            ncomp = 3 if args.pca_3d else 2
            pca = PCA(n_components=min(ncomp, n)).fit_transform(Xf)
            if pca.shape[1] < ncomp:
                pad = np.zeros((n, ncomp - pca.shape[1]), dtype=pca.dtype)
                pad += 1e-3 * rng.randn(*pad.shape)
                pca = np.hstack([pca, pad])

            if is_numeric:
                if args.pca_3d:
                    sc2 = axp.scatter(
                        pca[:, 0], pca[:, 1], pca[:, 2],
                        s=args.point_size, c=vals, cmap=cmap,
                        edgecolor="none" if args.edge_width <= 0 else "white",
                        linewidth=args.edge_width, alpha=args.point_alpha,
                    )
                else:
                    sc2 = axp.scatter(
                        pca[:, 0], pca[:, 1],
                        s=args.point_size, c=vals, cmap=cmap,
                        edgecolor="none" if args.edge_width <= 0 else "white",
                        linewidth=args.edge_width, alpha=args.point_alpha,
                    )
                if args.pca_3d:
                    cbar2 = fig.colorbar(sc2, ax=axp, fraction=0.03, pad=0.015, shrink=0.8)
                else:
                    cbar2 = add_side_colorbar(axp, sc2, label_text)
                cbar2.set_label(label_text, fontsize=12)
                style_colorbar(cbar2)
            else:
                if args.pca_3d:
                    axp.scatter(
                        pca[:, 0], pca[:, 1], pca[:, 2],
                        s=args.point_size, c=colors,
                        edgecolor="none" if args.edge_width <= 0 else "white",
                        linewidth=args.edge_width, alpha=args.point_alpha,
                    )
                else:
                    axp.scatter(
                        pca[:, 0], pca[:, 1],
                        s=args.point_size, c=colors,
                        edgecolor="none" if args.edge_width <= 0 else "white",
                        linewidth=args.edge_width, alpha=args.point_alpha,
                    )

            axp.grid(False)
            if args.pca_3d:
                unit_suffix = f" ({args.dim_units})" if args.dim_units else ""
                axp.set_xlabel(f"{args.pc_labels[0]}{unit_suffix}")
                axp.set_ylabel(f"{args.pc_labels[1]}{unit_suffix}")
                axp.set_zlabel(f"{args.pc_labels[2]}{unit_suffix}")
                axp.view_init(elev=18, azim=-60)
                axp.set_box_aspect([1, 1, 0.8])
                axp.tick_params(labelsize=10, width=0.9, pad=2, length=3)
                # soften panes for cleaner look
                try:
                    for pane in (axp.xaxis.pane, axp.yaxis.pane, axp.zaxis.pane):
                        pane.set_facecolor((1, 1, 1, 0))
                        pane.set_edgecolor((0.2, 0.2, 0.2, 0.6))
                    for axis in (axp.xaxis, axp.yaxis, axp.zaxis):
                        axis.line.set_color((0.2, 0.2, 0.2, 0.8))
                        axis.line.set_linewidth(1.0)
                except Exception:
                    pass
            else:
                unit_suffix = f" ({args.dim_units})" if args.dim_units else ""
                axp.set_xlabel(f"{args.pc_labels[0]}{unit_suffix}")
                axp.set_ylabel(f"{args.pc_labels[1]}{unit_suffix}")
                add_axes_labels(axp)
                axp.tick_params(labelsize=12, width=1.0, length=4, pad=4)

    if not any_plotted:
        print("[ERROR] No samples found for any model. Check --results_root and indices.")
        return

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print("Saved", args.out)

if __name__ == "__main__":
    main()
