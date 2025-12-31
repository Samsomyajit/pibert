# viz_embeddings.py
# Train-once + load checkpoints, then produce:
# - PCA grids
# - Embedding heatmaps (coolwarm, bottom horizontal cbar)
# - Score tiles (boxes) + multi-scale EPA line plot (seaborn)
# - Matrix heatmap (models × scales) for EPA R^2
#
# NOTE: this version drops the silhouette plot (we can still compute it if needed).

import argparse, os, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
# UMAP is optional; fallback handled later
try:
    from umap import UMAP
except Exception:
    UMAP = None

from src.data import make_loaders
from src.runner import (
    build_model, pick_device, WarmupCosineLR, EMA,
    train_one, eval_metrics
)
from src.models import _ddx, _ddy, _grid_spacing, FourierEmbed, WaveletLikeEmbed, ns_physics_loss  # finite-diff helpers


# ==============================
# Matplotlib / seaborn defaults (bigger fonts, no bold)
# ==============================
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})
sns.set_theme(style="white", context="talk")


# ----------------------------------------------------------
# Small helpers for robust colorbar placement & path inference
# ----------------------------------------------------------
def _add_bottom_cbar(fig, ax, sm, label, pad=0.06, height=0.03, shrink=0.80):
    """
    Adds a horizontal colorbar centered under `ax` inside `fig`,
    with clamps to avoid negative widths/heights.
    """
    bbox = ax.get_position()
    width = max(bbox.width * shrink, 0.05)
    x = bbox.x0 + (bbox.width - width) / 2.0
    y = max(bbox.y0 - pad, 0.02)
    # Clamp inside [0,1] in figure coordinates
    x = min(max(x, 0.02), 0.98 - width)
    y = min(max(y, 0.02), 0.98 - height)
    cax = fig.add_axes([x, y, width, height])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label(label)
    return cb


def _infer_results_root_from_config_path(cfg_path):
    """
    Infers a dataset-specific default results root from the config filename.
    Examples:
      config_cylinder.json -> results_cylinder
      config_tube.json     -> results_tube
      cavity.json          -> results_cavity
    """
    base = os.path.basename(cfg_path)
    stem = os.path.splitext(base)[0]
    if stem.startswith("config_"):
        stem = stem[len("config_"):]
    return f"results_{stem}"


# ----------------------------------------------------------
# Embedders (forward hooks or native features, per model)
# ----------------------------------------------------------
def get_embedder(model):
    """
    Returns: embed(xyt, cond) -> (B, N, D) tokens per grid point.

    Taps:
      - PINN .................. last hidden Tanh output  -> (B,H,W,D)
      - DeepONet2d ............ Kronecker (trunk ⊗ branch)
      - PIBERT ................ refine feature ⊕ skip(cond)
      - PIBERTNoCoords ........ refine feature ⊕ skip(cond) (legacy, no coord concat)
      - PIBERT_DeepONet2d ..... Kronecker with PIBERT branch
    """
    name = type(model).__name__
    storage = {"feat": None}
    handle = None

    def hook(_m, _inp, out):
        storage["feat"] = out.detach()

    if name == "PINN":
        tanhs = [m for m in model.net if isinstance(m, torch.nn.Tanh)]
        assert len(tanhs) > 0, "PINN: no Tanh layers found"
        handle = tanhs[-1].register_forward_hook(hook)
        kind = "pinn_tanh"

    elif name == "DeepONet2d":
        kind = "deeponet_native"

    elif name == "PIBERT":
        assert hasattr(model, "refine") and hasattr(model, "skip")
        handle = model.refine.register_forward_hook(hook)
        kind = "pibert_feat_plus_skip"

    elif name == "PIBERTNoCoords":
        assert hasattr(model, "refine") and hasattr(model, "skip")
        handle = model.refine.register_forward_hook(hook)
        kind = "pibert_feat_plus_skip"

    elif name == "PIBERT_DeepONet2d":
        kind = "pibert_deeponet_native"

    elif name == "FNO2d":
        storage["feat"] = None

        def _hook(_m, _inp, out):
            storage["feat"] = out.detach()
        handle = model.fc1.register_forward_hook(_hook)
        kind = "fno_fc1"

    else:
        raise ValueError(f"No embedder rule for {name}")

    @torch.no_grad()
    def embed(xyt, cond):
        if kind == "pinn_tanh":
            storage["feat"] = None
            _ = model(xyt, cond)  # fires hook
            z = storage["feat"]  # (B,H,W,D)
            return z.view(z.size(0), -1, z.size(-1))

        if kind == "deeponet_native":
            B, Cin, H, W = cond.shape
            T = xyt.view(B, -1, 3)
            t = model.trunk(T).view(B, -1, model.cout, model.basis)  # (B,N,C,Basis)
            b = model.branch(cond.mean(dim=[2, 3]))                   # (B,Basis)
            e = t * b.view(B, 1, 1, -1)
            return e.view(B, -1, model.cout * model.basis)

        if kind == "pibert_feat_plus_skip":
            storage["feat"] = None
            _ = model(xyt, cond)                # refine -> hook
            feat = storage["feat"]              # (B,d,H,W)
            sk   = model.skip(cond).detach()    # (B,2,H,W)
            z = torch.cat([feat, sk], dim=1)
            return z.permute(0, 2, 3, 1).reshape(z.size(0), -1, z.size(1))

        if kind == "pibert_deeponet_native":
            B, Cin, H, W = cond.shape
            x = torch.sigmoid(model.g_ff) * model.ff(cond) + \
                torch.sigmoid(model.g_wv) * model.wv(cond)
            x = model.fuse(x)
            dn = model.deeponet
            T = xyt.view(B, -1, 3)
            t = dn.trunk(T).view(B, -1, dn.cout, dn.basis)
            b = dn.branch(x.mean(dim=[2, 3]))
            e = t * b.view(B, 1, 1, -1)
            return e.view(B, -1, dn.cout * dn.basis)

        if kind == "fno_fc1":
            storage["feat"] = None
            _ = model(xyt, cond)  # runs hook on fc1
            z = storage["feat"]  # (B, Cmid, H, W)
            return z.view(z.size(0), -1, z.size(1))

    embed._hook_handle = handle
    return embed


class PIBERTNoCoords(torch.nn.Module):
    """
    Legacy PIBERT variant (no coord concatenation) to load older checkpoints.
    """
    def __init__(self, cin, cout, d=128, depth=4, heads=4, mlp=512, fourier_modes=16,
                 patch=2, nu=1e-3, w_div=1.0, w_vort=1.0, attn_dropout=0.0, ff_dropout=0.0):
        super().__init__()
        assert cout >= 1
        self.nu, self.w_div, self.w_vort = nu, w_div, w_vort
        self.use_skip = bool(cout >= 2)

        cin_joint = cin
        self.ff = FourierEmbed(cin_joint, d, modes=fourier_modes)
        self.wv = WaveletLikeEmbed(cin_joint, d)
        self.g_ff = torch.nn.Parameter(torch.tensor(0.5))
        self.g_wv = torch.nn.Parameter(torch.tensor(0.5))
        self.fuse = torch.nn.Conv2d(d, d, 1)

        self.patch_sz = int(patch)
        self.patch = torch.nn.Conv2d(d, d, kernel_size=self.patch_sz, stride=self.patch_sz)

        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=mlp,
            dropout=float(attn_dropout), batch_first=True, norm_first=True, activation="gelu"
        )
        self.enc = torch.nn.TransformerEncoder(enc_layer, num_layers=depth)

        self.refine = torch.nn.Sequential(
            torch.nn.Conv2d(d, d, 3, padding=1, groups=d), torch.nn.GELU(),
            torch.nn.Conv2d(d, d, 1), torch.nn.GELU()
        )
        self.head = torch.nn.Conv2d(d, cout, 1)
        self.skip = torch.nn.Conv2d(cin, cout, 1) if self.use_skip else None

    def forward(self, xyt, cond):
        B, _, H, W = cond.shape
        x = torch.sigmoid(self.g_ff) * self.ff(cond) + torch.sigmoid(self.g_wv) * self.wv(cond)
        x = self.fuse(x)
        x = self.patch(x)
        Hp, Wp = x.shape[-2:]
        tok = x.permute(0, 2, 3, 1).reshape(B, Hp * Wp, -1).contiguous()

        tok_dtype = tok.dtype
        tok = tok.to(torch.float32)
        tok = self.enc(tok)
        tok = tok.to(tok_dtype)

        x_low = tok.view(B, Hp, Wp, -1).permute(0, 3, 1, 2).contiguous()
        x_up  = torch.nn.functional.interpolate(x_low, size=(H, W), mode="bilinear", align_corners=False)
        x_up  = self.refine(x_up)

        out = self.head(x_up)
        if self.use_skip:
            out = out + self.skip(cond)
        return out

    def physics_loss(self, xyt, pred):
        loss, _, _ = ns_physics_loss(pred, xyt, self.nu, self.w_div, self.w_vort)
        return loss


def _filtered_state_load(model, state, label=""):
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items()
                if k in model_state and model_state[k].shape == v.shape}
    missing = sorted(set(model_state.keys()) - set(filtered.keys()))
    extra = sorted(set(state.keys()) - set(filtered.keys()))
    if label:
        print(f"[warn] {label} filtered load (missing {len(missing)}, extra {len(extra)})")
    model.load_state_dict(filtered, strict=False)
    return missing, extra


# ----------------------------------------------------------
# Physics scalars and multi-scale variants
# ----------------------------------------------------------
@torch.no_grad()
def physical_scalars(y_src, xyt):
    """
    y_src: unnormalized [u,v] tensor to derive physics from.
    Returns dict of 1D arrays (flattened HW) for speed, vorticity, divergence.
    """
    u, v = y_src[:, 0:1], y_src[:, 1:2]
    dx, dy = _grid_spacing(xyt)
    vort = _ddx(v, dx) - _ddy(u, dy)
    div  = _ddx(u, dx) + _ddy(v, dy)
    speed = (u*u + v*v).sqrt()
    out = {
        "speed":      speed.flatten(1),
        "vorticity":  vort.flatten(1),
        "divergence": div.flatten(1),
    }
    return {k: v.cpu().numpy() for k, v in out.items()}


@torch.no_grad()
def multiscale_targets_metric(y_src, xyt, metric="vorticity", scales=(1, 2, 4, 8)):
    """
    Compute chosen metric at several coarse scales by average pooling
    (proxy for multi-scale dynamics). Each coarse field is upsampled back
    to H×W and flattened. Returns dict: {scale: (B,HW) np array}.
    metric: "vorticity" (default), "speed", or "divergence".
    """
    u, v = y_src[:, 0:1], y_src[:, 1:2]
    if metric == "speed":
        base = (u * u + v * v).sqrt()  # (B,1,H,W)
    else:
        dx, dy = _grid_spacing(xyt)
        if metric == "vorticity":
            base = _ddx(v, dx) - _ddy(u, dy)
        else:  # divergence
            base = _ddx(u, dx) + _ddy(v, dy)

    outs = {}
    for s in scales:
        if s <= 1:
            vv = base
        else:
            k = int(s)
            vv = F.avg_pool2d(base, kernel_size=k, stride=k)
            vv = F.interpolate(vv, size=base.shape[-2:], mode="bilinear", align_corners=False)
        outs[int(s)] = vv.flatten(1).cpu().numpy()
    return outs


# ----------------------------------------------------------
# Data collection (embeddings + physics)
# ----------------------------------------------------------
@torch.no_grad()
def collect_embeddings(model, loader, device, unnorm, max_slices=20,
                       color_source="gt", pool="none", embed_mode="feature"):
    model.eval()
    embed = get_embedder(model)
    zs, times = [], []
    phys_cols = {"speed": [], "vorticity": [], "divergence": []}

    y_mean = torch.tensor(unnorm["y_mean"], device=device, dtype=torch.float32)
    y_std  = torch.tensor(unnorm["y_std"],  device=device, dtype=torch.float32)

    taken = 0
    for x, y, xyt in loader:
        x, y, xyt = x.to(device), y.to(device), xyt.to(device)

        # embeddings (B=1 expected)
        if embed_mode == "output":
            if color_source == "gt":
                y_src = y.float() * y_std + y_mean
            else:
                y_src = (model(xyt, x).float() * y_std + y_mean)
            z_np = y_src.cpu().numpy().reshape(y_src.shape[0], -1)  # (B, C*H*W)
        else:
            z = embed(xyt, x)[0]              # (HW, D)
            z_np = z.cpu().numpy()
        if pool == "mean":
            z_np = z_np.mean(axis=0, keepdims=True)
        elif pool == "meanvar":
            mu = z_np.mean(axis=0, keepdims=True)
            std = z_np.std(axis=0, keepdims=True)
            z_np = np.concatenate([mu, std], axis=1)  # (1, 2D)
        zs.append(z_np)

        # color/target source
        if color_source == "gt":
            y_src = y.float() * y_std + y_mean
        else:
            y_src = (model(xyt, x).float() * y_std + y_mean)

        scal = physical_scalars(y_src, xyt)
        for k in phys_cols:
            if pool in ("mean", "meanvar") or embed_mode == "output":
                phys_cols[k].append(scal[k][0].mean(keepdims=True))
            else:
                phys_cols[k].append(scal[k][0])

        tval = float(xyt[0, ..., 2].mean().item())
        times.append(np.full((z_np.shape[0],), tval, dtype=np.float32))

        taken += 1
        if taken >= max_slices:
            break

    Z = np.concatenate(zs, axis=0)                          # (N_all, D or 2D)
    T = np.concatenate(times, axis=0)
    PHYS = {k: np.concatenate(v, axis=0) for k, v in phys_cols.items()}
    return Z, T, PHYS


# ----------------------------------------------------------
# Linear-probe EPA (held-out Ridge)
# ----------------------------------------------------------
def _standardize_train_test(Z, y, tr_idx, te_idx):
    Zm, Zs = Z[tr_idx].mean(axis=0), Z[tr_idx].std(axis=0) + 1e-8
    ym, ys = y[tr_idx].mean(), y[tr_idx].std() + 1e-8
    Zt = (Z - Zm) / Zs
    yt = (y - ym) / ys
    return Zt, yt


def epa_r2_split(Z, y, test_frac=0.2, seed=0, alpha=1e-3):
    if Z.shape[0] < 4 or np.std(y) < 1e-8:
        return float("nan")
    n = Z.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    te = idx[: max(1, int(test_frac * n))]
    tr = idx[max(1, int(test_frac * n)):]
    Zt, yt = _standardize_train_test(Z, y, tr, te)
    if np.std(yt[tr]) < 1e-8:
        return float("nan")
    model = Ridge(alpha=float(alpha), fit_intercept=False, random_state=seed)
    model.fit(Zt[tr], yt[tr])
    return float(model.score(Zt[te], yt[te]))


def embedding_physics_alignment(Z, phys_dict, test_frac=0.2, seed=0, alpha=1e-3):
    return {k: epa_r2_split(Z, v, test_frac=test_frac, seed=seed, alpha=alpha)
            for k, v in phys_dict.items()}

def zscore_embed(Z, eps=1e-8):
    m = Z.mean(axis=0, keepdims=True)
    s = Z.std(axis=0, keepdims=True) + eps
    return (Z - m) / s


# ----------------------------------------------------------
# Plots
# ----------------------------------------------------------
def pca_grid_png(model_results, color_key, out_png, view=(22, -65), cmap_name="coolwarm",
                 pc_labels=("PC1", "PC2", "PC3"), pc_units="", marker_size=2.5, alpha=0.7,
                 grid_alpha=0.35, grid_width=0.8, clip_pct=99.0, symmetric=True,
                 color_vmin=None, color_vmax=None):
    """
    model_results: list of dicts with keys: name, Z, color, r2
    Plots up to 4 in a 2×2 grid with a shared horizontal colorbar (bottom).
    """
    nplot = min(4, len(model_results))

    # 2×2 grid of 3D subplots
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.0),
                             subplot_kw=dict(projection="3d"))
    axes = axes.ravel()

    # shared color scale
    all_colors = np.concatenate([m["color"] for m in model_results[:nplot]])
    if color_vmin is not None and color_vmax is not None:
        vmin, vmax = float(color_vmin), float(color_vmax)
    else:
        low = np.nanpercentile(all_colors, 100 - clip_pct)
        high = np.nanpercentile(all_colors, clip_pct)
        if symmetric:
            bound = max(abs(low), abs(high))
            vmin, vmax = -bound, bound
        else:
            vmin, vmax = low, high
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    # draw panels
    for i in range(nplot):
        ax = axes[i]
        Z = model_results[i]["Z"]
        X = PCA(n_components=3, random_state=0).fit_transform(Z)
        cvals = model_results[i]["color"]

        pts = ax.scatter(
            X[:, 0], X[:, 1], X[:, 2],
            c=cvals, cmap=cmap, norm=norm,
            s=marker_size, alpha=alpha, linewidths=0, rasterized=False
        )
        ax.view_init(elev=view[0], azim=view[1])
        unit = f" ({pc_units})" if pc_units else ""
        ax.set_xlabel(f"{pc_labels[0]}{unit}", labelpad=6)
        ax.set_ylabel(f"{pc_labels[1]}{unit}", labelpad=6)
        ax.set_zlabel(f"{pc_labels[2]}{unit}", labelpad=6)
        ax.set_title(f"{model_results[i]['name']} (EPA $R^2$={model_results[i]['r2']:.3f})",
                     fontsize=12, pad=4)
        ax.set_box_aspect([1, 1, 0.8])
        ax.grid(False)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"]['linestyle'] = "-"
            axis.line.set_color((0.4, 0.4, 0.4, 0.6))
            axis.line.set_linewidth(0.6)
            axis.set_pane_color((1, 1, 1, 0))
            axis.label.set_size(11)
            axis.set_tick_params(labelsize=9, pad=1.5, width=0.6, length=2.5)

        try:
            for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
                pane.set_facecolor((1, 1, 1, 0))
                pane.set_edgecolor((0.7, 0.7, 0.7, 0.3))
        except Exception:
            pass

    # hide unused axes if fewer than 4 models
    for j in range(nplot, 4):
        axes[j].set_visible(False)

    # shared horizontal colorbar at bottom
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=axes.tolist(), orientation="horizontal",
        fraction=0.03, pad=0.08, aspect=35
    )
    cbar.set_label(color_key)
    cbar.ax.tick_params(labelsize=11, width=0.8, length=4, pad=4)

    # leave room for colorbar/labels
    fig.tight_layout(rect=[0.03, 0.11, 0.98, 0.97])
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"[saved] {out_png}")


def tsne_grid_png(model_results, color_key, out_png, cmap_name="coolwarm",
                  marker_size=2.5, alpha=0.7, clip_pct=99.0, symmetric=True,
                  color_vmin=None, color_vmax=None, perplexity=30.0, n_iter=500,
                  seed=0, max_samples=2000, panel_colorbar=False):
    """
    2×2 grid of t-SNE (2D) scatter plots with shared colorbar.
    If panel_colorbar=True, draw a separate vertical colorbar on each panel (no shared bar).
    """
    nplot = min(4, len(model_results))
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.5))
    axes = axes.ravel()

    # shared color scale
    all_colors = np.concatenate([m["color"] for m in model_results[:nplot]])
    if color_vmin is not None and color_vmax is not None:
        vmin, vmax = float(color_vmin), float(color_vmax)
    else:
        low = np.nanpercentile(all_colors, 100 - clip_pct)
        high = np.nanpercentile(all_colors, clip_pct)
        if symmetric:
            bound = max(abs(low), abs(high))
            vmin, vmax = -bound, bound
        else:
            vmin, vmax = low, high
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    last = None
    for i in range(nplot):
        ax = axes[i]
        Z = model_results[i]["Z"]
        # optional PCA to 50D for stability
        d = min(Z.shape[1], 50)
        if d < Z.shape[1]:
            Z_low = PCA(n_components=d, random_state=seed).fit_transform(Z)
        else:
            Z_low = Z
        # subsample if too many points
        if max_samples and Z_low.shape[0] > max_samples:
            rng = np.random.default_rng(seed)
            idx = rng.choice(Z_low.shape[0], size=max_samples, replace=False)
            Z_low = Z_low[idx]
            color = model_results[i]["color"][idx]
        else:
            color = model_results[i]["color"]
        Z_low = Z_low.astype(np.float64, copy=False)
        n_pts = Z_low.shape[0]
        per = min(perplexity, max(5, n_pts // 3))
        try:
            ts = TSNE(
                n_components=2,
                perplexity=per,
                max_iter=n_iter,
                learning_rate="auto",
                init="pca",
                random_state=seed,
                verbose=0,
                method="barnes_hut",
                n_jobs=1,
            ).fit_transform(Z_low)
            coords = ts
        except Exception as e:
            print(f"[warn] TSNE failed for {model_results[i]['name']} ({e}); falling back to 2D PCA.")
            coords = PCA(n_components=2, random_state=seed).fit_transform(Z_low)
        last = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=color,
            cmap=cmap, norm=norm,
            s=marker_size, alpha=alpha, linewidths=0
        )
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_title(f"{model_results[i]['name']} (EPA $R^2$={model_results[i]['r2']:.3f})", fontsize=12)
        ax.grid(False)
        if panel_colorbar:
            sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.05, pad=0.02)
            cb.set_label(color_key)
            cb.ax.tick_params(labelsize=10, width=0.8, length=3, pad=3)

    for j in range(nplot, 4):
        axes[j].set_visible(False)

    # shared horizontal colorbar only when per-panel bars are not requested
    if last is not None and not panel_colorbar:
        cbar = fig.colorbar(
            last, ax=axes.tolist(),
            orientation="horizontal",
            fraction=0.045, pad=0.08, aspect=40
        )
        cbar.set_label(color_key)
        cbar.ax.tick_params(labelsize=11, width=0.8, length=4, pad=4)

    fig.tight_layout(rect=[0.04, 0.12, 0.98, 0.98])
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"[saved] {out_png}")


def umap_grid_png(model_results, color_key, out_png, cmap_name="coolwarm",
                  marker_size=2.5, alpha=0.7, clip_pct=99.0, symmetric=True,
                  color_vmin=None, color_vmax=None, n_neighbors=15, min_dist=0.1,
                  seed=0, max_samples=2000, panel_colorbar=False, kmeans_k=0):
    """
    2×2 grid of UMAP (2D) scatter plots. Optional per-panel colorbars and optional KMeans clustering.
    If kmeans_k > 0, clusters are colored by cluster id (tab20) and no physics colorbar is drawn.
    """
    if UMAP is None:
        print("[warn] UMAP not installed; skipping UMAP grid.")
        return

    nplot = min(4, len(model_results))
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9.5))
    axes = axes.ravel()

    all_colors = np.concatenate([m["color"] for m in model_results[:nplot]])
    if color_vmin is not None and color_vmax is not None:
        vmin, vmax = float(color_vmin), float(color_vmax)
    else:
        low = np.nanpercentile(all_colors, 100 - clip_pct)
        high = np.nanpercentile(all_colors, clip_pct)
        if symmetric:
            bound = max(abs(low), abs(high))
            vmin, vmax = -bound, bound
        else:
            vmin, vmax = low, high
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    last = None
    cluster_handles = None
    for i in range(nplot):
        ax = axes[i]
        Z = model_results[i]["Z"]
        if max_samples and Z.shape[0] > max_samples:
            rng = np.random.default_rng(seed)
            idx = rng.choice(Z.shape[0], size=max_samples, replace=False)
            Z = Z[idx]
            color = model_results[i]["color"][idx]
        else:
            color = model_results[i]["color"]
        reducer = UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="euclidean",
            random_state=seed,
        )
        coords = reducer.fit_transform(Z.astype(np.float64, copy=False))

        if kmeans_k and kmeans_k > 0:
            km = KMeans(n_clusters=kmeans_k, random_state=seed, n_init=10)
            labels = km.fit_predict(coords)
            last = ax.scatter(
                coords[:, 0], coords[:, 1],
                c=labels, cmap="tab20", s=marker_size, alpha=alpha, linewidths=0
            )
            cluster_handles = last
        else:
            last = ax.scatter(
                coords[:, 0], coords[:, 1],
                c=color, cmap=cmap, norm=norm,
                s=marker_size, alpha=alpha, linewidths=0
            )
            if panel_colorbar:
                sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
                sm.set_array([])
                cb = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.05, pad=0.02)
                cb.set_label(color_key)
                cb.ax.tick_params(labelsize=10, width=0.8, length=3, pad=3)

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(f"{model_results[i]['name']} (EPA $R^2$={model_results[i]['r2']:.3f})", fontsize=12)
        ax.grid(False)

    for j in range(nplot, 4):
        axes[j].set_visible(False)

    if last is not None and not panel_colorbar and not (kmeans_k and kmeans_k > 0):
        cbar = fig.colorbar(
            last, ax=axes.tolist(),
            orientation="horizontal",
            fraction=0.045, pad=0.08, aspect=40
        )
        cbar.set_label(color_key)
        cbar.ax.tick_params(labelsize=11, width=0.8, length=4, pad=4)
    if cluster_handles is not None and kmeans_k and kmeans_k > 0:
        cb = fig.colorbar(cluster_handles, ax=axes.tolist(), orientation="horizontal",
                          fraction=0.045, pad=0.08, aspect=40)
        cb.set_label(f"KMeans (k={kmeans_k})")
        cb.ax.tick_params(labelsize=11, width=0.8, length=4, pad=4)

    fig.tight_layout(rect=[0.04, 0.12, 0.98, 0.98])
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"[saved] {out_png}")


def heatmap_grid_png(model_results, color_key, out_png, gridsize=70, cmap_name="rainbow",
                     clip_pct=99.0, symmetric=True, color_vmin=None, color_vmax=None):
    """
    2×2 grid of PC1–PC2 hexbins, colored by average physics scalar.
    Shared vmin/vmax; single horizontal colorbar centered at the bottom.
    Each subplot is titled with the model name.
    """
    nplot = min(4, len(model_results))
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.ravel()

    # shared vmin/vmax across subplots
    allc = np.concatenate([m["color"] for m in model_results[:nplot]])
    if color_vmin is not None and color_vmax is not None:
        vmin, vmax = float(color_vmin), float(color_vmax)
    else:
        low = np.nanpercentile(allc, 100 - clip_pct)
        high = np.nanpercentile(allc, clip_pct)
        if symmetric:
            bound = max(abs(low), abs(high))
            vmin, vmax = -bound, bound
        else:
            vmin, vmax = low, high

    last_hb = None
    cmap = plt.get_cmap(cmap_name)
    for i in range(nplot):
        ax = axes[i]
        Z = model_results[i]["Z"]
        X2 = PCA(n_components=2, random_state=0).fit_transform(Z)
        last_hb = ax.hexbin(
            X2[:, 0], X2[:, 1],
            C=model_results[i]["color"],
            gridsize=gridsize,
            reduce_C_function=np.mean,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            linewidths=0
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"{model_results[i]['name']} (EPA $R^2$={model_results[i]['r2']:.3f})", fontsize=12)
        ax.axis("equal")
        ax.grid(False)

    # Single horizontal colorbar spanning all subplots at the bottom
    # (uses the actual hexbin mappable so the scale is correct)
    cbar = fig.colorbar(
        last_hb, ax=axes.tolist(),
        orientation="horizontal",
        fraction=0.01,      # height of the cbar relative to a subplot
        pad=0.01,           # space between subplots and cbar
        aspect=70
    )
    cbar.set_label(color_key)

    # Leave extra room at the bottom for the colorbar
    fig.tight_layout(rect=[0.04, 0.12, 0.98, 0.98])
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"[saved] {out_png}")




def multiscale_table_heatmap(ms_epa_by_model, out_png):
    """
    Seaborn heatmap (models × scales) with numeric annotations.
    Right-side vertical colorbar.
    """
    models = sorted(ms_epa_by_model.keys())
    scales = sorted({int(s) for d in ms_epa_by_model.values() for s in d.keys()})
    data = [[ms_epa_by_model[m][s] for s in scales] for m in models]
    df = pd.DataFrame(data, index=models, columns=scales)

    fig, ax = plt.subplots(figsize=(11, 8))
    vmin, vmax = float(df.min().min()), float(df.max().max())

    # draw heatmap without seaborn's built-in colorbar
    sns.heatmap(
        df, annot=True, fmt=".3f", cmap="Blues", vmin=vmin, vmax=vmax,
        cbar=False, linewidths=1, linecolor="white", square=True, ax=ax
    )
    ax.set_xlabel("Pooling scale (pixels)")
    ax.set_ylabel("")
    ax.set_title(r"EPA $R^2$ (vorticity) across scales")

    # right-side vertical colorbar anchored to this axes
    sm = matplotlib.cm.ScalarMappable(
        cmap="Blues",
        norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    )
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cb.set_label(r"EPA $R^2$")

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"[saved] {out_png}")


# ----------------------------------------------------------
# Minimal one-seed training + checkpoint saving (if missing)
# ----------------------------------------------------------
def ensure_trained_checkpoint(cfg, model_name, seed, device, outdir_root):
    """
    If results_<root>/<model>_seed<seed>/last.pt exists, return its path.
    Otherwise, train once using cfg and save last.pt.
    """
    model_dir = os.path.join(outdir_root, f"{model_name}_seed{seed}")
    ckpt_path = os.path.join(model_dir, "last.pt")
    if os.path.exists(ckpt_path):
        return ckpt_path

    print(f"[train] No checkpoint for {model_name} (seed {seed}); training one pass...")
    os.makedirs(model_dir, exist_ok=True)

    # Data
    train_loader, val_loader, test_loader, norm, shapes = make_loaders(
        cfg["data"]["root"],
        fmt=cfg["data"].get("format", "npz"),
        batch_size=cfg["train"].get("batch_size", 2),
        normalize=True,
    )

    # Model
    model = build_model(model_name, shapes["Cin"], shapes["Cout"], cfg.get("model_cfg", {})).to(device)

    # Train setup
    epochs       = int(cfg["train"].get("epochs", 200))
    lr           = float(cfg["train"].get("lr", 8e-4))
    weight_decay = float(cfg["train"].get("weight_decay", 0.0))
    warmup_ep    = int(cfg["train"].get("warmup_epochs", 0))
    eta_min      = float(cfg["train"].get("eta_min", 1e-5))
    lambda_phys  = float(cfg["train"].get("lambda_phys", 0.0))
    phys_warmup  = int(cfg["train"].get("phys_warmup", 10))
    grad_loss_w  = float(cfg["train"].get("grad_loss_w", 0.0))
    clip_norm    = float(cfg["train"].get("clip_norm", cfg["train"].get("clip_grad_norm", 1.0)))
    use_amp      = bool(cfg.get("amp", {}).get("enabled", True))
    amp_dtype_s  = str(cfg.get("amp", {}).get("dtype", "fp16")).lower()
    amp_dtype    = torch.float16 if amp_dtype_s in ("fp16", "float16", "half") else torch.bfloat16

    torch.manual_seed(int(seed)); np.random.seed(int(seed))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99))
    sch = WarmupCosineLR(opt, warmup_epochs=warmup_ep, total_epochs=epochs, eta_min=eta_min)
    ema = EMA(model, decay=0.999)

    # Train loop (compact)
    for ep in range(epochs):
        tr_tot, tr_dat, tr_phy, tr_div = train_one(
            model, train_loader, opt, device, norm,
            lambda_phys=lambda_phys, epoch=ep, phys_warmup=phys_warmup,
            clip_norm=clip_norm, grad_loss_w=grad_loss_w, ema=ema,
            use_amp=use_amp, amp_dtype=amp_dtype, scaler=None,
            micro_batch_size=int(cfg["train"].get("micro_batch_size", 1))
        )
        if (ep + 1) % max(10, cfg["eval"].get("eval_every", 10)) == 0 or ep == 0 or ep == epochs - 1:
            ema.apply_to(model)
            va = eval_metrics(model, val_loader, device, norm, use_amp=use_amp, amp_dtype=amp_dtype, max_batches=0)
            ema.restore(model)
            print(f"  ep {ep+1}/{epochs} | train_tot {tr_tot:.4e} | val_nmse {va[2]:.3e}")

        sch.step()
        if device.type == "mps":
            try: torch.mps.empty_cache()
            except Exception: pass

    # Save final
    torch.save({"model": model.state_dict()}, ckpt_path)
    print(f"[train] Saved {ckpt_path}")
    return ckpt_path


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON config (e.g., config_cylinder.json)")
    ap.add_argument("--models", nargs="+", default=["PINN", "DeepONet2d", "PIBERT", "PIBERT_DeepONet2d"])
    ap.add_argument("--slices", type=int, default=0, help="number of batches to sample (0 = all in chosen split)")
    ap.add_argument("--color_by", default="vorticity", choices=["speed", "vorticity", "divergence"])
    ap.add_argument("--multiscale_metric", default="vorticity", choices=["speed", "vorticity", "divergence"],
                    help="Metric to use for multi-scale EPA table.")
    ap.add_argument("--color_source", default="gt", choices=["gt", "pred"])
    ap.add_argument("--embed-mode", default="feature", choices=["feature", "output"],
                    help="Use model features (default) or flattened outputs/GT as embeddings.")
    ap.add_argument("--token-pool", default="meanvar", choices=["none", "mean", "meanvar"],
                    help="Pooling for EPA features per slice: none, mean, or mean+std concatenation.")
    ap.add_argument("--token-pool-plot", default="none", choices=["none", "mean", "meanvar"],
                    help="Pooling for plotted embeddings; defaults to none to keep density.")
    ap.add_argument("--pc-cmap", default="coolwarm", help="Colormap for PCA grid (3D and 2D).")
    ap.add_argument("--cmap", default="rainbow", help="Colormap for PCA/hexbin panels")
    ap.add_argument("--pc-marker-size", type=float, default=2.5, help="Marker size for 3D PCA grid")
    ap.add_argument("--pc-alpha", type=float, default=0.7, help="Marker alpha for 3D PCA grid")
    ap.add_argument("--pc-grid-alpha", type=float, default=0.35, help="Grid alpha for 3D PCA grid")
    ap.add_argument("--epa-pca-dim", type=int, default=20,
                    help="Optional PCA projection dim for EPA R^2 computation (0=off).")
    ap.add_argument("--tsne", action="store_true", help="Also run TSNE (2D) for plots.")
    ap.add_argument("--tsne-perplexity", type=float, default=30.0, help="TSNE perplexity.")
    ap.add_argument("--tsne-iter", type=int, default=500, help="TSNE iterations.")
    ap.add_argument("--tsne-max-samples", type=int, default=2000,
                    help="Max samples per model for TSNE (subsampled if larger).")
    ap.add_argument("--tsne-device", default="cpu", help="Device hint for TSNE (cpu recommended).")
    ap.add_argument("--panel-cbar", action="store_true",
                    help="Draw a separate colorbar on each TSNE panel (no shared bar).")
    ap.add_argument("--umap", action="store_true", help="Also run UMAP (2D) for plots.")
    ap.add_argument("--umap-nn", type=int, default=15, help="UMAP n_neighbors.")
    ap.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist.")
    ap.add_argument("--umap-max-samples", type=int, default=2000,
                    help="Max samples per model for UMAP (subsampled if larger; 0 uses all).")
    ap.add_argument("--kmeans", type=int, default=0,
                    help="If >0, run k-means on UMAP coords and color by cluster.")
    ap.add_argument("--color-clip-pct", type=float, default=99.0,
                    help="Percentile to clip color range for PCA/hexbins (e.g., 99 for robust vmin/vmax).")
    ap.add_argument("--color-symmetric", action="store_true", default=True,
                    help="Symmetrize color limits around zero using clipped max abs.")
    ap.add_argument("--color-vmin", type=float, default=None, help="Manual vmin for color scale (overrides clipping).")
    ap.add_argument("--color-vmax", type=float, default=None, help="Manual vmax for color scale (overrides clipping).")
    ap.add_argument("--test_frac", type=float, default=0.2, help="held-out fraction for EPA")
    ap.add_argument("--ridge_alpha", type=float, default=1e-3, help="Ridge alpha for EPA")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_if_missing", action="store_true", help="train one-seed model if checkpoint absent")
    ap.add_argument("--epochs_override", type=int, default=0, help="optional: override epochs only for quick runs")
    # NEW: optional override for checkpoint root
    ap.add_argument("--outdir_root", default=None,
                    help="Override checkpoint root. If unset, uses eval.outdir in config "
                         "or auto-inferred results_<config_basename>.")
    ap.add_argument("--ckpt-fallback", nargs="*", default=["results_tube", "results"],
                    help="Additional roots to search if checkpoint missing.")
    ap.add_argument("--pibert-legacy", action="store_true",
                    help="Force PIBERT checkpoints to load with legacy no-coords architecture.")
    ap.add_argument("--skip-missing", action="store_true", default=True,
                    help="Skip models with missing checkpoints instead of using random init.")
    ap.add_argument("--embed-split", default="train", choices=["train", "val", "test", "all"],
                    help="Which split to use for embeddings/EPA.")
    # CHANGED: default None; we will derive from outdir_root if omitted
    ap.add_argument("--out", default=None, help="Output directory for figures/CSVs (default: <outdir_root>/embeddings)")
    args = ap.parse_args()

    cfg = json.load(open(args.config, "r"))

    # Optionally override epochs for quick one-seed training
    if args.epochs_override > 0:
        cfg = json.loads(json.dumps(cfg))  # deep copy
        cfg["train"]["epochs"] = int(args.epochs_override)

    device = pick_device(cfg["train"].get("device", "auto"))
    print(f"[device] {device}")

    # Determine checkpoint root (results directory) robustly
    inferred_root = _infer_results_root_from_config_path(args.config)
    outdir_root = args.outdir_root or cfg.get("eval", {}).get("outdir", inferred_root)

    # Determine figure/CSV output directory
    if args.out is None:
        args.out = os.path.join(outdir_root, "embeddings")
    os.makedirs(args.out, exist_ok=True)

    print(f"[paths] outdir_root={outdir_root}")
    print(f"[paths] figures/CSVs -> {args.out}")

    # Slim loader for viz (B=1 per slice)
    train_loader, val_loader, test_loader, norm, shapes = make_loaders(
        cfg["data"]["root"], fmt=cfg["data"].get("format", "npz"),
        batch_size=1, normalize=True,
    )
    # Build list of batches per chosen split to avoid iterator exhaustion
    embed_batches = []
    def _append_batches(loader):
        for b in loader:
            embed_batches.append(b)
    if args.embed_split in ("train", "all"):
        _append_batches(train_loader)
    if args.embed_split in ("val", "all"):
        _append_batches(val_loader)
    if args.embed_split in ("test", "all"):
        _append_batches(test_loader)

    # Collect panels + metrics
    panels = []
    ms_epa_by_model = {}

    for mname in args.models[:4]:  # grid plots show up to 4
        print(f"\n=== {mname} ===")

        # Ensure a trained checkpoint exists; load it
        if args.train_if_missing:
            _ = ensure_trained_checkpoint(cfg, mname, args.seed, device, outdir_root)

        model_cfg = cfg.get("model_cfg", {})
        model = build_model(mname, shapes["Cin"], shapes["Cout"], model_cfg).to(device)
        ckpt_roots = [outdir_root] + [r for r in args.ckpt_fallback if r != outdir_root]
        ckpt = None
        for root in ckpt_roots:
            path = os.path.join(root, f"{mname}_seed{args.seed}", "last.pt")
            if os.path.exists(path):
                ckpt = path
                break
        if ckpt is not None:
            state = torch.load(ckpt, map_location=device)["model"]
            try:
                if mname == "PIBERT" and args.pibert_legacy:
                    raise RuntimeError("forcing legacy PIBERT")
                model.load_state_dict(state)
                print(f"[load] {ckpt}")
            except RuntimeError as e:
                if mname == "PIBERT":
                    force_legacy = args.pibert_legacy or any(k.startswith("pos.") for k in state.keys())
                    if force_legacy:
                        print(f"[warn] PIBERT legacy load (forced/pos-keys): using no-coords arch + filtered state")
                        model = PIBERTNoCoords(shapes["Cin"], shapes["Cout"], **(model_cfg.get("PIBERT", {}))).to(device)
                        _filtered_state_load(model, state, label="PIBERT legacy")
                    else:
                        print(f"[warn] strict load failed for PIBERT ({e}); trying filtered load on current arch")
                        missing, extra = _filtered_state_load(model, state, label="PIBERT")
                        if missing:
                            print(f"[warn] PIBERT filtered missing {len(missing)} keys; switching to legacy for completeness.")
                            model = PIBERTNoCoords(shapes["Cin"], shapes["Cout"], **(model_cfg.get("PIBERT", {}))).to(device)
                            _filtered_state_load(model, state, label="PIBERT legacy")
                        else:
                            try:
                                _ = model(torch.zeros(1, shapes["H"], shapes["W"], 3, device=device),
                                          torch.zeros(1, shapes["Cin"], shapes["H"], shapes["W"], device=device))
                                print("[info] PIBERT forward sanity passed after filtered load.")
                            except Exception as ee:
                                print(f"[warn] PIBERT forward check failed ({ee}); switching to legacy no-coords.")
                                model = PIBERTNoCoords(shapes["Cin"], shapes["Cout"], **(model_cfg.get("PIBERT", {}))).to(device)
                                _filtered_state_load(model, state, label="PIBERT legacy")
                else:
                    raise e
        else:
            msg = f"[warn] checkpoint not found for {mname} in {ckpt_roots}; "
            if args.skip_missing:
                print(msg + "skipping model.")
                continue
            else:
                print(msg + "using random init")

        # Collect embeddings + physics (GT or pred coloring)
        max_batches = None if args.slices <= 0 else args.slices
        # helper to iterate limited batches
        def _iter_batches():
            cnt = 0
            for b in embed_batches:
                yield b
                cnt += 1
                if max_batches is not None and cnt >= max_batches:
                    break
        Z_plot, T_plot, PHYS_plot = collect_embeddings(
            model, _iter_batches(), device, norm,
            max_slices=max_batches if max_batches is not None else 10**9,
            color_source=args.color_source,
            pool=args.token_pool_plot,
            embed_mode=args.embed_mode
        )
        if Z_plot.shape[0] < 2:
            print(f"[warn] Too few slices/points for {mname}; skipping.")
            continue
        Z = zscore_embed(Z_plot)
        Z_epa, PHYS_epa = Z, PHYS_plot
        # EPA pooling (per-slice features)
        if args.token_pool in ("mean", "meanvar"):
            Z_pool, _, PHYS_pool = collect_embeddings(
                model, _iter_batches(), device, norm,
                max_slices=max_batches if max_batches is not None else 10**9,
                color_source=args.color_source,
                pool=args.token_pool
            )
            if Z_pool.shape[0] >= 2:
                Z_epa = zscore_embed(Z_pool)
                PHYS_epa = {
                    k: np.concatenate([np.atleast_1d(a) for a in v], axis=0)
                    for k, v in PHYS_pool.items() if len(v) > 0
                }
        # EPA PCA
        if args.epa_pca_dim and args.epa_pca_dim > 0:
            ncomp = max(1, min(args.epa_pca_dim, Z_epa.shape[1], Z_epa.shape[0] - 1))
            Z_epa = PCA(n_components=ncomp, random_state=args.seed).fit_transform(Z_epa)
        # EPA R^2 (use chosen color_by)
        scores = embedding_physics_alignment(Z_epa, PHYS_epa, test_frac=args.test_frac,
                                             seed=args.seed, alpha=args.ridge_alpha)
        print("EPA R^2:", ", ".join([f"{k}={v:.3f}" for k, v in scores.items()]))

        # Multi-scale EPA on chosen metric (GT source, same # slices)
        ms_all = {1: [], 2: [], 4: [], 8: []}
        taken = 0
        y_mean = torch.tensor(norm["y_mean"], device=device, dtype=torch.float32)
        y_std  = torch.tensor(norm["y_std"],  device=device, dtype=torch.float32)
        for x, y, xyt in test_loader:
            x, y, xyt = x.to(device), y.to(device), xyt.to(device)
            y_src = y.float() * y_std + y_mean
            ms = multiscale_targets_metric(y_src, xyt, metric=args.multiscale_metric, scales=(1, 2, 4, 8))
            for s in ms_all:
                ms_all[s].append(ms[s][0])
            taken += 1
            if taken >= args.slices:
                break
        ms_all = {s: np.concatenate(v, axis=0) for s, v in ms_all.items()}
        ms_scores = {}
        for s, ytar in ms_all.items():
            n = min(Z_epa.shape[0], ytar.shape[0])
            Ztmp = Z_epa[:n]
            ytmp = ytar[:n]
            ms_scores[s] = epa_r2_split(Ztmp, ytmp, test_frac=args.test_frac,
                                        seed=args.seed, alpha=args.ridge_alpha)
        ms_epa_by_model[mname] = ms_scores
        print(f"Multi-scale EPA ({args.multiscale_metric}): " + ", ".join(f"s{s}={v:.3f}" for s, v in ms_scores.items()))

        panels.append({"name": mname, "Z": Z, "color": PHYS_plot[args.color_by], "r2": scores[args.color_by]})

    # --- Save all figures ---
    pca_grid_png(
        panels, args.color_by,
        os.path.join(args.out, f"pca_grid_{args.color_by}_{args.color_source}.png"),
        cmap_name=args.pc_cmap,
        marker_size=args.pc_marker_size,
        alpha=args.pc_alpha,
        grid_alpha=args.pc_grid_alpha,
        clip_pct=args.color_clip_pct,
        symmetric=args.color_symmetric,
        color_vmin=args.color_vmin,
        color_vmax=args.color_vmax,
    )
    if args.tsne:
        tsne_grid_png(
            panels, args.color_by,
            os.path.join(args.out, f"tsne_grid_{args.color_by}_{args.color_source}.png"),
            cmap_name=args.pc_cmap,
            marker_size=args.pc_marker_size,
            alpha=args.pc_alpha,
            clip_pct=args.color_clip_pct,
            symmetric=args.color_symmetric,
            color_vmin=args.color_vmin,
            color_vmax=args.color_vmax,
            perplexity=args.tsne_perplexity,
            n_iter=args.tsne_iter,
            seed=args.seed,
            max_samples=args.tsne_max_samples,
            panel_colorbar=args.panel_cbar
        )
    if args.umap:
        umap_grid_png(
            panels, args.color_by,
            os.path.join(args.out, f"umap_grid_{args.color_by}_{args.color_source}.png"),
            cmap_name=args.pc_cmap,
            marker_size=args.pc_marker_size,
            alpha=args.pc_alpha,
            clip_pct=args.color_clip_pct,
            symmetric=args.color_symmetric,
            color_vmin=args.color_vmin,
            color_vmax=args.color_vmax,
            n_neighbors=args.umap_nn,
            min_dist=args.umap_min_dist,
            seed=args.seed,
            max_samples=args.umap_max_samples,
            panel_colorbar=args.panel_cbar,
            kmeans_k=args.kmeans
        )
    multiscale_table_heatmap(
        ms_epa_by_model,
        os.path.join(args.out, f"multiscale_table_{args.multiscale_metric}_{args.color_source}.png")
    )

    # --- CSV summary ---
    rows = []
    for p in panels:
        row = {"Model": p["name"], "EPA_R2_"+args.color_by: p["r2"]}
        row.update({f"EPA_R2_ms_s{int(s)}": ms_epa_by_model[p["name"]][s]
                    for s in sorted(ms_epa_by_model[p["name"]].keys())})
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out, f"embedding_metrics_{args.color_by}_{args.color_source}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()
