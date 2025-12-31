#!/usr/bin/env python3
# JCP-style 2×(3×N) grid: [u row, v row] × [ GT | Pred | RelErr ] per sample block
import os, math, json, random, argparse, re
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from einops import rearrange

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# -------------------- Style (JCP-ish) --------------------
def use_jcp_style(serif=True):
    sns.set_theme(style="white", context="talk")
    plt.rcParams.update({
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "legend.frameon": False,
        "figure.dpi": 300,
    })
    if serif:
        plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})

def set_clean_axes(ax, show_ylabel=False, show_xlabel=True):
    sns.despine(ax=ax, top=True, right=True)
    if not show_ylabel:
        ax.set_yticks([]); ax.set_ylabel("")
    else:
        ax.set_ylabel("y", fontsize=14)
    if not show_xlabel:
        ax.set_xticks([]); ax.set_xlabel("")
    else:
        ax.set_xlabel("x", fontsize=14)
    ax.tick_params(length=4, width=1.0, labelsize=12)
    ax.set_aspect("equal")

def robust_relerr(gt, pr, percentile=95, eps_scale=1e-3):
    scale = np.percentile(np.abs(gt), percentile)
    eps = max(eps_scale * scale, 1e-8)
    return np.abs(pr - gt) / (np.abs(gt) + eps)

def nmse(gt, pr):
    mse = np.mean((pr - gt) ** 2)
    var = np.var(gt)
    return float(mse / (var + 1e-12))

# -------------------- Minimal model (matches your ckpt family) --------------------
class FourierEmbed(nn.Module):
    def __init__(self, c_in, dim, modes=16):
        super().__init__()
        self.modes = modes
        self.wr = nn.Parameter(torch.randn(c_in, dim, modes, modes) * 0.02)
        self.wi = nn.Parameter(torch.randn(c_in, dim, modes, modes) * 0.02)
    def forward(self, x):
        B, C, H, W = x.shape
        X = torch.fft.rfft2(x.to(torch.float32), norm="ortho")
        Wc = torch.complex(self.wr, self.wi)
        out_ft = torch.zeros(B, Wc.shape[1], H, X.size(-1), dtype=torch.complex64, device=x.device)
        mh = min(self.modes, X.size(-2)); mw = min(self.modes, X.size(-1))
        out_ft[:, :, :mh, :mw] = torch.einsum("bchw,cdhw->bdhw", X[:, :, :mh, :mw], Wc[:, :, :mh, :mw])
        y = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho").real
        return y

class WaveletLikeEmbed(nn.Module):
    def __init__(self, c_in, dim):
        super().__init__()
        k_ll = torch.ones((3, 3), dtype=torch.float32) / 9.0
        k_lh = torch.tensor([[1,1,1],[0,0,0],[-1,-1,-1]], dtype=torch.float32) / 3.0
        k_hl = torch.tensor([[1,0,-1],[1,0,-1],[1,0,-1]], dtype=torch.float32) / 3.0
        k_hh = torch.tensor([[1,0,-1],[0,0,0],[-1,0,1]], dtype=torch.float32) / 2.0
        kernels = torch.stack([k_ll, k_lh, k_hl, k_hh], 0)
        self.depthwise = nn.Conv2d(c_in, c_in*4, 3, 1, 1, groups=c_in, bias=False)
        with torch.no_grad():
            w = torch.zeros_like(self.depthwise.weight)
            for ci in range(c_in):
                w[ci*4+0, 0] = kernels[0]
                w[ci*4+1, 0] = kernels[1]
                w[ci*4+2, 0] = kernels[2]
                w[ci*4+3, 0] = kernels[3]
            self.depthwise.weight.copy_(w)
        for p in self.depthwise.parameters(): p.requires_grad_(False)
        self.proj = nn.Conv2d(c_in*4, dim, 1, bias=True)
    def forward(self, x): return self.proj(self.depthwise(x))

class BottleneckAdapter(nn.Module):
    def __init__(self, d_model, down=32):
        super().__init__()
        self.down = nn.Linear(d_model, down, False)
        self.up   = nn.Linear(down, d_model, False)
        nn.init.zeros_(self.up.weight)
    def forward(self, x): return self.up(F.gelu(self.down(x)))

class PIBERT(nn.Module):
    def __init__(self, img_ch=3, prm_dim=16, d=256, depth=4, heads=4, mlp=768,
                 fourier_modes=20, use_adapters=True, adapter_dim=32, grad_ckpt=True):
        super().__init__()
        self.hybrid = nn.ModuleDict({"ff": FourierEmbed(img_ch, d, modes=fourier_modes),
                                     "wv": WaveletLikeEmbed(img_ch, d)})
        self.g_ff = nn.Parameter(torch.tensor(0.5))
        self.g_wv = nn.Parameter(torch.tensor(0.5))
        self.fuse = nn.Conv2d(d, d, 1)
        self.prm_fc = nn.Linear(prm_dim, d)
        enc_layer = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=mlp, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.grad_ckpt = grad_ckpt
        self.use_adapters = use_adapters
        self.adapters = nn.ModuleList([BottleneckAdapter(d, adapter_dim) for _ in range(depth)]) if use_adapters else None
        self.to_img = nn.Linear(d, img_ch)
    def spectral_embed(self, x):
        y = self.g_ff.sigmoid()*self.hybrid["ff"](x) + self.g_wv.sigmoid()*self.hybrid["wv"](x)
        return self.fuse(y)
    def forward(self, img, prm):
        b,c,h,w = img.shape
        x = self.spectral_embed(img)                      # [B,d,H,W]
        x = rearrange(x, "b d h w -> b (h w) d")
        x = torch.cat([self.prm_fc(prm).unsqueeze(1), x], 1)
        for li, layer in enumerate(self.enc.layers):
            if self.grad_ckpt and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
            if self.use_adapters: x = x + self.adapters[li](x)
        x = x[:, 1:]
        x = rearrange(x, "b (h w) d -> b d h w", h=h, w=w)
        return self.to_img(x.permute(0,2,3,1)).permute(0,3,1,2)

# -------------------- Data helpers --------------------
_U_PAT = re.compile(r"(?:^|[_\\-])(?:u|ux|vx?)\\b", re.I)
_V_PAT = re.compile(r"(?:^|[_\\-])(?:v|uy|vy?)\\b", re.I)
_P_PAT = re.compile(r"(?:^|[_\\-])(?:p|pres|pressure)\\b", re.I)

def _domain_onehot(path: Path):
    d = ["bc","prop","geo"]
    parts = [p.lower() for p in path.parts]
    return np.array([int(k in parts) for k in d], dtype=np.float32)

def _load_casejson(case_dir: Path):
    js = case_dir / "case.json"
    scal = []
    if js.exists():
        try:
            j = json.load(open(js, "r"))
            for _, v in (j.items() if isinstance(j, dict) else []):
                if isinstance(v, (int,float)) and math.isfinite(v): scal.append(float(v))
        except Exception: pass
    return np.array(scal, dtype=np.float32)

def _make_prm(case_dir: Path, max_extra=13):
    dom = _domain_onehot(case_dir)
    extra = _load_casejson(case_dir)
    if extra.size > max_extra: extra = extra[:max_extra]
    if extra.size < max_extra: extra = np.pad(extra, (0, max_extra - extra.size))
    return np.concatenate([dom, extra]).astype(np.float32)

def _find_uvp_npy(files):
    lower = {f.name.lower(): f for f in files if f.suffix==".npy"}
    u = lower.get("u.npy") or next((f for f in files if _U_PAT.search(f.name.lower())), None)
    v = lower.get("v.npy") or next((f for f in files if _V_PAT.search(f.name.lower())), None)
    p = lower.get("p.npy") or next((f for f in files if _P_PAT.search(f.name.lower())), None)
    return (u,v,p) if (u and v) else None

def _fast_shape(npy_path: Path): return np.load(npy_path, mmap_mode="r").shape

def _try_uvp_lengths(u_path: Path, v_path: Path, p_path: Path|None):
    su, sv = _fast_shape(u_path), _fast_shape(v_path)
    if len(su)<2 or len(sv)<2: return None
    Hu,Wu = su[-2], su[-1]; Hv,Wv = sv[-2], sv[-1]
    if (Hu,Wu)!=(Hv,Wv): return None
    Tu = su[0] if len(su)==3 else 1; Tv = sv[0] if len(sv)==3 else 1
    if p_path and p_path.exists():
        sp = _fast_shape(p_path); Hp,Wp = sp[-2], sp[-1]
        if (Hp,Wp)!=(Hu,Wu): return None
        Tp = sp[0] if len(sp)==3 else 1; T = min(Tu,Tv,Tp)
    else: T = min(Tu,Tv)
    return T,Hu,Wu

def scan_cases(root: Path, stride=1, debug=False):
    items_by_case = {}
    for case in root.rglob("*"):
        if not case.is_dir(): continue
        npys = list(case.glob("*.npy"))
        uvp = _find_uvp_npy(npys) if npys else None
        if not uvp: continue
        u_path,v_path,p_path = uvp
        check = _try_uvp_lengths(u_path,v_path,p_path)
        if check is None: continue
        T,H,W = check
        for t in range(0, T, stride):
            items_by_case.setdefault(case, []).append(("npy_uvp", (u_path,v_path,p_path,H,W), t))
    if debug:
        print(f"[scan] cases={len(items_by_case)} frames={sum(len(v) for v in items_by_case.values())}")
    return items_by_case

def split_cases(items_by_case, split, seed=42):
    rng = random.Random(seed)
    cases = list(items_by_case.keys()); rng.shuffle(cases)
    n = len(cases); n_train = int(split["train"]*n); n_val = int(split["val"]*n)
    train = set(cases[:n_train]); val = set(cases[n_train:n_train+n_val]); test = set(cases[n_train+n_val:])
    return train, val, test

def resize_if_needed(x, size):
    C,H,W = x.shape
    if (H,W)==(size,size): return x
    with torch.no_grad():
        xt = torch.from_numpy(x).unsqueeze(0)
        xr = F.interpolate(xt, size=(size,size), mode="bilinear", align_corners=False)
    return xr.squeeze(0).numpy()

class CFDBenchCasewise(torch.utils.data.Dataset):
    def __init__(self, items_by_case, case_set, img_size, norm=None, prm_dim=16):
        self.samples = []
        for c in case_set:
            for tup in items_by_case[c]:
                self.samples.append((c, tup))
        self.img_size = img_size; self.norm = norm; self.prm_dim = prm_dim
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        case_dir, (mode, payload, t) = self.samples[idx]
        u_path,v_path,p_path,H,W = payload
        u = np.load(u_path, mmap_mode="r")[t]
        v = np.load(v_path, mmap_mode="r")[t]
        p = np.load(p_path, mmap_mode="r")[t] if p_path and Path(p_path).exists() else np.zeros((H,W), np.float32)
        u = u[...,0] if (u.ndim==3 and u.shape[-1]==1) else u
        v = v[...,0] if (v.ndim==3 and v.shape[-1]==1) else v
        p = p[...,0] if (p.ndim==3 and p.shape[-1]==1) else p
        x = np.stack([u,v,p]).astype(np.float32)
        x = resize_if_needed(x, self.img_size)
        if self.norm is not None:
            m,s = self.norm["mean"], self.norm["std"]; x = (x - m)/(s+1e-8)
        prm = _make_prm(case_dir)
        return torch.from_numpy(x), torch.from_numpy(prm)

def compute_train_norm(ds, max_samples=2000):
    n=0; mean=torch.zeros(3); M2=torch.zeros(3)
    for i in range(min(len(ds), max_samples)):
        x,_=ds[i]; x=x.float()
        ch_mean = x.view(3,-1).mean(1); ch_var = x.view(3,-1).var(1, unbiased=False)
        n+=1; delta=ch_mean-mean; mean=mean+delta/n; M2=M2+delta*(ch_mean-mean)+ch_var
    var = M2/max(n,1)
    return {"mean": mean.view(3,1,1).numpy().astype(np.float32),
            "std":  torch.sqrt(var).view(3,1,1).numpy().astype(np.float32)}

# -------------------- Infer PIBERT cfg from checkpoint --------------------
def infer_pibert_cfg_from_state_dict(sd: dict):
    # d (embed dim)
    if "hybrid.ff.wr" in sd:
        d = sd["hybrid.ff.wr"].shape[1]
        modes = sd["hybrid.ff.wr"].shape[2]
    else:
        d = sd["enc.layers.0.self_attn.out_proj.weight"].shape[0]
        modes = 16
    # depth
    layer_ids = []
    for k in sd.keys():
        m = re.match(r"enc\.layers\.(\d+)\.", k)
        if m: layer_ids.append(int(m.group(1)))
    depth = (max(layer_ids) + 1) if layer_ids else 4
    # mlp dim
    mlp = sd.get("enc.layers.0.linear1.weight", torch.empty(768, d)).shape[0]
    # adapters
    use_adapters = any(k.startswith("adapters.0.") for k in sd.keys())
    adapter_dim = sd.get("adapters.0.down.weight", torch.empty(32, d)).shape[0] if use_adapters else 32
    # heads: choose a sensible divisor of d
    for h in (8, 4, 16, 2):
        if d % h == 0:
            heads = h; break
    else:
        heads = 4
    return dict(d=d, depth=depth, heads=heads, mlp=mlp, fourier_modes=modes,
                use_adapters=use_adapters, adapter_dim=adapter_dim)

# -------------------- Plotting --------------------
def plot_uv_grid_jcp(x_true_list, x_pred_list, out_path,
                     sample_titles=None,
                     field_cmap="RdBu_r", err_cmap="magma",
                     serif=True):
    """
    x_true_list / x_pred_list: list of arrays [C,H,W] (C>=2). Only u,v (0,1) are plotted.
    Layout: rows = [u, v], columns = 3*N → (GT | Pred | RelErr)*N
    """
    use_jcp_style(serif=serif)
    N = len(x_true_list)
    assert N == len(x_pred_list), "Mismatched lists"

    Xgt = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in x_true_list]
    Xpr = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in x_pred_list]

    # symmetric per-row color range
    vmins = []; vmaxs = []
    for ch in (0,1):
        all_vals = np.concatenate([np.ravel(a[ch]) for a in (Xgt+Xpr)]) if Xgt else np.array([0.0])
        vmax = np.percentile(np.abs(all_vals), 99.5) if all_vals.size else 1.0
        vmins.append(-vmax); vmaxs.append(vmax)

    # err cap per-row
    rmax = []
    for ch in (0,1):
        all_re = []
        for i in range(N):
            re = robust_relerr(Xgt[i][ch], Xpr[i][ch])
            all_re.append(re.ravel())
        rmax.append(np.percentile(np.concatenate(all_re), 99.5) if all_re else 1.0)

    nmse_uv = [[nmse(Xgt[i][ch], Xpr[i][ch]) for ch in (0,1)] for i in range(N)]

    # figure dims
    fig_w = max(12.0, 3.8 * 3 * N / 2.0)
    fig_h = 7.6
    fig, axes = plt.subplots(2, 3*N, figsize=(fig_w, fig_h), dpi=300)

    # column headers
    for i in range(N):
        base = 3*i
        for j, title in enumerate(["GT", "Prediction", "Relative Error"]):
            axes[0, base+j].set_title(title, fontsize=18, weight="bold", pad=8)

    # sample titles above center column
    if sample_titles:
        for i in range(N):
            mid_ax = axes[0, 3*i+1]
            pos = mid_ax.get_position()
            fig.text(pos.x0 + 0.5*pos.width, 0.985, str(sample_titles[i]),
                     ha="center", va="top", fontsize=16)

    # left labels
    fig.text(0.012, axes[0,0].get_position().y0 + axes[0,0].get_position().height/2,
             r"$u$", rotation=90, va="center", ha="left", fontsize=18, weight="bold")
    fig.text(0.012, axes[1,0].get_position().y0 + axes[1,0].get_position().height/2,
             r"$v$", rotation=90, va="center", ha="left", fontsize=18, weight="bold")

    field_handles = [None, None]
    err_handles   = [None, None]

    for row, ch in enumerate((0,1)):  # u, v
        for i in range(N):
            base = 3*i
            gt = Xgt[i][ch]; pr = Xpr[i][ch]
            re = robust_relerr(gt, pr)

            # GT
            ax = axes[row, base+0]
            im0 = ax.imshow(gt, origin="lower", cmap=field_cmap, vmin=vmins[row], vmax=vmaxs[row], interpolation="bilinear")
            set_clean_axes(ax, show_ylabel=(base==0), show_xlabel=True)
            if field_handles[row] is None: field_handles[row] = im0
            ax.text(0.02, 0.98, f"NMSE={nmse_uv[i][row]:.3g}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=13,
                    bbox=dict(facecolor="white", edgecolor="0.5", boxstyle="round,pad=0.25", alpha=0.9))

            # Pred
            ax = axes[row, base+1]
            ax.imshow(pr, origin="lower", cmap=field_cmap, vmin=vmins[row], vmax=vmaxs[row], interpolation="bilinear")
            set_clean_axes(ax, show_ylabel=False, show_xlabel=True)
            ax.text(0.02, 0.98, f"NMSE={nmse_uv[i][row]:.3g}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=13,
                    bbox=dict(facecolor="white", edgecolor="0.5", boxstyle="round,pad=0.25", alpha=0.9))

            # RelErr
            ax = axes[row, base+2]
            im2 = ax.imshow(np.clip(re, 0, rmax[row]), origin="lower", cmap=err_cmap, vmin=0, vmax=rmax[row], interpolation="bilinear")
            set_clean_axes(ax, show_ylabel=False, show_xlabel=True)
            if err_handles[row] is None: err_handles[row] = im2

    # spacing
    plt.subplots_adjust(left=0.08, right=0.985, top=0.95, bottom=0.12, wspace=0.25, hspace=0.35)

    # bottom colorbars (row-wise)
    cbar_h = 0.014
    row_u_y = 0.025
    row_v_y = 0.055
    frac = 0.30
    padx = 0.10

    # u row
    cax_fu = fig.add_axes([padx, row_u_y, frac, cbar_h])
    cax_eu = fig.add_axes([1.0 - padx - frac, row_u_y, frac, cbar_h])
    cb_fu = fig.colorbar(field_handles[0], cax=cax_fu, orientation="horizontal"); cb_fu.set_label("Field (u)", fontsize=13)
    cb_eu = fig.colorbar(err_handles[0],   cax=cax_eu, orientation="horizontal"); cb_eu.set_label("Relative Error", fontsize=13)

    # v row
    cax_fv = fig.add_axes([padx, row_v_y, frac, cbar_h])
    cax_ev = fig.add_axes([1.0 - padx - frac, row_v_y, frac, cbar_h])
    cb_fv = fig.colorbar(field_handles[1], cax=cax_fv, orientation="horizontal"); cb_fv.set_label("Field (v)", fontsize=13)
    cb_ev = fig.colorbar(err_handles[1],   cax=cax_ev, orientation="horizontal"); cb_ev.set_label("Relative Error", fontsize=13)

    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")

# -------------------- CLI / main --------------------
def main():
    parser = argparse.ArgumentParser(description="JCP-style 2x(3*N) UV grid from PIBERT checkpoint")
    parser.add_argument("--data_root", default="./cfdb_tmp/extracted")
    parser.add_argument("--ckpt", default="./checkpoints/pibert_best.pt")
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--prm_dim", type=int, default=16)
    parser.add_argument("--N", type=int, default=3, help="number of samples (blocks)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="fig_uv_grid_jcp.png")
    parser.add_argument("--serif", action="store_true", help="use serif fonts")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[Device] {device}")

    # ---------- scan + split ----------
    items_by_case = scan_cases(Path(args.data_root), stride=1, debug=True)
    if len(items_by_case) == 0:
        print("[ERROR] No cases found under --data_root. Ensure it contains case subfolders with u.npy and v.npy (and optional p.npy).")
        return
    tr, va, te = split_cases(items_by_case, {"train":0.8, "val":0.1, "test":0.1}, seed=args.seed)

    # ---------- normalization from train ----------
    ds_train = CFDBenchCasewise(items_by_case, tr, args.img_size, norm=None, prm_dim=args.prm_dim)
    norm = compute_train_norm(ds_train, max_samples=2000)
    ds_test  = CFDBenchCasewise(items_by_case, te, args.img_size, norm=norm, prm_dim=args.prm_dim)

    # ---------- load checkpoint & infer cfg ----------
    ckpt = torch.load(args.ckpt, map_location=device)
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    mcfg = ckpt.get("mcfg", None)
    if mcfg is None:
        mcfg = infer_pibert_cfg_from_state_dict(sd)
        print(f"[info] inferred cfg from checkpoint: {mcfg}")
    else:
        print(f"[info] using saved cfg: {mcfg}")

    model = PIBERT(img_ch=3, prm_dim=args.prm_dim,
                   d=int(mcfg.get("d", 256)),
                   depth=int(mcfg.get("depth", 4)),
                   heads=int(mcfg.get("heads", 4)),
                   mlp=int(mcfg.get("mlp", 768)),
                   fourier_modes=int(mcfg.get("fourier_modes", 20)),
                   use_adapters=bool(mcfg.get("use_adapters", True)),
                   adapter_dim=int(mcfg.get("adapter_dim", 32)),
                   grad_ckpt=bool(mcfg.get("grad_ckpt", True))).to(device)

    # strict load; fall back to non-strict with a warning
    try:
        missing, unexpected = model.load_state_dict(sd, strict=True)
        if missing or unexpected:
            print(f"[warn] strict load reported missing={len(missing)} unexpected={len(unexpected)}")
    except RuntimeError as e:
        print(f"[warn] strict load failed ({e}); retrying with strict=False")
        model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt}")

    # ---------- gather N samples ----------
    x_true_list, x_pred_list, titles = [], [], []
    with torch.no_grad():
        for i in range(min(args.N, len(ds_test))):
            x, prm = ds_test[i]
            x = x.to(device); prm = prm.to(device)
            yp = model(x.unsqueeze(0), prm.unsqueeze(0)).squeeze(0).cpu()
            x_true_list.append(x.cpu())
            x_pred_list.append(yp)
            titles.append(f"Sample {i}")

    # ---------- plot ----------
    plot_uv_grid_jcp(x_true_list, x_pred_list, args.out,
                     sample_titles=titles,
                     field_cmap="RdBu_r", err_cmap="magma", serif=args.serif)

if __name__ == "__main__":
    main()
