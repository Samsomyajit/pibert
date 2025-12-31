# runner.py (MPS-stable, AMP off on MPS, non-finite-safe)
import os, json, argparse, time, inspect, math
import numpy as np
import pandas as pd
import torch
from torch import nn, optim

from src.data import make_loaders
from src.models import (
    PINN, FNO2d, DeepONet2d,
    PIBERT, PIBERT_FNO, PIBERT_PINNsformer, PIBERT_DeepONet2d,
    ns_physics_loss, _ddx, _ddy, _grid_spacing
)
from src.metrics import mae, mse, nmse, relative_error, table_from_results

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ------------------------- utils -------------------------
def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def param_millions(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


# ---- lightweight helpers for embedding capture -----------------
@torch.no_grad()
def _physical_scalars(y_src, xyt):
    """
    y_src: unnormalized tensor (B,2,H,W)
    Returns dict of flattened arrays for speed/vorticity/divergence.
    """
    u, v = y_src[:, 0:1], y_src[:, 1:2]
    dx, dy = _grid_spacing(xyt)
    vort = _ddx(v, dx) - _ddy(u, dy)
    div  = _ddx(u, dx) + _ddy(v, dy)
    speed = (u * u + v * v).sqrt()
    out = {
        "speed":      speed.flatten(1).cpu().numpy(),
        "vorticity":  vort.flatten(1).cpu().numpy(),
        "divergence": div.flatten(1).cpu().numpy(),
    }
    return out


def _get_embedder(model):
    """
    Minimal embedder hook reused from viz script.
    Returns embed(xyt, cond) -> (B, N, D) tokens.
    """
    name = type(model).__name__
    store = {"feat": None}
    handle = None

    def hook(_m, _inp, out):
        store["feat"] = out.detach()

    if name == "PINN":
        tanhs = [m for m in model.net if isinstance(m, torch.nn.Tanh)]
        handle = tanhs[-1].register_forward_hook(hook)
        kind = "pinn"
    elif name == "DeepONet2d":
        kind = "deeponet"
    elif name in ("PIBERT", "PIBERTNoCoords"):
        handle = model.refine.register_forward_hook(hook)
        kind = "pibert"
    elif name == "PIBERT_DeepONet2d":
        kind = "pibert_deeponet"
    elif name == "FNO2d":
        store["feat"] = None
        def _hook(_m, _inp, out):
            store["feat"] = out.detach()
        handle = model.fc1.register_forward_hook(_hook)
        kind = "fno"
    else:
        return None

    @torch.no_grad()
    def embed(xyt, cond):
        if kind == "pinn":
            store["feat"] = None
            _ = model(xyt, cond)
            z = store["feat"]
            return z.view(z.size(0), -1, z.size(-1))
        if kind == "deeponet":
            B, Cin, H, W = cond.shape
            T = xyt.view(B, -1, 3)
            t = model.trunk(T).view(B, -1, model.cout, model.basis)
            b = model.branch(cond.mean(dim=[2, 3]))
            e = t * b.view(B, 1, 1, -1)
            return e.view(B, -1, model.cout * model.basis)
        if kind == "pibert":
            store["feat"] = None
            _ = model(xyt, cond)
            feat = store["feat"]
            sk = model.skip(cond).detach() if hasattr(model, "skip") and model.skip is not None else None
            if sk is not None:
                z = torch.cat([feat, sk], dim=1)
            else:
                z = feat
            return z.permute(0, 2, 3, 1).reshape(z.size(0), -1, z.size(1))
        if kind == "pibert_deeponet":
            B, Cin, H, W = cond.shape
            x = torch.sigmoid(model.g_ff) * model.ff(cond) + torch.sigmoid(model.g_wv) * model.wv(cond)
            x = model.fuse(x)
            dn = model.deeponet
            T = xyt.view(B, -1, 3)
            t = dn.trunk(T).view(B, -1, dn.cout, dn.basis)
            b = dn.branch(x.mean(dim=[2, 3]))
            e = t * b.view(B, 1, 1, -1)
            return e.view(B, -1, dn.cout * dn.basis)
        if kind == "fno":
            store["feat"] = None
            _ = model(xyt, cond)
            z = store["feat"]
            return z.view(z.size(0), -1, z.size(1))
    embed._hook_handle = handle
    return embed

def _construct_model(model_cls, cin, cout, cfg_section):
    cfg_section = dict(cfg_section or {})
    sig = inspect.signature(model_cls.__init__)
    allowed = set(sig.parameters.keys()) - {"self", "cin", "cout"}
    kwargs = {k: v for k, v in cfg_section.items() if k in allowed}
    dropped = sorted(set(cfg_section.keys()) - set(kwargs.keys()))
    if dropped:
        print(f"[WARN] Ignoring unknown {model_cls.__name__} kwargs: {dropped}")
    return model_cls(cin, cout, **kwargs)

def build_model(name, cin, cout, cfg):
    name = str(name)
    if name == "PINN":               return _construct_model(PINN, cin, cout, cfg.get("PINN"))
    if name == "FNO2d":              return _construct_model(FNO2d, cin, cout, cfg.get("FNO2d"))
    if name == "DeepONet2d":         return _construct_model(DeepONet2d, cin, cout, cfg.get("DeepONet2d"))
    if name == "PIBERT":             return _construct_model(PIBERT, cin, cout, cfg.get("PIBERT"))
    if name == "PIBERT_FNO":         return _construct_model(PIBERT_FNO, cin, cout, cfg.get("PIBERT_FNO"))
    if name == "PIBERT_PINNsformer": return _construct_model(PIBERT_PINNsformer, cin, cout, cfg.get("PIBERT_PINNsformer"))
    if name == "PIBERT_DeepONet2d":  return _construct_model(PIBERT_DeepONet2d, cin, cout, cfg.get("PIBERT_DeepONet2d"))
    raise ValueError(f"Unknown model {name}")

def pick_device(pref="auto"):
    pref = str(pref).lower()
    if pref == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if pref == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def _amp_enabled_for(device: torch.device, use_amp: bool, dtype, *, for_eval: bool):
    if not use_amp:
        return False
    if device.type == "cuda":
        return True
    if device.type == "mps":
        # FULLY DISABLE AMP on MPS for stability
        return False
    return False

def latency_ms_for(model, x, xyt, reps=50, device=torch.device("cpu"),
                   use_amp=False, amp_dtype=torch.float16):
    if reps <= 0: return None
    model.eval()
    amp_en = _amp_enabled_for(device, use_amp, amp_dtype, for_eval=True)
    ctx = torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_en)
    with torch.inference_mode(), ctx:
        for _ in range(5):
            _ = model(xyt, x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            _ = model(xyt, x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / reps

def _image_grad_l2_loss(pred, y):
    gx_p = pred[..., :, 1:] - pred[..., :, :-1]
    gx_y = y   [..., :, 1:] - y   [..., :, :-1]
    gy_p = pred[..., 1:, :] - pred[..., :-1, :]
    gy_y = y   [..., 1:, :] - y   [..., :-1, :]
    return (gx_p - gx_y).pow(2).mean() + (gy_p - gy_y).pow(2).mean()

def _make_invalid_ecp_samples(cond, strategy="noise"):
    """Synthesize negative samples for ECP classification."""
    strategy = str(strategy or "noise").lower()
    if strategy == "noise":
        std = cond.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return torch.randn_like(cond) * std
    if strategy == "zeros":
        return torch.zeros_like(cond)
    if strategy == "shuffle":
        if cond.size(0) <= 1:
            return cond.clone()
        idx = torch.randperm(cond.size(0), device=cond.device)
        return cond[idx]
    # default to noise if unknown
    std = cond.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return torch.randn_like(cond) * std

def _parse_mpp_cfg(cfg: dict):
    """Normalize MPP config, supporting legacy train.* keys."""
    train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    mpp_cfg = dict(cfg.get("mpp", {})) if isinstance(cfg, dict) else {}
    if "lambda_mpp" in train_cfg and "lambda_mpp" not in mpp_cfg:
        mpp_cfg["lambda_mpp"] = train_cfg["lambda_mpp"]
    if "mpp_mask_ratio" in train_cfg and "mask_ratio" not in mpp_cfg:
        mpp_cfg["mask_ratio"] = train_cfg["mpp_mask_ratio"]
    if "lambda_mpp_div" in train_cfg and "lambda_div" not in mpp_cfg:
        mpp_cfg["lambda_div"] = train_cfg["lambda_mpp_div"]
    lam = float(mpp_cfg.get("lambda_mpp", 0.0))
    mpp_cfg["lambda_mpp"] = lam
    mpp_cfg.setdefault("mask_ratio", 0.15)
    mpp_cfg.setdefault("lambda_div", 0.0)
    mpp_cfg.setdefault("enabled", lam > 0)
    return mpp_cfg

def _parse_ecp_cfg(cfg: dict):
    """Normalize ECP config, supporting legacy train.* keys."""
    train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    ecp_cfg = dict(cfg.get("ecp", {})) if isinstance(cfg, dict) else {}
    if "lambda_ecp" in train_cfg and "lambda_ecp" not in ecp_cfg:
        ecp_cfg["lambda_ecp"] = train_cfg["lambda_ecp"]
    lam = float(ecp_cfg.get("lambda_ecp", 0.0))
    ecp_cfg["lambda_ecp"] = lam
    ecp_cfg.setdefault("invalid_strategy", "noise")
    ecp_cfg.setdefault("enabled", lam > 0)
    return ecp_cfg

# ------------------------- EMA -------------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.collected = {}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model):
        self.collected = {}
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            self.collected[n] = p.detach().clone()
            p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model):
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            p.data.copy_(self.collected[n])
        self.collected = {}

# ------------------------- schedulers -------------------------
class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=1e-5, last_epoch=-1):
        self.warmup_epochs = int(max(0, warmup_epochs))
        self.total_epochs  = int(total_epochs)
        self.eta_min       = float(eta_min)
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        e = self.last_epoch + 1
        lrs = []
        for base_lr, _ in zip(self.base_lrs, self.optimizer.param_groups):
            if self.warmup_epochs > 0 and e <= self.warmup_epochs:
                scale = e / float(self.warmup_epochs)
                lrs.append(base_lr * scale)
            else:
                t = max(0, e - self.warmup_epochs)
                T = max(1, self.total_epochs - self.warmup_epochs)
                cos = 0.5 * (1 + math.cos(math.pi * t / T))
                lrs.append(self.eta_min + (base_lr - self.eta_min) * cos)
        return lrs

# ------------------------- train / eval -------------------------
def _all_grads_finite(model):
    for p in model.parameters():
        if p.grad is None: continue
        if not torch.isfinite(p.grad).all():
            return False
    return True

def train_one(
    model, loader, opt, device, unnorm,
    lambda_phys=0.0, epoch=0, phys_warmup=10, clip_norm=1.0,
    grad_loss_w=0.0, ema: EMA | None = None,
    use_amp=False, amp_dtype=torch.float16, scaler=None,
    micro_batch_size=0, mpp_cfg=None, ecp_cfg=None
):
    model.train()
    crit = nn.MSELoss()

    tot = dat = phy = mpp = ecp = 0.0
    div_accum = 0.0
    n = 0

    lam_phys = float(lambda_phys) * min(1.0, float(epoch + 1) / max(1, int(phys_warmup)))

    mpp_cfg = mpp_cfg or {}
    ecp_cfg = ecp_cfg or {}
    lam_mpp = float(mpp_cfg.get("lambda_mpp", 0.0)) if mpp_cfg else 0.0
    mpp_enabled = bool(mpp_cfg.get("enabled", False)) and lam_mpp > 0.0
    mpp_mask_ratio = float(mpp_cfg.get("mask_ratio", 0.15))
    mpp_lambda_div = float(mpp_cfg.get("lambda_div", 0.0))
    lam_mpp = lam_mpp if mpp_enabled else 0.0

    lam_ecp = float(ecp_cfg.get("lambda_ecp", 0.0)) if ecp_cfg else 0.0
    ecp_enabled = bool(ecp_cfg.get("enabled", False)) and lam_ecp > 0.0
    ecp_strategy = str(ecp_cfg.get("invalid_strategy", "noise"))
    lam_ecp = lam_ecp if ecp_enabled else 0.0

    y_mean = torch.tensor(unnorm["y_mean"], device=device, dtype=torch.float32)
    y_std  = torch.tensor(unnorm["y_std"],  device=device, dtype=torch.float32)

    amp_en_train = _amp_enabled_for(device, use_amp, amp_dtype, for_eval=False)

    for x, y, xyt in loader:
        x, y, xyt = x.to(device), y.to(device), xyt.to(device)
        bs = x.size(0)
        mb = int(micro_batch_size) if micro_batch_size else bs
        mb = max(1, min(mb, bs))
        steps = math.ceil(bs / mb)

        opt.zero_grad(set_to_none=True)
        for i in range(0, bs, mb):
            x_mb, y_mb, xyt_mb = x[i:i+mb], y[i:i+mb], xyt[i:i+mb]

            ctx = torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_en_train)
            mpp_loss = torch.zeros((), device=device, dtype=torch.float32)
            ecp_loss = torch.zeros((), device=device, dtype=torch.float32)
            with ctx:
                pred = model(xyt_mb, x_mb)
                if not torch.isfinite(pred).all():
                    continue
                data_loss = crit(pred, y_mb)
                if not torch.isfinite(data_loss):
                    continue
                if grad_loss_w and grad_loss_w > 0:
                    data_loss = data_loss + float(grad_loss_w) * _image_grad_l2_loss(pred, y_mb)
                if mpp_enabled and hasattr(model, "mpp_loss"):
                    mpp_loss = model.mpp_loss(
                        xyt_mb, x_mb, mask_ratio=mpp_mask_ratio, mask=None, lambda_div=mpp_lambda_div
                    )
                if ecp_enabled and hasattr(model, "ecp_loss"):
                    invalid = _make_invalid_ecp_samples(x_mb, strategy=ecp_strategy)
                    x_ecp = torch.cat([x_mb, invalid], dim=0)
                    xyt_ecp = torch.cat([xyt_mb, xyt_mb], dim=0)
                    labels = torch.cat([
                        torch.ones(x_mb.size(0), device=device, dtype=x_mb.dtype),
                        torch.zeros(x_mb.size(0), device=device, dtype=x_mb.dtype)
                    ], dim=0)
                    ecp_loss = model.ecp_loss(xyt_ecp, x_ecp, labels)

            phys_loss = torch.zeros((), device=device, dtype=torch.float32)
            div_metric = torch.zeros((), device=device, dtype=torch.float32)
            if mpp_enabled and not torch.isfinite(mpp_loss):
                mpp_loss = torch.zeros_like(mpp_loss)
            if ecp_enabled and not torch.isfinite(ecp_loss):
                ecp_loss = torch.zeros_like(ecp_loss)
            if lam_phys > 0.0:
                pred_phys = (pred * y_std + y_mean).float()
                xyt_f = xyt_mb.float()
                out = ns_physics_loss(pred_phys, xyt_f,
                                      nu=getattr(model, "nu", 1e-3),
                                      w_div=getattr(model, "w_div", 1.0),
                                      w_vort=getattr(model, "w_vort", 1.0))
                if isinstance(out, (tuple, list)) and len(out) >= 2:
                    phys_loss = out[0]; div_metric = out[1]
                else:
                    phys_loss = out
                if not torch.isfinite(phys_loss):
                    phys_loss = torch.zeros_like(phys_loss)

            loss = (data_loss + lam_phys * phys_loss + lam_mpp * mpp_loss + lam_ecp * ecp_loss) / steps

            # backward
            loss.backward()

            # guard: if any grad is non-finite, drop this micro-batch
            if not _all_grads_finite(model):
                opt.zero_grad(set_to_none=True)
                continue

            dat += data_loss.detach().float().item() * x_mb.size(0)
            phy += phys_loss.detach().float().item() * x_mb.size(0)
            mpp += mpp_loss.detach().float().item() * x_mb.size(0)
            ecp += ecp_loss.detach().float().item() * x_mb.size(0)
            div_accum += div_metric.float().mean().item() * x_mb.size(0)

        if clip_norm is not None and clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        opt.step()
        if ema is not None:
            ema.update(model)

        n += bs
        tot = dat + lam_phys * phy + lam_mpp * mpp + lam_ecp * ecp

    return (
        tot / max(n, 1),
        dat / max(n, 1),
        phy / max(n, 1),
        mpp / max(n, 1),
        ecp / max(n, 1),
        div_accum / max(n, 1)
    )

@torch.no_grad()
def _divergence_mean_abs(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    u = u.float(); v = v.float()
    du_dy, du_dx = torch.gradient(u, dim=(1, 2))
    dv_dy, dv_dx = torch.gradient(v, dim=(1, 2))
    div = du_dx + dv_dy
    return div.abs().mean()

@torch.no_grad()
def eval_metrics(model, loader, device, unnorm, use_amp=False, amp_dtype=torch.float16,
                 max_batches=0):
    model.eval()
    amp_en_eval = _amp_enabled_for(device, use_amp, amp_dtype, for_eval=True)
    ctx = torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_en_eval)

    e_mae = e_mse = e_nmse = 0.0
    rel_l2_num = rel_l2_den = 0.0

    y_mean = torch.tensor(unnorm["y_mean"], device=device, dtype=torch.float32)
    y_std  = torch.tensor(unnorm["y_std"],  device=device, dtype=torch.float32)

    first_batch_div = None
    n = 0
    seen = 0
    with ctx:
        for it, (x, y, xyt) in enumerate(loader):
            x, y, xyt = x.to(device), y.to(device), xyt.to(device)
            pred = model(xyt, x)
            if not torch.isfinite(pred).all():
                continue

            pred = pred.float(); y = y.float()
            yhat = pred * y_std + y_mean
            ygt  = y    * y_std + y_mean

            if not torch.isfinite(yhat).all() or not torch.isfinite(ygt).all():
                continue

            bs = x.size(0)
            e_mae  += mae(yhat, ygt) * bs
            e_mse  += mse(yhat, ygt) * bs
            e_nmse += nmse(yhat, ygt) * bs

            num = torch.linalg.vector_norm((yhat - ygt).reshape(bs, -1), ord=2, dim=1).sum().item()
            den = torch.linalg.vector_norm(ygt.reshape(bs, -1), ord=2, dim=1).sum().item()
            rel_l2_num += num
            rel_l2_den += max(den, 1e-12)

            if first_batch_div is None and yhat.shape[1] >= 2:
                u = yhat[:, 0]; v = yhat[:, 1]
                first_batch_div = _divergence_mean_abs(u, v).item()

            n += bs
            seen += 1
            if max_batches and seen >= max_batches:
                break

    rel_l2 = rel_l2_num / max(rel_l2_den, 1e-12)
    div_proxy = float(first_batch_div if first_batch_div is not None else 0.0)
    return e_mae / max(n,1), e_mse / max(n,1), e_nmse / max(n,1), rel_l2, div_proxy


@torch.no_grad()
def capture_embeddings(model, loader, device, unnorm, max_batches=2, pool="meanvar"):
    """
    Collect a small slice of embeddings + physics scalars for later PCA/TSNE.
    """
    emb_fn = _get_embedder(model)
    if emb_fn is None:
        return None
    y_mean = torch.tensor(unnorm["y_mean"], device=device, dtype=torch.float32)
    y_std  = torch.tensor(unnorm["y_std"],  device=device, dtype=torch.float32)
    zs, phys = [], {"speed": [], "vorticity": [], "divergence": []}
    taken = 0
    for x, y, xyt in loader:
        x, y, xyt = x.to(device), y.to(device), xyt.to(device)
        z = emb_fn(xyt, x)[0].cpu().numpy()  # (N,D)
        if pool == "mean":
            z = z.mean(axis=0, keepdims=True)
        elif pool == "meanvar":
            mu = z.mean(axis=0, keepdims=True)
            std = z.std(axis=0, keepdims=True)
            z = np.concatenate([mu, std], axis=1)
        zs.append(z)

        y_src = y.float() * y_std + y_mean
        scal = _physical_scalars(y_src, xyt)
        for k in phys:
            if pool in ("mean", "meanvar"):
                phys[k].append(scal[k][0].mean(keepdims=True))
            else:
                phys[k].append(scal[k][0])

        taken += 1
        if taken >= max_batches:
            break

    Z = np.concatenate(zs, axis=0)
    phys = {k: np.concatenate(v, axis=0) for k, v in phys.items()}
    return Z, phys

@torch.no_grad()
def eval_one(model, loader, device, unnorm, save_dir=None, save_n=8,
             use_amp=False, amp_dtype=torch.float16, max_batches=0):
    model.eval()
    amp_en_eval = _amp_enabled_for(device, use_amp, amp_dtype, for_eval=True)
    ctx = torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_en_eval)

    e_mae = e_mse = e_nmse = 0.0
    saved = 0
    n = 0

    y_mean = torch.tensor(unnorm["y_mean"], device=device, dtype=torch.float32)
    y_std  = torch.tensor(unnorm["y_std"],  device=device, dtype=torch.float32)

    seen = 0
    with ctx:
        for i, (x, y, xyt) in enumerate(loader):
            x, y, xyt = x.to(device), y.to(device), xyt.to(device)
            pred = model(xyt, x)
            if not torch.isfinite(pred).all():
                continue

            pred = pred.float(); y = y.float()
            yhat = pred * y_std + y_mean
            ygt  = y    * y_std + y_mean
            if not torch.isfinite(yhat).all() or not torch.isfinite(ygt).all():
                continue

            bs = x.size(0)
            e_mae += mae(yhat, ygt) * bs
            e_mse += mse(yhat, ygt) * bs
            e_nmse += nmse(yhat, ygt) * bs
            n += bs

            if save_dir and saved < save_n:
                rel = relative_error(yhat, ygt).cpu().numpy()[0]
                import numpy as np
                np.savez(
                    os.path.join(save_dir, f"sample_{saved:03d}.npz"),
                    gt=ygt.cpu().numpy()[0],
                    pred=yhat.cpu().numpy()[0],
                    relerr=rel,
                )
                saved += 1

            seen += 1
            if max_batches and seen >= max_batches:
                break

    return e_mae / max(n,1), e_mse / max(n,1), e_nmse / max(n,1)

# ------------------------- aggregation (unchanged) -------------------------
def bootstrap_ci(values, reducer="mean", n_boot=1000, ci=0.95, rng=None):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0: return (np.nan, np.nan, np.nan)
    if arr.size == 1: return (arr[0], arr[0], arr[0])
    center_fn = np.median if reducer == "median" else np.mean
    rng = np.random.default_rng(None if rng is None else rng)
    stats = []
    n = arr.size
    for _ in range(int(n_boot)):
        samp = arr[rng.integers(0, n, size=n)]
        stats.append(center_fn(samp))
    stats = np.sort(stats)
    lo = stats[int(((1 - ci) / 2) * (n_boot - 1))]
    hi = stats[int((1 - (1 - ci) / 2) * (n_boot - 1))]
    return float(center_fn(arr)), float(lo), float(hi)

def aggregate_over_seeds(rows, include_latency=True):
    df = pd.DataFrame(rows)
    out_rows = []
    for model, g in df.groupby("Model"):
        one = {"Model": model, "Seeds": int(g["Seed"].nunique()), "Param(M)": round(float(g["Param(M)"].iloc[0]), 3)}
        for col in ["MAE(1e-3)", "MSE", "NMSE"] + (["Latency(ms)"] if include_latency and "Latency(ms)" in g else []):
            vals = g[col].dropna().values.astype(float)
            m_mean, lo_mean, hi_mean = bootstrap_ci(vals, "mean")
            m_med,  lo_med,  hi_med  = bootstrap_ci(vals, "median")
            base = col
            one[f"{base}_mean"]    = round(m_mean, 6)
            one[f"{base}_mean_lo"] = round(lo_mean, 6)
            one[f"{base}_mean_hi"] = round(hi_mean, 6)
            one[f"{base}_median"]    = round(m_med, 6)
            one[f"{base}_median_lo"] = round(lo_med, 6)
            one[f"{base}_median_hi"] = round(hi_med, 6)
        out_rows.append(one)
    return pd.DataFrame(out_rows).sort_values("Model").reset_index(drop=True)

# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    mpp_cfg_base = _parse_mpp_cfg(cfg)
    ecp_cfg_base = _parse_ecp_cfg(cfg)

    device = pick_device(cfg["train"].get("device", "auto"))
    print(f"Using device: {device}")

    amp_cfg     = cfg.get("amp", {})
    use_amp     = bool(amp_cfg.get("enabled", True))
    amp_dtype_s = str(amp_cfg.get("dtype", "fp16")).lower()
    amp_dtype   = torch.float16 if amp_dtype_s in ("fp16","float16","half") else torch.bfloat16

    # Force AMP off on MPS for stability
    if device.type == "mps":
        use_amp = False
        print("[NOTE] AMP disabled on MPS for stability.")

    base_seed = int(cfg["train"].get("seed", 42))
    seeds_cfg = cfg["train"].get("seeds", None)
    if isinstance(seeds_cfg, list) and len(seeds_cfg) > 0:
        seeds = [int(s) for s in seeds_cfg]
    else:
        repeats = int(cfg["train"].get("seed_repeats", cfg["train"].get("repeats", 1)))
        seeds = [base_seed + i for i in range(max(1, repeats))]
    print(f"[Seeds] {seeds}")

    train_loader, val_loader, test_loader, norm, shapes = make_loaders(
        cfg["data"]["root"],
        fmt=cfg["data"].get("format", "npz"),
        batch_size=cfg["train"]["batch_size"],
        normalize=True,
    )
    print(f"[Shapes] Cin={shapes['Cin']} Cout={shapes['Cout']}")

    outdir = cfg["eval"]["outdir"]
    os.makedirs(outdir, exist_ok=True)

    lambda_phys = float(cfg["train"].get("lambda_phys", 0.0))
    phys_warmup = int(cfg["train"].get("phys_warmup", 10))
    clip_norm   = float(cfg["train"].get("clip_norm", cfg["train"].get("clip_grad_norm", 1.0)))
    reps        = int(cfg["eval"].get("latency_reps", 0))
    grad_loss_w = float(cfg["train"].get("grad_loss_w", 0.0))
    micro_bsz   = int(cfg["train"].get("micro_batch_size", 0) or (2 if device.type=="mps" else 0))

    eval_every      = int(cfg["eval"].get("eval_every", 5))
    max_val_batches = int(cfg["eval"].get("max_val_batches", 0))
    log_embeds      = bool(cfg["eval"].get("log_embeddings", False))
    embed_pool      = cfg["eval"].get("embed_pool", "meanvar")
    embed_batches   = int(cfg["eval"].get("embed_batches", 2))

    lr          = float(cfg["train"].get("lr", 1e-3))
    weight_decay= float(cfg["train"].get("weight_decay", 1e-4))
    warmup_ep   = int(cfg["train"].get("warmup_epochs", 0))
    eta_min     = float(cfg["train"].get("eta_min", 1e-5))

    ema_cfg     = cfg.get("ema", {})
    ema_enabled = bool(ema_cfg.get("enabled", False))
    ema_decay   = float(ema_cfg.get("decay", 0.999))

    per_seed_rows = []

    for seed in seeds:
        print(f"\n================= SEED {seed} =================")
        set_seed(seed)

        for name in cfg["models"]:
            print(f"\n=== {name} ===")
            model = build_model(name, shapes["Cin"], shapes["Cout"], cfg.get("model_cfg", {})).to(device)

            opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99))
            sch = WarmupCosineLR(opt, warmup_epochs=warmup_ep, total_epochs=int(cfg["train"]["epochs"]), eta_min=eta_min)
            ema = EMA(model, decay=ema_decay) if ema_enabled else None

            mpp_cfg = dict(mpp_cfg_base)
            ecp_cfg = dict(ecp_cfg_base)
            if mpp_cfg.get("enabled", False) and not hasattr(model, "mpp_loss"):
                print(f"[WARN] {name} lacks mpp_loss; disabling MPP loss.")
                mpp_cfg["enabled"] = False
            if ecp_cfg.get("enabled", False) and not hasattr(model, "ecp_loss"):
                print(f"[WARN] {name} lacks ecp_loss; disabling ECP loss.")
                ecp_cfg["enabled"] = False
            lambda_mpp_val = float(mpp_cfg.get("lambda_mpp", 0.0)) if mpp_cfg.get("enabled", False) else 0.0
            lambda_ecp_val = float(ecp_cfg.get("lambda_ecp", 0.0)) if ecp_cfg.get("enabled", False) else 0.0

            history = []
            epochs = int(cfg["train"]["epochs"])
            for ep in range(epochs):
                tr_tot, tr_dat, tr_phy, tr_mpp, tr_ecp, tr_div = train_one(
                    model, train_loader, opt, device, norm,
                    lambda_phys=lambda_phys, epoch=ep, phys_warmup=phys_warmup,
                    clip_norm=clip_norm, grad_loss_w=grad_loss_w, ema=ema,
                    use_amp=use_amp, amp_dtype=amp_dtype, scaler=None,
                    micro_batch_size=micro_bsz,
                    mpp_cfg=mpp_cfg, ecp_cfg=ecp_cfg
                )

                if (ep % max(1, eval_every) == 0) or (ep == epochs - 1):
                    if ema is not None: ema.apply_to(model)
                    va_mae, va_mse, va_nmse, va_rel_l2, va_div = eval_metrics(
                        model, val_loader, device, norm,
                        use_amp=use_amp, amp_dtype=amp_dtype, max_batches=max_val_batches
                    )
                    if ema is not None: ema.restore(model)
                else:
                    va_mae = va_mse = va_nmse = va_rel_l2 = va_div = float("nan")

                cur_lr = opt.param_groups[0]["lr"]
                history.append({
                    "epoch": ep + 1,
                    "lr": cur_lr,
                    "train_total": tr_tot,
                    "train_data":  tr_dat,
                    "train_phys":  tr_phy,
                    "train_mpp":   tr_mpp,
                    "train_mpp_w": lambda_mpp_val * tr_mpp,
                    "train_ecp":   tr_ecp,
                    "train_ecp_w": lambda_ecp_val * tr_ecp,
                    "train_div":   tr_div,
                    "val_mae":     va_mae,
                    "val_mse":     va_mse,
                    "val_nmse":    va_nmse,
                    "val_rel_l2":  va_rel_l2,
                    "val_div":     va_div,
                })

                if (ep + 1) % max(10, eval_every) == 0 or ep == 0 or ep == epochs - 1:
                    print(f"Epoch {ep+1}/{epochs} | lr {cur_lr:.2e} | "
                          f"train: total {tr_tot:.6f} data {tr_dat:.6f} phys {tr_phy:.6f} mpp {tr_mpp:.6f} ecp {tr_ecp:.6f} div {tr_div:.3e} | "
                          f"val: relL2 {va_rel_l2:.3e} nmse {va_nmse:.3e} div {va_div:.3e}")

                # optional embedding capture on val split
                if log_embeds and ((ep + 1) % max(1, eval_every) == 0 or ep == epochs - 1):
                    emb = capture_embeddings(
                        model, val_loader, device, norm,
                        max_batches=max(1, embed_batches),
                        pool=embed_pool
                    )
                    if emb is not None:
                        Z, phys = emb
                        mdir = os.path.join(outdir, f"{name}_seed{seed}")
                        os.makedirs(mdir, exist_ok=True)
                        np.savez_compressed(
                            os.path.join(mdir, f"embeds_epoch{ep+1:04d}.npz"),
                            Z=Z, speed=phys["speed"], vorticity=phys["vorticity"],
                            divergence=phys["divergence"],
                            epoch=ep+1, split="val", pool=embed_pool
                        )

                sch.step()

                if device.type == "mps":
                    try: torch.mps.empty_cache()
                    except Exception: pass

            # save history CSV
            mdir = os.path.join(outdir, f"{name}_seed{seed}")
            os.makedirs(mdir, exist_ok=True)
            pd.DataFrame(history).to_csv(os.path.join(mdir, "history.csv"), index=False)

            # latency (optional)
            lat = None
            if reps > 0:
                x_s, _, xyt_s = next(iter(val_loader))
                x_s, xyt_s = x_s.to(device), xyt_s.to(device)
                if ema is not None: ema.apply_to(model)
                lat = latency_ms_for(model, x_s, xyt_s, reps=reps, device=device,
                                     use_amp=use_amp, amp_dtype=amp_dtype)
                if ema is not None: ema.restore(model)

            # eval on val/test + save panels (EMA weights)
            if ema is not None: ema.apply_to(model)
            val_mae, val_mse, val_nmse = eval_one(
                model, val_loader, device, norm,
                save_dir=mdir, save_n=cfg["eval"]["save_n"],
                use_amp=use_amp, amp_dtype=amp_dtype, max_batches=max_val_batches
            )
            test_mae, test_mse, test_nmse = eval_one(
                model, test_loader, device, norm,
                save_dir=None, save_n=0, use_amp=use_amp, amp_dtype=amp_dtype, max_batches=max_val_batches
            )
            if ema is not None: ema.restore(model)

            # save checkpoint (last weights)
            ckpt_path = os.path.join(mdir, "last.pt")
            torch.save({"model": model.state_dict()}, ckpt_path)
            print(f"[save] {ckpt_path}")

            row = {
                "Model": name,
                "Seed": int(seed),
                "Param(M)": round(param_millions(model), 3),
                "MAE(1e-3)": round(1e3 * val_mae, 6),
                "MSE": round(val_mse, 6),
                "NMSE": round(val_nmse, 6),
            }
            if lat is not None:
                row["Latency(ms)"] = round(lat, 6)

            per_seed_rows.append(row)
            print(row)

    per_seed_df = pd.DataFrame(per_seed_rows).sort_values(["Model", "Seed"]).reset_index(drop=True)
    per_seed_df.to_csv(os.path.join(outdir, "metrics_per_seed.csv"), index=False)

    agg_df = aggregate_over_seeds(per_seed_rows, include_latency=("Latency(ms)" in per_seed_df.columns))
    agg_df.to_csv(os.path.join(outdir, "metrics.csv"), index=False)

    view_cols = ["Model", "Seeds", "Param(M)",
                 "MAE(1e-3)_median", "MAE(1e-3)_median_lo", "MAE(1e-3)_median_hi",
                 "MSE_median", "MSE_median_lo", "MSE_median_hi",
                 "NMSE_median", "NMSE_median_lo", "NMSE_median_hi"]
    if "Latency(ms)_median" in agg_df.columns:
        view_cols += ["Latency(ms)_median", "Latency(ms)_median_lo", "Latency(ms)_median_hi"]

    print("\nValidation results aggregated across seeds (median with 95% CI):")
    print(agg_df[view_cols].to_string(index=False))
    print(f"\nSaved per-seed to {os.path.join(outdir, 'metrics_per_seed.csv')}")
    print(f"Saved aggregated metrics with bootstrap CIs to {os.path.join(outdir, 'metrics.csv')}")

if __name__ == "__main__":
    main()
