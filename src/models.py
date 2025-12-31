# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

__all__ = [
    "count_parameters", "ns_physics_loss",
    "PINN", "FNO2d", "DeepONet2d", "PINNsformer",
    "PIBERT", "PIBERT_FNO", "PIBERT_PINNsformer", "PIBERT_DeepONet2d",
]

# -------------------------------
# Utilities
# -------------------------------
def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

class WaveAct(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.ones(1))
        self.w2 = nn.Parameter(torch.ones(1))
    def forward(self, x):
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)

# ============================================================
# Finite differences (autograd-friendly)
# f shape: (B, C, H, W) ; x ≡ last axis (W), y ≡ second-last axis (H)
# ============================================================
def _ddx(f, dx):
    left   = (f[..., :, 1]   - f[..., :, 0])   / dx
    right  = (f[..., :, -1]  - f[..., :, -2])  / dx
    center = (f[..., :, 2:]  - f[..., :, :-2]) / (2.0 * dx)
    out = torch.zeros_like(f)
    out[..., :, 0]    = left
    out[..., :, 1:-1] = center
    out[..., :, -1]   = right
    return out

def _ddy(f, dy):
    bottom = (f[..., 1, :]   - f[..., 0, :])   / dy
    top    = (f[..., -1, :]  - f[..., -2, :])  / dy
    center = (f[..., 2:, :]  - f[..., :-2, :]) / (2.0 * dy)
    out = torch.zeros_like(f)
    out[..., 0, :]    = bottom
    out[..., 1:-1, :] = center
    out[..., -1, :]   = top
    return out

def _laplacian(f, dx, dy):
    d2 = torch.zeros_like(f)
    d2[..., :, 1:-1] += (f[..., :, 2:] - 2.0 * f[..., :, 1:-1] + f[..., :, :-2]) / (dx * dx)
    d2[..., 1:-1, :] += (f[..., 2:, :] - 2.0 * f[..., 1:-1, :] + f[..., :-2, :]) / (dy * dy)
    d2[..., :, 0]    += (f[..., :, 1]  - 2.0 * f[..., :, 0]  + f[..., :, 1])  / (dx * dx)
    d2[..., :, -1]   += (f[..., :, -2] - 2.0 * f[..., :, -1] + f[..., :, -2]) / (dx * dx)
    d2[..., 0, :]    += (f[..., 1, :]  - 2.0 * f[..., 0, :]  + f[..., 1, :])  / (dy * dy)
    d2[..., -1, :]   += (f[..., -2, :] - 2.0 * f[..., -1, :] + f[..., -2, :]) / (dy * dy)
    return d2

def _grid_spacing(xyt):
    x = xyt[..., 0]; y = xyt[..., 1]
    dx = (x[:, :, 1:] - x[:, :, :-1]).abs().mean().clamp_min(1e-6)
    dy = (y[:, 1:, :] - y[:, :-1, :]).abs().mean().clamp_min(1e-6)
    return dx, dy

# ============================================================
# Physics loss (steady incompressible 2D in vorticity form)
# - If pred has <2 channels (e.g., pressure-only), return zeros.
# ============================================================
def ns_physics_loss(pred, xyt, nu=1e-3, w_div=1.0, w_vort=1.0):
    if pred.shape[1] < 2:
        zero = torch.zeros((), device=pred.device, dtype=torch.float32)
        return zero, zero, zero

    u = pred[:, 0:1, ...]
    v = pred[:, 1:2, ...]
    dx, dy = _grid_spacing(xyt)

    du_dx = _ddx(u, dx)
    dv_dy = _ddy(v, dy)
    r_div = du_dx + dv_dy

    dv_dx = _ddx(v, dx)
    du_dy = _ddy(u, dy)
    w = dv_dx - du_dy
    dw_dx = _ddx(w, dx)
    dw_dy = _ddy(w, dy)
    lap_w = _laplacian(w, dx, dy)
    r_vort = u * dw_dx + v * dw_dy - nu * lap_w

    loss_div  = (r_div  ** 2).mean()
    loss_vort = (r_vort ** 2).mean()
    total = w_div * loss_div + w_vort * loss_vort
    return total, loss_div.detach(), loss_vort.detach()

# =============================== MODELS ===============================

# ---------------- PINN ----------------
class PINN(nn.Module):
    """Grid MLP that predicts >=1 channels (e.g., p or [u,v])."""
    def __init__(self, cin, cout, hidden=128, layers=6, nu=1e-3, w_div=1.0, w_vort=1.0):
        super().__init__()
        assert cout >= 1
        self.nu, self.w_div, self.w_vort = nu, w_div, w_vort
        in_dim = 3 + cin
        net = []
        for i in range(layers - 1):
            net += [nn.Linear(in_dim if i == 0 else hidden, hidden), nn.Tanh()]
        net += [nn.Linear(hidden, cout)]
        self.net = nn.Sequential(*net)

    def forward(self, xyt, cond):
        feat = torch.cat([xyt, cond.permute(0, 2, 3, 1)], dim=-1)
        out = self.net(feat)
        return out.permute(0, 3, 1, 2)

    def physics_loss(self, xyt, pred):
        loss, _, _ = ns_physics_loss(pred, xyt, self.nu, self.w_div, self.w_vort)
        return loss

# ---------------- FNO2d ----------------
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.modes1, self.modes2 = modes1, modes2
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, 2))

    def compl_mul2d(self, x, w_ri):
        w = torch.view_as_complex(w_ri)
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x):
        inp_dtype = x.dtype
        if inp_dtype in (torch.float16, torch.bfloat16):
            x = x.to(torch.float32)
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out = torch.zeros(B, self.out_channels, H, W // 2 + 1, dtype=x_ft.dtype, device=x.device)
        m = min(H, self.modes1); n = min(W // 2 + 1, self.modes2)
        out[:, :, :m, :n] = self.compl_mul2d(x_ft[:, :, :m, :n], self.weights[:, :, :m, :n, :])
        y = torch.fft.irfft2(out, s=(H, W), norm="ortho")
        return y.to(inp_dtype)

class FNO2d(nn.Module):
    def __init__(self, cin, cout, width=48, modes=12, depth=4, nu=1e-3, w_div=1.0, w_vort=1.0):
        super().__init__()
        assert cout >= 1
        self.nu, self.w_div, self.w_vort = nu, w_div, w_vort
        self.fc0 = nn.Conv2d(cin + 3, width, 1)
        self.blocks = nn.ModuleList([
            nn.ModuleList([SpectralConv2d(width, width, modes, modes), nn.Conv2d(width, width, 1)])
            for _ in range(depth)
        ])
        self.act = nn.GELU()
        self.fc1 = nn.Conv2d(width, width // 2, 1)
        self.fc2 = nn.Conv2d(width // 2, cout, 1)

    def forward(self, xyt, cond):
        x = torch.cat([cond, xyt.permute(0, 3, 1, 2)], dim=1)
        x = self.fc0(x)
        for sc, pw in self.blocks:
            x = x + sc(x) + pw(x)
            x = self.act(x)
        x = self.act(self.fc1(x))
        return self.fc2(x)

    def physics_loss(self, xyt, pred):
        loss, _, _ = ns_physics_loss(pred, xyt, self.nu, self.w_div, self.w_vort)
        return loss

# ---------------- DeepONet2d ----------------
class MLP(nn.Module):
    def __init__(self, inp, hidden, out, layers, act=nn.GELU):
        super().__init__()
        net = []
        for i in range(layers - 1):
            net += [nn.Linear(inp if i == 0 else hidden, hidden), act()]
        net += [nn.Linear(hidden, out)]
        self.net = nn.Sequential(*net)
    def forward(self, x): return self.net(x)

class DeepONet2d(nn.Module):
    def __init__(self, cin, cout, branch=256, trunk=256, layers=4, basis=64,
                 nu=1e-3, w_div=1.0, w_vort=1.0):
        super().__init__()
        assert cout >= 1
        self.nu, self.w_div, self.w_vort = nu, w_div, w_vort
        self.basis = basis
        self.cout = cout
        self.branch = MLP(cin, branch, basis, layers)
        self.trunk  = MLP(3,  trunk,  basis * cout, layers)

    def forward(self, xyt, cond):
        B, Cin, H, W = cond.shape
        b = self.branch(cond.mean(dim=[2, 3]))
        T = xyt.view(B, -1, 3)
        t = self.trunk(T).view(B, T.shape[1], self.cout, self.basis)
        out = torch.einsum("bk,bhck->bhc", b, t).view(B, H, W, self.cout).permute(0, 3, 1, 2)
        return out

    def physics_loss(self, xyt, pred):
        loss, _, _ = ns_physics_loss(pred, xyt, self.nu, self.w_div, self.w_vort)
        return loss

# ---------------- PIBERT core + hybrids ----------------
class FourierEmbed(nn.Module):
    def __init__(self, cin, d, modes=16):
        super().__init__()
        self.proj = nn.Conv2d(cin, d, 1)
        self.modes = modes
        self.weight = nn.Parameter(torch.randn(d, d, modes, modes, 2) * 0.02)
    def forward(self, x):
        inp_dtype = x.dtype
        x = self.proj(x)
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.to(torch.float32)
        B, D, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        m, n = min(H, self.modes), min(W // 2 + 1, self.modes)
        w = torch.view_as_complex(self.weight[:, :, :m, :n])
        out = torch.zeros_like(x_ft)
        out[:, :, :m, :n] = torch.einsum("bimn,iomn->bomn", x_ft[:, :, :m, :n], w)
        y = torch.fft.irfft2(out, s=(H, W), norm="ortho")
        return y.to(inp_dtype)

class WaveletLikeEmbed(nn.Module):
    def __init__(self, cin, d):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(cin, d // 2, 3, padding=1), nn.GELU(),
            nn.Conv2d(d // 2, d, 3, padding=2, dilation=2), nn.GELU(),
            nn.Conv2d(d, d, 1),
        )
    def forward(self, x): return self.seq(x)

class PIBERT(nn.Module):
    """
    PIBERT for scalar or vector fields.
    Key tweaks for scalar p:
      - Directly embed coords together with the input (cin+3).
      - Disable residual skip when cout==1.
    """
    def __init__(self, cin, cout, d=128, depth=4, heads=4, mlp=512, fourier_modes=16,
                 patch=2, nu=1e-3, w_div=1.0, w_vort=1.0, attn_dropout=0.0, ff_dropout=0.0):
        super().__init__()
        assert cout >= 1
        self.nu, self.w_div, self.w_vort = nu, w_div, w_vort
        self.cin = cin
        self.use_skip = bool(cout >= 2)  # skip only for multi-channel targets

        # >>> embed input+coords jointly (crucial for scalar p)
        cin_joint = cin + 3
        self.ff = FourierEmbed(cin_joint, d, modes=fourier_modes)
        self.wv = WaveletLikeEmbed(cin_joint, d)
        self.g_ff = nn.Parameter(torch.tensor(0.5))
        self.g_wv = nn.Parameter(torch.tensor(0.5))
        self.fuse = nn.Conv2d(d, d, 1)

        self.patch_sz = int(patch)
        self.patch = nn.Conv2d(d, d, kernel_size=self.patch_sz, stride=self.patch_sz)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=mlp,
            dropout=float(attn_dropout), batch_first=True, norm_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=depth)

        self.refine = nn.Sequential(
            nn.Conv2d(d, d, 3, padding=1, groups=d), nn.GELU(),
            nn.Conv2d(d, d, 1), nn.GELU()
        )
        self.head = nn.Conv2d(d, cout, 1)
        # MPP/ECP heads
        self.mpp_head = nn.Conv2d(d, cin, 1)
        self.ecp_head = nn.Linear(d, 1)
        self.skip = nn.Conv2d(cin, cout, 1) if self.use_skip else None

    def _encode_tokens(self, cond, coords):
        """Encode masked or unmasked inputs; return tokens and patch dims."""
        B = cond.shape[0]
        inp = torch.cat([cond, coords], dim=1)

        x = torch.sigmoid(self.g_ff) * self.ff(inp) + torch.sigmoid(self.g_wv) * self.wv(inp)
        x = self.fuse(x)

        x = self.patch(x)
        Hp, Wp = x.shape[-2:]
        tok = x.permute(0, 2, 3, 1).reshape(B, Hp * Wp, -1).contiguous()

        tok_dtype = tok.dtype
        tok = tok.to(torch.float32)       # MPS/AMP stability
        tok = self.enc(tok)
        tok = tok.to(tok_dtype)
        return tok, Hp, Wp

    def _decode_tokens(self, tok, Hp, Wp, H, W):
        x_low = tok.view(tok.shape[0], Hp, Wp, -1).permute(0, 3, 1, 2).contiguous()
        x_up  = F.interpolate(x_low, size=(H, W), mode="bilinear", align_corners=False)
        x_up  = self.refine(x_up)
        return x_up

    def forward(self, xyt, cond):
        B, _, H, W = cond.shape
        coords = xyt.permute(0, 3, 1, 2)  # (B,3,H,W)
        tok, Hp, Wp = self._encode_tokens(cond, coords)
        x_up = self._decode_tokens(tok, Hp, Wp, H, W)

        out = self.head(x_up)
        if self.use_skip:
            out = out + self.skip(cond)
        return out

    def forward_with_features(self, xyt, cond):
        """Return (prediction, refined feature map, tokens)."""
        B, _, H, W = cond.shape
        coords = xyt.permute(0, 3, 1, 2)
        tok, Hp, Wp = self._encode_tokens(cond, coords)
        x_up = self._decode_tokens(tok, Hp, Wp, H, W)
        out = self.head(x_up)
        if self.use_skip:
            out = out + self.skip(cond)
        return out, x_up, tok

    def _divergence(self, uv, xyt):
        """Simple finite-difference divergence on reconstructed (u,v)."""
        if uv.shape[1] < 2:
            return uv.new_zeros([])
        u = uv[:, 0]
        v = uv[:, 1]
        x = xyt[..., 0]
        y = xyt[..., 1]
        dx = (x[:, :, 1:] - x[:, :, :-1]).mean().clamp_min(1e-6)
        dy = (y[:, 1:, :] - y[:, :-1, :]).mean().clamp_min(1e-6)
        du_dx = torch.diff(u, dim=2, append=u[:, :, -1:]) / dx
        dv_dy = torch.diff(v, dim=1, append=v[:, -1:, :]) / dy
        div = du_dx + dv_dy
        return div

    @torch.no_grad()
    def _make_mask(self, cond, ratio):
        B, _, H, W = cond.shape
        mask = (torch.rand(B, 1, H, W, device=cond.device, dtype=cond.dtype) < ratio)
        return mask

    def mpp_loss(self, xyt, cond, mask_ratio=0.15, mask=None, lambda_div=0.0):
        """
        Masked Physics Prediction loss (Eq. 3.2) + optional divergence penalty.
        mask_ratio: fraction of spatial sites to mask.
        """
        if mask is None:
            mask = self._make_mask(cond, mask_ratio)
        mask = mask.bool()

        masked = cond.clone()
        if mask.any():
            rand = torch.rand_like(mask, dtype=cond.dtype)
            zero_idx = mask & (rand < 0.8)
            noise_idx = mask & (rand >= 0.8) & (rand < 0.9)
            keep_idx = mask & (rand >= 0.9)

            if zero_idx.any():
                masked = masked * (~zero_idx).expand_as(masked)
            if noise_idx.any():
                scale = cond.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
                noise = torch.randn_like(masked) * scale
                masked = torch.where(noise_idx.expand_as(masked), noise, masked)
            # keep_idx automatically stays as cond

        pred, feats, _ = self.forward_with_features(xyt, masked)
        recon = self.mpp_head(feats)

        masked_pos = mask.expand_as(cond)
        if masked_pos.any():
            mse = F.mse_loss(recon[masked_pos], cond[masked_pos], reduction="mean")
        else:
            mse = recon.new_zeros([])

        if lambda_div > 0 and self.cin >= 2:
            div = self._divergence(recon, xyt)
            div_pen = (div ** 2).mean()
            mse = mse + lambda_div * div_pen
        return mse

    def ecp_logits(self, xyt, cond):
        B, _, H, W = cond.shape
        coords = xyt.permute(0, 3, 1, 2)
        tok, _, _ = self._encode_tokens(cond, coords)
        feats = tok.mean(dim=1)  # pooled tokens
        logits = self.ecp_head(feats).squeeze(-1)
        return logits

    def ecp_loss(self, xyt, cond, labels):
        logits = self.ecp_logits(xyt, cond)
        return F.binary_cross_entropy_with_logits(logits, labels.float())

    def physics_loss(self, xyt, pred):
        loss, _, _ = ns_physics_loss(pred, xyt, self.nu, self.w_div, self.w_vort)
        return loss

# ---- Hybrids stay general (work for scalar/vector); no special changes needed
class PIBERT_FNO(nn.Module):
    def __init__(self, cin, cout, d=128, depth=3, heads=4, mlp=384, fourier_modes=12, width=48, modes=12,
                 nu=1e-3, w_div=1.0, w_vort=1.0):
        super().__init__()
        assert cout >= 1
        self.nu, self.w_div, self.w_vort = nu, w_div, w_vort
        self.ff = FourierEmbed(cin, d, modes=fourier_modes)
        self.wv = WaveletLikeEmbed(cin, d)
        self.g_ff = nn.Parameter(torch.tensor(0.5))
        self.g_wv = nn.Parameter(torch.tensor(0.5))
        self.fuse = nn.Conv2d(d, d, 1)
        self.fno  = FNO2d(cin=d, cout=cout, width=width, modes=modes, depth=4)

    def forward(self, xyt, cond):
        x = torch.sigmoid(self.g_ff) * self.ff(cond) + torch.sigmoid(self.g_wv) * self.wv(cond)
        x = self.fuse(x)
        return self.fno(xyt, x)

    def physics_loss(self, xyt, pred):
        loss, _, _ = ns_physics_loss(pred, xyt, self.nu, self.w_div, self.w_vort)
        return loss

class PIBERT_PINNsformer(nn.Module):
    def __init__(self, cin, cout, d=128, depth=3, heads=4, mlp=384, fourier_modes=12,
                 nu=1e-3, w_div=1.0, w_vort=1.0):
        super().__init__()
        assert cout >= 1
        self.nu, self.w_div, self.w_vort = nu, w_div, w_vort
        self.ff = FourierEmbed(cin, d, modes=fourier_modes)
        self.wv = WaveletLikeEmbed(cin, d)
        self.g_ff = nn.Parameter(torch.tensor(0.5))
        self.g_wv = nn.Parameter(torch.tensor(0.5))
        self.fuse = nn.Conv2d(d, d, 1)
        self.core = PINNsformer(cin=d, cout=cout, d_model=d, heads=heads, depth=depth, mlp=mlp)

    def forward(self, xyt, cond):
        x = torch.sigmoid(self.g_ff) * self.ff(cond) + torch.sigmoid(self.g_wv) * self.wv(cond)
        x = self.fuse(x)
        return self.core(xyt, x)

    def physics_loss(self, xyt, pred):
        loss, _, _ = ns_physics_loss(pred, xyt, self.nu, self.w_div, self.w_vort)
        return loss

class PIBERT_DeepONet2d(nn.Module):
    def __init__(self, cin, cout, d=128, depth=3, heads=4, mlp=384, fourier_modes=12,
                 branch=256, layers=3, nu=1e-3, w_div=1.0, w_vort=1.0):
        super().__init__()
        assert cout >= 1
        self.nu, self.w_div, self.w_vort = nu, w_div, w_vort
        self.ff = FourierEmbed(cin, d, modes=fourier_modes)
        self.wv = WaveletLikeEmbed(cin, d)
        self.g_ff = nn.Parameter(torch.tensor(0.5))
        self.g_wv = nn.Parameter(torch.tensor(0.5))
        self.fuse = nn.Conv2d(d, d, 1)
        self.deeponet = DeepONet2d(cin=d, cout=cout, branch=branch, trunk=branch, layers=layers)

    def forward(self, xyt, cond):
        x = torch.sigmoid(self.g_ff) * self.ff(cond) + torch.sigmoid(self.g_wv) * self.wv(cond)
        x = self.fuse(x)
        return self.deeponet(xyt, x)

    def physics_loss(self, xyt, pred):
        loss, _, _ = ns_physics_loss(pred, xyt, self.nu, self.w_div, self.w_vort)
        return loss
