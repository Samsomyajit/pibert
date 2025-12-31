# export_samples.py
# Export additional prediction/GT npz samples from an existing checkpoint
# without retraining. Useful for producing more embedding points.

import os, json, argparse, torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import make_loaders
from src.runner import build_model, pick_device, eval_one
from src.models import FourierEmbed, WaveletLikeEmbed


class PIBERTNoCoords(nn.Module):
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
        self.skip = nn.Conv2d(cin, cout, 1) if self.use_skip else None

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
        x_up  = F.interpolate(x_low, size=(H, W), mode="bilinear", align_corners=False)
        x_up  = self.refine(x_up)

        out = self.head(x_up)
        if self.use_skip:
            out = out + self.skip(cond)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON config used for training (for data + model cfg)")
    ap.add_argument("--model", default="PIBERT", help="Model name (e.g., PIBERT)")
    ap.add_argument("--seed", type=int, default=42, help="Seed to pick checkpoint folder")
    ap.add_argument("--split", choices=["val", "test", "train"], default="test", help="Dataset split to export")
    ap.add_argument("--save-n", type=int, default=80, help="Number of samples to save")
    ap.add_argument("--export-root", default="results_tube_export", help="Root to write exported samples")
    ap.add_argument("--max-batches", type=int, default=0, help="Optional cap on batches (0 = all until save-n)")
    ap.add_argument("--legacy-no-coords", action="store_true",
                    help="Use legacy PIBERT (no coord concat) to load older checkpoints.")
    args = ap.parse_args()

    cfg = json.load(open(args.config, "r"))

    device = pick_device(cfg["train"].get("device", "auto"))
    amp_cfg = cfg.get("amp", {})
    use_amp = bool(amp_cfg.get("enabled", True))
    amp_dtype_s = str(amp_cfg.get("dtype", "fp16")).lower()
    amp_dtype = torch.float16 if amp_dtype_s in ("fp16", "float16", "half") else torch.bfloat16
    if device.type == "mps":
        use_amp = False  # stability

    train_loader, val_loader, test_loader, norm, shapes = make_loaders(
        cfg["data"]["root"], fmt=cfg["data"].get("format", "npz"),
        batch_size=1, normalize=True,
    )
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]

    if args.model == "PIBERT" and args.legacy_no_coords:
        model = PIBERTNoCoords(shapes["Cin"], shapes["Cout"], **(cfg.get("model_cfg", {}).get("PIBERT", {}))).to(device)
    else:
        model = build_model(args.model, shapes["Cin"], shapes["Cout"], cfg.get("model_cfg", {})).to(device)
    ckpt_root = cfg["eval"].get("outdir", "results")
    ckpt_dir = os.path.join(ckpt_root, f"{args.model}_seed{args.seed}")
    ckpt = os.path.join(ckpt_dir, "last.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location=device)["model"]
    def _filtered_load(model_obj):
        model_state = model_obj.state_dict()
        filtered = {k: v for k, v in state.items()
                    if k in model_state and model_state[k].shape == v.shape}
        missing = sorted(set(model_state.keys()) - set(filtered.keys()))
        extra = sorted(set(state.keys()) - set(filtered.keys()))
        print(f"[warn ] filtered load (missing: {len(missing)}, extra: {len(extra)})")
        model_obj.load_state_dict(filtered, strict=False)

    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        if args.model == "PIBERT":
            if args.legacy_no_coords:
                _filtered_load(model)
            else:
                print("[warn ] strict load failed; retrying legacy PIBERT (no coords)...")
                model = PIBERTNoCoords(shapes["Cin"], shapes["Cout"], **(cfg.get("model_cfg", {}).get("PIBERT", {}))).to(device)
                _filtered_load(model)
        else:
            raise e
    model.eval()

    export_dir = os.path.join(args.export_root, f"{args.model}_seed{args.seed}")
    os.makedirs(export_dir, exist_ok=True)

    print(f"[device] {device} | amp={use_amp} ({amp_dtype})")
    print(f"[load ] {ckpt}")
    print(f"[save ] {export_dir} (split={args.split}, save_n={args.save_n})")

    eval_one(
        model, loader, device, norm,
        save_dir=export_dir, save_n=args.save_n,
        use_amp=use_amp, amp_dtype=amp_dtype,
        max_batches=args.max_batches,
    )
    print("[done]")


if __name__ == "__main__":
    main()
