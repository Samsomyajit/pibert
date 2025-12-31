#!/usr/bin/env python3
"""
Export PIBERT encoder outputs (Fourier + Wavelet), fused tokens, and preds/GT
for a handful of samples. Useful to inspect hybrid spectral embeddings directly
from a trained checkpoint without rerunning training.
"""

import os, json, argparse
import numpy as np
import torch

from src.data import make_loaders
from src.runner import build_model, pick_device


def unnormalize(t, mean, std):
    return t * std + mean


def main():
    ap = argparse.ArgumentParser(description="Dump PIBERT encoder embeddings (ff/wv/tokens)")
    ap.add_argument("--config", required=True, help="Training config JSON")
    ap.add_argument("--model", default="PIBERT", help="Model name (default: PIBERT)")
    ap.add_argument("--seed", type=int, default=42, help="Seed folder to load checkpoint from")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test", help="Dataset split")
    ap.add_argument("--save-n", type=int, default=8, help="Number of samples to export")
    ap.add_argument("--export-root", default="encoder_exports", help="Root dir for outputs")
    ap.add_argument("--max-batches", type=int, default=0, help="Cap batches (0 = just save-n limit)")
    ap.add_argument("--start-idx", type=int, default=0, help="Skip this many samples before exporting")
    ap.add_argument("--legacy-no-coords", action="store_true",
                    help="Use PIBERTNoCoords to load checkpoints trained without coord concat (cin mismatch).")
    args = ap.parse_args()

    cfg = json.load(open(args.config, "r"))
    device = pick_device(cfg["train"].get("device", "auto"))
    amp_cfg = cfg.get("amp", {})
    use_amp = bool(amp_cfg.get("enabled", True))
    amp_dtype_s = str(amp_cfg.get("dtype", "fp16")).lower()
    amp_dtype = torch.float16 if amp_dtype_s in ("fp16","float16","half") else torch.bfloat16
    if device.type == "mps":
        use_amp = False  # stability

    train_loader, val_loader, test_loader, norm, shapes = make_loaders(
        cfg["data"]["root"], fmt=cfg["data"].get("format", "npz"),
        batch_size=1, normalize=True,
    )
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]

    model = build_model(args.model, shapes["Cin"], shapes["Cout"], cfg.get("model_cfg", {})).to(device)
    ckpt_root = cfg["eval"].get("outdir", "results")
    ckpt_dir = os.path.join(ckpt_root, f"{args.model}_seed{args.seed}")
    ckpt = os.path.join(ckpt_dir, "last.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location=device)["model"]
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        # fall back to filtered load if shapes mismatch (e.g., legacy no-coords checkpoints)
        model_state = model.state_dict()
        filtered = {k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape}
        missing = sorted(set(model_state.keys()) - set(filtered.keys()))
        extra = sorted(set(state.keys()) - set(filtered.keys()))
        print(f"[warn] filtered load due to mismatch ({len(missing)} missing, {len(extra)} extra): {e}")
        model.load_state_dict(filtered, strict=False)
    model.eval()

    y_mean = torch.tensor(norm["y_mean"], device=device, dtype=torch.float32)
    y_std  = torch.tensor(norm["y_std"],  device=device, dtype=torch.float32)
    x_mean = torch.tensor(norm["x_mean"], device=device, dtype=torch.float32)
    x_std  = torch.tensor(norm["x_std"],  device=device, dtype=torch.float32)

    os.makedirs(args.export_root, exist_ok=True)
    out_dir = os.path.join(args.export_root, f"{args.model}_seed{args.seed}_{args.split}")
    os.makedirs(out_dir, exist_ok=True)

    ctx = torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp)

    saved = 0
    seen = 0
    with torch.no_grad():
        for x, y, xyt in loader:
            if seen < args.start_idx:
                seen += 1
                continue
            if saved >= args.save_n:
                break

            x, y, xyt = x.to(device), y.to(device), xyt.to(device)
            coords = xyt.permute(0, 3, 1, 2)  # (B,3,H,W)
            inp = torch.cat([x, coords], dim=1)

            with ctx:
                ff = model.ff(inp)
                wv = model.wv(inp)
                fuse = torch.sigmoid(model.g_ff) * ff + torch.sigmoid(model.g_wv) * wv
                pred, feats, tok = model.forward_with_features(xyt, x)

            # upsample tokens back to spatial map for inspection
            HpWp = int(np.sqrt(tok.shape[1]))
            tok_map = tok.view(tok.shape[0], HpWp, HpWp, -1).permute(0, 3, 1, 2).contiguous()
            tok_up = torch.nn.functional.interpolate(tok_map, size=x.shape[-2:], mode="bilinear", align_corners=False)

            # unnormalize data
            x_unn = unnormalize(x, x_mean, x_std).cpu().numpy()
            y_gt  = unnormalize(y, y_mean, y_std).cpu().numpy()
            y_pr  = unnormalize(pred, y_mean, y_std).cpu().numpy()

            np.savez_compressed(
                os.path.join(out_dir, f"embed_{saved:03d}.npz"),
                input=x_unn[0],
                gt=y_gt[0],
                pred=y_pr[0],
                ff=ff.cpu().numpy()[0],
                wv=wv.cpu().numpy()[0],
                fuse=fuse.cpu().numpy()[0],
                tok=tok.cpu().numpy()[0],
                tok_up=tok_up.cpu().numpy()[0],
            )
            saved += 1
            seen += 1
            if args.max_batches and seen >= args.max_batches:
                break

    print(f"[done] saved {saved} embedding dumps to {out_dir}")


if __name__ == "__main__":
    main()
