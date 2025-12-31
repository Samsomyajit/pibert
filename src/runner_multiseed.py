# runner_multiseed.py
import os, json, argparse, time, numpy as np, pandas as pd, torch
from torch import nn, optim
from src.data import make_loaders
from src.models import PINN, FNO2d, DeepONet2d, PINNsformer, PIBERT, PIBERT_FNO, PIBERT_PINNsformer, PIBERT_DeepONet2d
from src.metrics import mae, mse, nmse, relative_error, table_from_results

def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def param_millions(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def build_model(name, cin, cout, cfg):
    if name=="PINN": return PINN(cin, cout, **cfg.get("PINN", {}))
    if name=="FNO2d": return FNO2d(cin, cout, **cfg.get("FNO2d", {}))
    if name=="DeepONet2d": return DeepONet2d(cin, cout, **cfg.get("DeepONet2d", {}))
    if name=="PINNsformer": return PINNsformer(cin, cout, **cfg.get("PINNsformer", {}))
    if name=="PIBERT": return PIBERT(cin, cout, **cfg.get("PIBERT", {}))
    if name=="PIBERT_FNO": return PIBERT_FNO(cin, cout, **cfg.get("PIBERT_FNO", {}))
    if name=="PIBERT_PINNsformer": return PIBERT_PINNsformer(cin, cout, **cfg.get("PIBERT_PINNsformer", {}))
    if name=="PIBERT_DeepONet2d": return PIBERT_DeepONet2d(cin, cout, **cfg.get("PIBERT_DeepONet2d", {}))
    raise ValueError(f"Unknown model {name}")

def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass

def train_one(model, loader, opt, device):
    model.train()
    loss_meter = 0.0
    crit = nn.MSELoss()
    for x,y,xyt in loader:
        x,y,xyt = x.to(device), y.to(device), xyt.to(device)
        opt.zero_grad(set_to_none=True)
        pred = model(xyt, x)
        loss = crit(pred, y)
        loss.backward(); opt.step()
        loss_meter += loss.item() * x.size(0)
    return loss_meter / len(loader.dataset)

@torch.no_grad()
def eval_one(model, loader, device, unnorm, save_dir=None, save_n=8):
    model.eval()
    e_mae = 0.0; e_mse = 0.0; e_nmse = 0.0
    saved = 0
    for i, (x,y,xyt) in enumerate(loader):
        x,y,xyt = x.to(device), y.to(device), xyt.to(device)
        pred = model(xyt, x)
        yhat = pred * torch.tensor(unnorm["y_std"]).to(device) + torch.tensor(unnorm["y_mean"]).to(device)
        ygt  = y    * torch.tensor(unnorm["y_std"]).to(device) + torch.tensor(unnorm["y_mean"]).to(device)

        e_mae += mae(yhat, ygt) * x.size(0)
        e_mse += mse(yhat, ygt) * x.size(0)
        e_nmse += nmse(yhat, ygt) * x.size(0)

        if save_dir and saved < save_n:
            rel = relative_error(yhat, ygt).cpu().numpy()[0]
            np.savez(os.path.join(save_dir, f"sample_{saved:03d}.npz"),
                     gt=ygt.cpu().numpy()[0],
                     pred=yhat.cpu().numpy()[0],
                     relerr=rel)
            saved += 1
    n = len(loader.dataset)
    return e_mae/n, e_mse/n, e_nmse/n

@torch.no_grad()
def measure_latency(model, loader, device, reps=30, warmup=5):
    # take first batch and measure per-item latency (ms)
    model.eval()
    xb, yb, xytb = next(iter(loader))
    xb, xytb = xb.to(device), xytb.to(device)
    # warmup
    for _ in range(warmup):
        _ = model(xytb, xb)
    synchronize(device)
    t0 = time.time()
    for _ in range(reps):
        _ = model(xytb, xb)
    synchronize(device)
    dt = (time.time() - t0) / reps
    per_item = (dt / xb.size(0)) * 1000.0
    return per_item

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = json.load(f)

    # device pick
    device_pref = str(cfg["train"].get("device", "auto")).lower()
    if device_pref == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        if device_pref == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif device_pref == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    print(f"Using device: {device}")

    outdir_root = cfg["eval"]["outdir"]
    os.makedirs(outdir_root, exist_ok=True)

    # loaders (train stats are used for all splits in data.make_loaders)
    train_loader, val_loader, test_loader, norm, shapes = make_loaders(
        cfg["data"]["root"], fmt=cfg["data"].get("format","npz"), batch_size=cfg["train"]["batch_size"], normalize=True
    )

    seeds = cfg["train"].get("seeds", [cfg["train"]["seed"]])
    models = cfg["models"]

    all_rows = []  # aggregated across seeds
    per_seed_tables = {}

    for seed in seeds:
        set_seed(seed)
        print(f"\n=== Seed {seed} ===")
        rows = []
        for name in models:
            print(f"\n--- {name} ---")
            model = build_model(name, shapes["Cin"], shapes["Cout"], cfg.get("model_cfg", {})).to(device)
            opt = optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

            for ep in range(cfg["train"]["epochs"]):
                tr_loss = train_one(model, train_loader, opt, device)
                print(f"Epoch {ep+1}/{cfg['train']['epochs']} - train MSE: {tr_loss:.6f}")

            # save a few val predictions for visuals
            mdir = os.path.join(outdir_root, f"{name}_seed{seed}")
            os.makedirs(mdir, exist_ok=True)
            val_mae, val_mse, val_nmse = eval_one(model, val_loader, device, norm, save_dir=mdir, save_n=cfg["eval"]["save_n"])
            test_mae, test_mse, test_nmse = eval_one(model, test_loader, device, norm, save_dir=None, save_n=0)

            # latency (ms / item)
            lat_ms = measure_latency(model, val_loader, device, reps=cfg["eval"].get("latency_reps", 30))

            row = {
                "Seed": seed,
                "Model": name,
                "Params(M)": round(param_millions(model), 3),
                "Latency(ms)": round(lat_ms, 3),
                "Val_MAE": float(val_mae),
                "Val_MSE": float(val_mse),
                "Val_NMSE": float(val_nmse),
                "Test_MAE": float(test_mae),
                "Test_MSE": float(test_mse),
                "Test_NMSE": float(test_nmse),
            }
            rows.append(row)
            all_rows.append(row)
            print(row)

        df_seed = pd.DataFrame(rows)
        per_seed_tables[seed] = df_seed
        df_seed.to_csv(os.path.join(outdir_root, f"metrics_seed{seed}.csv"), index=False)

    # aggregate mean±std over seeds
    df_all = pd.DataFrame(all_rows)
    agg = df_all.groupby("Model").agg(
        Params_M_mean=("Params(M)","mean"),
        Latency_ms_mean=("Latency(ms)","mean"),
        Latency_ms_std=("Latency(ms)","std"),
        Val_MAE_mean=("Val_MAE","mean"),
        Val_MAE_std=("Val_MAE","std"),
        Val_MSE_mean=("Val_MSE","mean"),
        Val_MSE_std=("Val_MSE","std"),
        Val_NMSE_mean=("Val_NMSE","mean"),
        Val_NMSE_std=("Val_NMSE","std"),
    ).reset_index()

    # pretty printable table with MAE ×1e3
    pretty = pd.DataFrame({
        "Model": agg["Model"],
        "Params(M)": agg["Params_M_mean"].round(3),
        "Latency(ms)": (agg["Latency_ms_mean"].round(2)).astype(float).astype(str) + " ± " + (agg["Latency_ms_std"].fillna(0).round(2)).astype(str),
        "MAE(1e-3)": (1e3*agg["Val_MAE_mean"]).round(3).astype(str) + " ± " + (1e3*agg["Val_MAE_std"].fillna(0)).round(3).astype(str),
        "MSE": agg["Val_MSE_mean"].round(6).astype(str) + " ± " + agg["Val_MSE_std"].fillna(0).round(6).astype(str),
        "NMSE": agg["Val_NMSE_mean"].round(6).astype(str) + " ± " + agg["Val_NMSE_std"].fillna(0).round(6).astype(str),
    })

    print("\nAggregate over seeds (validation):")
    print(pretty.to_string(index=False))

    df_all.to_csv(os.path.join(outdir_root, "metrics_all_seeds.csv"), index=False)
    agg.to_csv(os.path.join(outdir_root, "metrics_agg_mean_std.csv"), index=False)
    pretty.to_csv(os.path.join(outdir_root, "metrics_pretty.csv"), index=False)

if __name__ == "__main__":
    main()
