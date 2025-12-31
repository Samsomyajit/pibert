#!/usr/bin/env python3
"""
Convert plasma_normal_data (unstructured HDF5) into regular-grid NPZ
compatible with runner.py (input/target tensors).

Default: target = T, inputs = [ne, te, v] + all num_of_density species + 7 conditions.
Conditions are broadcast as extra channels over the grid.
Grid: default 64x64 in [0,1]x[0,1] using nearest interpolation.
"""
import argparse, os, glob
import numpy as np
import h5py
from tqdm import tqdm

try:
    from scipy.interpolate import griddata
    _has_griddata = True
except Exception:
    _has_griddata = False


def interp_to_grid(xy, values, H=64, W=64, method="nearest"):
    """Interpolate scattered data (N,) to regular grid (H,W)."""
    xs = np.linspace(0, 1, W)
    ys = np.linspace(0, 1, H)
    X, Y = np.meshgrid(xs, ys)
    pts = xy
    if _has_griddata:
        grid = griddata(pts, values, (X, Y), method=method, fill_value=np.nan)
        # fallback to nearest for NaNs
        if np.isnan(grid).any() and method != "nearest":
            grid_near = griddata(pts, values, (X, Y), method="nearest")
            nan_mask = np.isnan(grid)
            grid[nan_mask] = grid_near[nan_mask]
    else:
        # simple nearest using argmin (slower)
        grid = np.empty((H, W), dtype=values.dtype)
        for i in range(H):
            for j in range(W):
                dx = pts[:, 0] - X[i, j]
                dy = pts[:, 1] - Y[i, j]
                idx = np.argmin(dx * dx + dy * dy)
                grid[i, j] = values[idx]
    return grid.astype(np.float32)


def load_one(path, H=64, W=64, method="nearest"):
    with h5py.File(path, "r") as f:
        cond = f["conditions"][()]  # (7,)
        xy = f["node"]["node_pos"][()]  # (N,2)
        state = f["state"]
        T = state["T"][0]          # (82, N)
        ne = state["ne"][0]
        te = state["te"][0]
        v  = state["v"][0]
        t_axis = state["t"][()]    # (82,)
        dens = state["num_of_density"]
        species = {k: dens[k][0] for k in dens.keys()}

    # normalize xy to [0,1] if not already
    xy_min = xy.min(axis=0)
    xy_max = xy.max(axis=0)
    xy_n = (xy - xy_min) / (xy_max - xy_min + 1e-8)

    X_list = []
    Y_list = []
    for ti in range(T.shape[0]):
        chans = []
        # broadcast conditions
        for c in cond:
            chans.append(np.full((H, W), c, dtype=np.float32))
        # scalar fields
        for arr in (ne[ti], te[ti], v[ti]):
            chans.append(interp_to_grid(xy_n, arr, H=H, W=W, method=method))
        # densities
        for k in sorted(species.keys()):
            chans.append(interp_to_grid(xy_n, species[k][ti], H=H, W=W, method=method))
        x_tensor = np.stack(chans, axis=0)  # (Cin,H,W)
        y_tensor = interp_to_grid(xy_n, T[ti], H=H, W=W, method=method)[None, ...]  # target T
        X_list.append(x_tensor)
        Y_list.append(y_tensor)
    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    return X, Y


def process_split(split_dir, H, W, method):
    files = sorted(glob.glob(os.path.join(split_dir, "*.h5")))
    Xs = []
    Ys = []
    for p in tqdm(files, desc=f"{os.path.basename(split_dir)} files", unit="file"):
        X, Y = load_one(p, H=H, W=W, method=method)
        Xs.append(X)
        Ys.append(Y)
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    return X, Y


def main():
    ap = argparse.ArgumentParser(description="Convert plasma_normal_data to regular-grid NPZ (input/target).")
    ap.add_argument("--root", default="plasma_normal_data")
    ap.add_argument("--out-root", default="plasma_npz")
    ap.add_argument("--grid", type=int, default=64, help="Grid resolution (H=W)")
    ap.add_argument("--method", default="nearest", choices=["nearest", "linear", "cubic"], help="Interpolation")
    ap.add_argument("--max-train-samples", type=int, default=0, help="Cap total train samples (0 = all)")
    ap.add_argument("--max-val-samples", type=int, default=0, help="Cap total val samples (0 = all)")
    args = ap.parse_args()

    H = W = args.grid
    os.makedirs(args.out_root, exist_ok=True)
    for split, cap in [("train", args.max_train_samples), ("val", args.max_val_samples)]:
        split_dir = os.path.join(args.root, split)
        if not os.path.isdir(split_dir):
            print(f"[skip] missing split dir {split_dir}")
            continue
        print(f"[info] processing {split_dir} ...")
        X, Y = process_split(split_dir, H=H, W=W, method=args.method)
        if cap and cap > 0:
            X = X[:cap]
            Y = Y[:cap]
        out_path = os.path.join(args.out_root, f"{split}.npz")
        np.savez_compressed(out_path, input=X, target=Y)
        print(f"[done] {split}: saved {X.shape[0]} samples to {out_path} (Cin={X.shape[1]}, H={H}, W={W}, Cout={Y.shape[1]})")


if __name__ == "__main__":
    main()
