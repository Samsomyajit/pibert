#!/usr/bin/env python3
"""
Convert a small subset of the EAGLE dataset (per-file .npz) to regular-grid NPZ
compatible with runner.py (input/target tensors).

Inputs: VX, VY, node_type
Target: PS (pressure) by default (can switch to velocity magnitude if desired)
"""
import argparse, os, glob
import numpy as np

try:
    from scipy.interpolate import griddata
    _has_griddata = True
except Exception:
    _has_griddata = False


def interp_to_grid(xy, values, H=64, W=64, method="nearest"):
    xs = np.linspace(0, 1, W)
    ys = np.linspace(0, 1, H)
    X, Y = np.meshgrid(xs, ys)
    pts = xy
    if _has_griddata:
        grid = griddata(pts, values, (X, Y), method=method, fill_value=np.nan)
        if np.isnan(grid).any() and method != "nearest":
            grid_near = griddata(pts, values, (X, Y), method="nearest")
            nan_mask = np.isnan(grid)
            grid[nan_mask] = grid_near[nan_mask]
    else:
        grid = np.empty((H, W), dtype=values.dtype)
        for i in range(H):
            for j in range(W):
                dx = pts[:, 0] - X[i, j]
                dy = pts[:, 1] - Y[i, j]
                idx = np.argmin(dx * dx + dy * dy)
                grid[i, j] = values[idx]
    return grid.astype(np.float32)


def _select_timesteps(arr, timesteps, stride):
    T = arr.shape[0]
    if timesteps <= 0:
        idx = np.arange(0, T, stride, dtype=int)
    else:
        idx = np.arange(0, T, stride, dtype=int)[:timesteps]
    return idx


def load_one(path, H=64, W=64, method="nearest", target="PS",
             timesteps=1, t_stride=1, use_pointcloud=False, use_mask=False):
    data = np.load(path)
    has_mesh = "mesh_pos" in data
    has_node = "node_type" in data

    if not has_mesh and use_pointcloud:
        xy_all = data["pointcloud"]
    else:
        xy_all = data["mesh_pos"]

    if not has_node and use_mask:
        node_all = data["mask"].astype(np.float32)
    else:
        node_all = data["node_type"].astype(np.float32) if has_node else np.zeros_like(data["VX"])

    vx_all = data["VX"]
    vy_all = data["VY"]
    target_all = data[target]

    idx = _select_timesteps(vx_all, timesteps, t_stride)

    Xs = []
    Ys = []
    for t in idx:
        xy = xy_all[t]
        vx = vx_all[t]
        vy = vy_all[t]
        node_type = node_all[t]
        target_arr = target_all[t]

        xy_min = xy.min(axis=0)
        xy_max = xy.max(axis=0)
        xy_n = (xy - xy_min) / (xy_max - xy_min + 1e-8)

        xch = []
        xch.append(interp_to_grid(xy_n, vx, H=H, W=W, method=method))
        xch.append(interp_to_grid(xy_n, vy, H=H, W=W, method=method))
        xch.append(interp_to_grid(xy_n, node_type, H=H, W=W, method=method))
        X = np.stack(xch, axis=0)

        Y = interp_to_grid(xy_n, target_arr, H=H, W=W, method=method)[None, ...]
        Xs.append(X[None, ...])
        Ys.append(Y[None, ...])

    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    return X, Y


def process_split(files, H, W, method, target, max_samples, timesteps, t_stride,
                  use_pointcloud=False, use_mask=False):
    Xs = []
    Ys = []
    take = files if max_samples <= 0 else files[:max_samples]
    for p in take:
        X, Y = load_one(p, H=H, W=W, method=method, target=target,
                        timesteps=timesteps, t_stride=t_stride,
                        use_pointcloud=use_pointcloud, use_mask=use_mask)
        Xs.append(X)
        Ys.append(Y)
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    return X, Y


def main():
    ap = argparse.ArgumentParser(description="Convert subset of EAGLE .npz files to runner-friendly NPZ.")
    ap.add_argument("--root", required=True, help="Directory containing EAGLE .npz samples")
    ap.add_argument("--out-root", default="eagle_npz")
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--method", default="nearest", choices=["nearest", "linear", "cubic"])
    ap.add_argument("--target", default="PS", choices=["PS", "PG", "VX", "VY", "VMAG"])
    ap.add_argument("--train-samples", type=int, default=200, help="Number of samples for train split")
    ap.add_argument("--val-samples", type=int, default=40, help="Number of samples for val split")
    ap.add_argument("--test-samples", type=int, default=40, help="Number of samples for test split")
    ap.add_argument("--timesteps-per-file", type=int, default=1, help="How many timesteps to extract per raw file (<=0 means all)")
    ap.add_argument("--t-stride", type=int, default=1, help="Stride between timesteps when sampling inside a file")
    ap.add_argument("--use-pointcloud", action="store_true", help="Use pointcloud as mesh_pos if mesh_pos key is missing (EAGLE spline format)")
    ap.add_argument("--use-mask", action="store_true", help="Use mask as node_type if node_type key is missing (EAGLE spline format)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.root, "*.npz")))
    if len(files) == 0:
        raise SystemExit(f"No .npz files found under {args.root}")

    H = W = args.grid
    os.makedirs(args.out_root, exist_ok=True)

    # split the list
    n_train = min(args.train_samples, len(files))
    n_val = min(args.val_samples, max(0, len(files) - n_train))
    n_test = min(args.test_samples, max(0, len(files) - n_train - n_val))

    train_files = files[:n_train]
    val_files   = files[n_train:n_train + n_val]
    test_files  = files[n_train + n_val:n_train + n_val + n_test]

    for split_name, flist in [("train", train_files), ("val", val_files), ("test", test_files)]:
        if not flist:
            print(f"[skip] {split_name} empty")
            continue
        print(f"[info] {split_name}: using {len(flist)} files")
        X, Y = process_split(flist, H=H, W=W, method=args.method,
                             target="PS" if args.target == "VMAG" else args.target,
                             max_samples=len(flist),
                             timesteps=args.timesteps_per_file,
                             t_stride=args.t_stride,
                             use_pointcloud=args.use_pointcloud,
                             use_mask=args.use_mask)
        if args.target == "VMAG":
            # recompute target as |v| from VX/VY
            vmag = np.sqrt(X[:,0]**2 + X[:,1]**2)  # (N,H,W)
            Y = vmag[:,None,...]
        out_path = os.path.join(args.out_root, f"{split_name}.npz")
        np.savez_compressed(out_path, input=X, target=Y)
        print(f"[done] {split_name}: saved {X.shape[0]} samples to {out_path} (Cin={X.shape[1]}, H={H}, W={W}, Cout={Y.shape[1]})")


if __name__ == "__main__":
    main()
