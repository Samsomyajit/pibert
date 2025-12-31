# get_cfdb_cylinder.py
import os, glob, zipfile, random, argparse, numpy as np
from pathlib import Path

# ---------------- helpers ----------------
def pick_2d(arr: np.ndarray, mode: str) -> np.ndarray:
    """Return (H,W): arr may be (H,W) or (T,H,W)."""
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if mode == "first": return arr[0]
        if mode == "last":  return arr[-1]
        return arr.mean(axis=0)
    raise ValueError(f"Unexpected array shape {arr.shape}")

def _numeric_key(p: str):
    """Sort key that respects numeric frame ids in filenames."""
    base = os.path.basename(p)
    digits = "".join(ch for ch in base if ch.isdigit())
    return int(digits) if digits else base

def load_field(case_dir: str, name: str, frame: str) -> np.ndarray:
    """
    Load 'u' or 'v' from a case folder supporting layouts:
      case/u.npy
      case/v.npy
      case/u/u.npy, case/v/v.npy
      case/u/*.npy (multiple frames)
      case/v/*.npy
    Returns (H,W) float32.
    """
    # 1) exact file in case root
    direct = os.path.join(case_dir, f"{name}.npy")
    if os.path.isfile(direct):
        return pick_2d(np.load(direct), frame).astype(np.float32)

    # 2) folder 'name' with a single file 'name.npy'
    nested = os.path.join(case_dir, name, f"{name}.npy")
    if os.path.isfile(nested):
        return pick_2d(np.load(nested), frame).astype(np.float32)

    # 3) folder 'name' with many frames *.npy
    multi = sorted(glob.glob(os.path.join(case_dir, name, "*.npy")), key=_numeric_key)
    if multi:
        stack = np.stack([np.load(p) for p in multi], axis=0)  # (T, H, W) or (T, ..., ...)
        return pick_2d(stack, frame).astype(np.float32)

    # 4) last-resort: recursive search for name.npy anywhere under case
    rec = glob.glob(os.path.join(case_dir, "**", f"{name}.npy"), recursive=True)
    if rec:
        return pick_2d(np.load(rec[0]), frame).astype(np.float32)

    raise FileNotFoundError(f"Could not locate '{name}' field in {case_dir}")

# ---------------- main build ----------------
def build_npz(unzip_root: Path, out_dir: Path, max_cases: int | None = None,
              seed: int = 42, frame: str = "mean") -> None:
    """
    Create NPZ splits with:
      input : (N, 2, H, W)  -> constant inflow [u_in, v_in]
      target: (N, 2, H, W)  -> [u, v]
    """
    cases = sorted(glob.glob(str(unzip_root / "case*")))
    if not cases:
        raise SystemExit(f"No cases found under {unzip_root}")

    if max_cases is not None:
        random.seed(seed)
        random.shuffle(cases)
        cases = cases[:max_cases]

    def make_arrays(case_paths):
        Xs, Ys = [], []
        for c in case_paths:
            try:
                u = load_field(c, "u", frame)   # (H,W)
                v = load_field(c, "v", frame)   # (H,W)
            except FileNotFoundError as e:
                print(f"[skip] {e}")
                continue

            H, W = u.shape
            u_in = float(u[:, 0].mean())
            v_in = float(v[:, 0].mean())

            X = np.stack([
                np.full((H, W), u_in, dtype=np.float32),
                np.full((H, W), v_in, dtype=np.float32)
            ], axis=0)  # (2,H,W)

            Y = np.stack([u.astype(np.float32), v.astype(np.float32)], axis=0)  # (2,H,W)

            Xs.append(X); Ys.append(Y)

        if not Xs:
            raise SystemExit("No valid cases after scanning. Check dataset layout.")
        return np.stack(Xs, axis=0), np.stack(Ys, axis=0)

    N = len(cases)
    n_tr = max(8, int(0.8 * N))
    n_va = max(4, int(0.1 * N))
    splits = {
        "train": cases[:n_tr],
        "val":   cases[n_tr:n_tr + n_va],
        "test":  cases[n_tr + n_va:],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for sp, lst in splits.items():
        X, Y = make_arrays(lst)
        np.savez(out_dir / f"{sp}.npz", input=X, target=Y)
        print(f"{sp}: input{X.shape} target{Y.shape}")
    print("Wrote NPZ to:", out_dir.resolve())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["bc", "geo", "prop"], default="bc")
    ap.add_argument("--out", default="cfdb_cylinder_npz")
    ap.add_argument("--max_cases", type=int, default=None)
    ap.add_argument("--frame", choices=["first", "mean", "last"], default="mean",
                    help="How to collapse time dimension when multiple frames exist.")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit("Please install: pip install -U huggingface_hub") from e

    repo_id = "chen-yingfa/CFDBench"
    allow = [f"cylinder/{args.subset}.zip"]
    print("Downloading", allow, "from", repo_id)
    local_repo = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=allow,
        local_dir="hf_cfdb",
        local_dir_use_symlinks=False,
    )

    zpath = Path(local_repo) / "cylinder" / f"{args.subset}.zip"
    if not zpath.exists():
        raise SystemExit(f"Zip not found: {zpath}")

    unzip_root = Path("hf_cfdb") / "cylinder" / args.subset
    unzip_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(unzip_root)

    build_npz(unzip_root, Path(args.out), max_cases=args.max_cases, frame=args.frame)

if __name__ == "__main__":
    main()
