#!/usr/bin/env python3
import os, re, glob, zipfile, argparse, random, shutil
from pathlib import Path
import numpy as np

# ---------- helpers for reading fields ----------
def _numeric_key(p: str):
    base = os.path.basename(p)
    digits = "".join(ch for ch in base if ch.isdigit())
    return int(digits) if digits else base

def pick_2d(arr: np.ndarray, mode: str) -> np.ndarray:
    if arr.ndim == 2:  # (H,W)
        return arr
    if arr.ndim == 3:  # (T,H,W)
        if mode == "first": return arr[0]
        if mode == "last":  return arr[-1]
        return arr.mean(axis=0)
    raise ValueError(f"Unexpected array shape {arr.shape}")

# Try a bunch of common velocity names
U_CANDIDATES = ["u", "ux", "u_vel", "u_velocity", "velocity_x", "vel_x"]
V_CANDIDATES = ["v", "uy", "v_vel", "v_velocity", "velocity_y", "vel_y"]

def _load_field_generic(case_dir: str, name: str, frame_mode: str) -> np.ndarray:
    """
    Load a field by trying several layout patterns:
      case/name.npy
      case/name/name.npy
      case/name/*.npy (multi-frame)
      recursive **/name.npy
    Returns (H,W) float32.
    """
    # 1) exact file in case root
    p = os.path.join(case_dir, f"{name}.npy")
    if os.path.isfile(p):
        return pick_2d(np.load(p), frame_mode).astype(np.float32)

    # 2) folder 'name' with a single file 'name.npy'
    p = os.path.join(case_dir, name, f"{name}.npy")
    if os.path.isfile(p):
        return pick_2d(np.load(p), frame_mode).astype(np.float32)

    # 3) folder 'name' with many frames *.npy
    multi = sorted(glob.glob(os.path.join(case_dir, name, "*.npy")), key=_numeric_key)
    if multi:
        stack = np.stack([np.load(q) for q in multi], axis=0)
        return pick_2d(stack, frame_mode).astype(np.float32)

    # 4) recursive search anywhere under case
    rec = glob.glob(os.path.join(case_dir, "**", f"{name}.npy"), recursive=True)
    if rec:
        return pick_2d(np.load(rec[0]), frame_mode).astype(np.float32)

    raise FileNotFoundError(f"Could not find '{name}' in {case_dir}")

def load_uv(case_dir: str, in_frame: str, out_frame: str, u_names=None, v_names=None):
    u_names = u_names or U_CANDIDATES
    v_names = v_names or V_CANDIDATES

    # find one existing name for U and V
    u_name = next((n for n in u_names if
                   os.path.exists(os.path.join(case_dir, f"{n}.npy")) or
                   os.path.exists(os.path.join(case_dir, n, f"{n}.npy")) or
                   glob.glob(os.path.join(case_dir, n, "*.npy")) or
                   glob.glob(os.path.join(case_dir, "**", f"{n}.npy"), recursive=True)), None)
    v_name = next((n for n in v_names if
                   os.path.exists(os.path.join(case_dir, f"{n}.npy")) or
                   os.path.exists(os.path.join(case_dir, n, f"{n}.npy")) or
                   glob.glob(os.path.join(case_dir, n, "*.npy")) or
                   glob.glob(os.path.join(case_dir, "**", f"{n}.npy"), recursive=True)), None)

    if u_name is None or v_name is None:
        raise FileNotFoundError(f"[{case_dir}] missing U/V fields (tried U={u_names}, V={v_names})")

    u_all = _load_field_generic(case_dir, u_name, out_frame)
    v_all = _load_field_generic(case_dir, v_name, out_frame)
    H, W = u_all.shape

    # inflow/conditioning from 'in_frame'
    u_in = pick_2d(np.expand_dims(u_all, 0), in_frame).astype(np.float32) if in_frame != out_frame else u_all
    v_in = pick_2d(np.expand_dims(v_all, 0), in_frame).astype(np.float32) if in_frame != out_frame else v_all

    # For constant inlet condition, take the left boundary mean
    u_cond = np.full((H, W), float(u_in[:, 0].mean()) if u_in.ndim==2 else float(u_in.mean()), dtype=np.float32)
    v_cond = np.full((H, W), float(v_in[:, 0].mean()) if v_in.ndim==2 else float(v_in.mean()), dtype=np.float32)

    X = np.stack([u_cond, v_cond], axis=0)  # (2,H,W)
    Y = np.stack([u_all.astype(np.float32), v_all.astype(np.float32)], axis=0)  # (2,H,W)
    return X, Y

# ---------- HF listing + download ----------
def list_available(repo_id="chen-yingfa/CFDBench"):
    """
    Return a sorted list of *.zip file paths in the dataset repo.
    Uses HfApi().list_repo_files (robust across hub versions).
    """
    try:
        from huggingface_hub import HfApi
    except Exception as e:
        raise SystemExit("Please install: pip install -U huggingface_hub") from e

    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except TypeError:
        # older hub versions don't accept repo_type in list_repo_files
        files = api.list_repo_files(repo_id=repo_id)

    # keep only zips
    zips = [p for p in files if p.lower().endswith(".zip")]
    return sorted(zips)


ALIASES = {
    "dam": ["dam", "dambreak", "dam-break", "dam_break"],
    "cylinder": ["cylinder"],
    "channel": ["channel", "pipe", "duct"],
    "cavity": ["cavity", "lid", "lid-driven"],
}

def fuzzy_match(problem, zip_paths):
    pats = ALIASES.get(problem.lower(), [problem.lower()])
    hits = []
    for p in zip_paths:
        low = p.lower()
        if any(a in low for a in pats):
            hits.append(p)
    return sorted(set(hits))

def unzip_all(zips, dest_root: Path):
    dest_root.mkdir(parents=True, exist_ok=True)
    out_dirs = []
    for zp in zips:
        zname = os.path.basename(zp)
        sub = zname.replace(".zip", "")
        out_dir = dest_root / sub
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall(out_dir)
        out_dirs.append(out_dir)
    return out_dirs

def scan_cases(parent: Path):
    # consider any folder that contains at least one *.npy as a "case"
    cases = []
    for root, dirs, files in os.walk(parent):
        if any(f.endswith(".npy") for f in files):
            cases.append(root)
    # keep only top-most case folders (avoid heavy nesting duplicates)
    cases = sorted(set(cases))
    # heuristic: drop parent itself if it’s just a container with many subfolders
    return [c for c in cases if c != str(parent)]

def build_npz_from_dirs(case_dirs, out_dir: Path, in_frame: str, out_frame: str, seed=42,
                        max_cases=None):
    random.seed(seed)
    if max_cases is not None:
        random.shuffle(case_dirs)
        case_dirs = case_dirs[:max_cases]

    Xs, Ys = [], []
    for c in case_dirs:
        try:
            X, Y = load_uv(c, in_frame=in_frame, out_frame=out_frame)
            Xs.append(X); Ys.append(Y)
        except Exception as e:
            print(f"[skip] {c}: {e}")

    if not Xs:
        raise SystemExit("No valid cases built; check field names and folder structure.")

    X = np.stack(Xs, axis=0)
    Y = np.stack(Ys, axis=0)
    N = len(Xs)

    n_tr = max(8, int(0.8 * N))
    n_va = max(4, int(0.1 * N))
    splits = {
        "train": (0, n_tr),
        "val":   (n_tr, n_tr + n_va),
        "test":  (n_tr + n_va, N),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for sp, (a, b) in splits.items():
        np.savez(out_dir / f"{sp}.npz", input=X[a:b], target=Y[a:b])
        print(f"{sp}: input{X[a:b].shape} target{Y[a:b].shape}")
    print("Wrote NPZ to:", out_dir.resolve())

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="cylinder",
                    help="Problem name or alias (try --list to see available)")
    ap.add_argument("--out", default="cfdb_npz")
    ap.add_argument("--in-frame", choices=["first","mean","last"], default="first",
                    help="How to compute the conditioning field from raw frames")
    ap.add_argument("--out-frame", choices=["first","mean","last"], default="last",
                    help="Which target frame to use")
    ap.add_argument("--input-mode", choices=["first","mean","last"], default="first",
                    help="(kept for CLI compatibility; same as --in-frame)")
    ap.add_argument("--max_cases", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hf-repo", default="chen-yingfa/CFDBench")
    ap.add_argument("--list", action="store_true", help="List available *.zip paths in the repo and exit")
    args = ap.parse_args()

    if args.input_mode != args.in_frame:
        # keep flag compatibility but avoid confusion
        args.in_frame = args.input_mode

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit("Please install: pip install -U huggingface_hub") from e

    # 1) List available zips
    zips_in_repo = list_available(args.hf_repo)
    if args.list:
        if not zips_in_repo:
            print("No *.zip files visible in the repo (or network issue).")
            return
        print("Available zip paths:")
        for p in zips_in_repo:
            print(" -", p)
        return

    # 2) Fuzzy match the problem
    matches = fuzzy_match(args.problem, zips_in_repo)
    if not matches:
        print(f"No matches for '{args.problem}' in {args.hf_repo}.")
        if zips_in_repo:
            roots = sorted(set(p.split("/")[0] for p in zips_in_repo))
            print("Top-level options I can see:", ", ".join(roots))
        print("Tip: run with --list to inspect all zip paths.")
        return

    print("Downloading", matches, "from", args.hf_repo)
    local_repo = snapshot_download(
        repo_id=args.hf_repo,
        repo_type="dataset",
        allow_patterns=matches,
        local_dir="hf_cfdb",
        local_dir_use_symlinks=False,
    )

    # 3) Unzip into hf_cfdb/<problem_like>/subfolder
    dest_root = Path("hf_cfdb") / args.problem
    # clean previous to avoid accidental mixing
    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    # find the downloaded zip files on disk
    disk_zips = []
    for m in matches:
        p = Path(local_repo) / m
        if p.exists():
            disk_zips.append(str(p))
    if not disk_zips:
        raise SystemExit("Download finished but no zip files were found on disk.")

    print("Unzipping:", disk_zips)
    out_dirs = unzip_all(disk_zips, dest_root)

    # 4) Collect case folders and build NPZ
    case_dirs = []
    for d in out_dirs:
        case_dirs.extend(scan_cases(d))
    case_dirs = sorted(set(case_dirs))
    if not case_dirs:
        raise SystemExit("Unzipped, but found no case folders with .npy files.")

    print(f"Found ~{len(case_dirs)} candidate case folders.")
    build_npz_from_dirs(case_dirs, Path(args.out), in_frame=args.in_frame,
                        out_frame=args.out_frame, seed=args.seed, max_cases=args.max_cases)

if __name__ == "__main__":
    main()
