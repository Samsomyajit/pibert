#!/usr/bin/env python3
import os, glob, zipfile, argparse, random, shutil
from pathlib import Path
import numpy as np

# ---------- helpers ----------
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

# Candidate names
P_CANDIDATES = ["p", "pressure", "press", "p_field", "pressure_field", "pressure2d"]
U_CANDIDATES = ["u", "ux", "u_vel", "u_velocity", "velocity_x", "vel_x"]
V_CANDIDATES = ["v", "uy", "v_vel", "v_velocity", "velocity_y", "vel_y"]
# Common dam-break / shallow-water height names
HLIKE_CANDIDATES = ["h", "height", "water_height", "depth", "eta", "free_surface", "surface_height"]

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

def find_existing_name(case_dir: str, candidates):
    return next((n for n in candidates if
                 os.path.exists(os.path.join(case_dir, f"{n}.npy")) or
                 os.path.exists(os.path.join(case_dir, n, f"{n}.npy")) or
                 glob.glob(os.path.join(case_dir, n, "*.npy")) or
                 glob.glob(os.path.join(case_dir, "**", f"{n}.npy"), recursive=True)), None)

# ---------- loaders for pressure targets ----------
def load_pressure_direct(case_dir: str, in_frame: str, out_frame: str):
    # True pressure ground truth
    p_name = find_existing_name(case_dir, P_CANDIDATES)
    if p_name is None:
        raise FileNotFoundError(f"[{case_dir}] missing pressure field (tried {P_CANDIDATES})")

    p_all = _load_field_generic(case_dir, p_name, out_frame)   # target (H,W)
    H, W = p_all.shape

    # conditioning: constant mean of chosen in_frame
    p_in = pick_2d(np.expand_dims(p_all, 0), in_frame).astype(np.float32) if in_frame != out_frame else p_all
    p_cond = np.full((H, W), float(p_in.mean()), dtype=np.float32)

    X = np.expand_dims(p_cond, axis=0)   # (1,H,W)
    Y = np.expand_dims(p_all.astype(np.float32), axis=0)  # (1,H,W)
    return X, Y

def _load_uv(case_dir: str, in_frame: str, out_frame: str):
    u_name = find_existing_name(case_dir, U_CANDIDATES)
    v_name = find_existing_name(case_dir, V_CANDIDATES)
    if u_name is None or v_name is None:
        raise FileNotFoundError(f"[{case_dir}] missing U/V fields (tried U={U_CANDIDATES}, V={V_CANDIDATES})")
    u = _load_field_generic(case_dir, u_name, out_frame)
    v = _load_field_generic(case_dir, v_name, out_frame)
    return u, v

def load_pressure_from_height(case_dir: str, in_frame: str, out_frame: str,
                              scalar_candidates, rho: float, g: float, scale: float = 1.0):
    s_name = find_existing_name(case_dir, scalar_candidates)
    if s_name is None:
        raise FileNotFoundError(f"[{case_dir}] none of scalar candidates present (tried {scalar_candidates})")

    s_all = _load_field_generic(case_dir, s_name, out_frame)   # (H,W)
    H, W = s_all.shape

    p_all = (rho * g * s_all * scale).astype(np.float32)

    s_in = pick_2d(np.expand_dims(s_all, 0), in_frame).astype(np.float32) if in_frame != out_frame else s_all
    s_cond = np.full((H, W), float(s_in.mean()), dtype=np.float32)

    X = np.expand_dims(s_cond, axis=0)   # (1,H,W)
    Y = np.expand_dims(p_all, axis=0)    # (1,H,W)
    return X, Y

def load_pressure_from_uv(case_dir: str, in_frame: str, out_frame: str,
                          method: str, rho: float, dx: float, dy: float):
    u, v = _load_uv(case_dir, in_frame, out_frame)
    H, W = u.shape

    if method == "dynamic":
        p_all = 0.5 * rho * (u*u + v*v)
    elif method == "poisson":
        # Compute RHS = -rho * sum_i sum_j (∂i u_j)(∂j u_i)
        # 2D expansion: (∂x u)^2 + 2*(∂x v)*(∂y u) + (∂y v)^2
        du_dx = np.gradient(u, dx, axis=1)
        du_dy = np.gradient(u, dy, axis=0)
        dv_dx = np.gradient(v, dx, axis=1)
        dv_dy = np.gradient(v, dy, axis=0)
        rhs = -rho * (du_dx**2 + 2.0 * dv_dx * du_dy + dv_dy**2)

        # Periodic Poisson solver in Fourier space using discrete Laplacian symbol
        kx = 2.0 * np.pi * np.fft.fftfreq(W, d=dx)  # shape (W,)
        ky = 2.0 * np.pi * np.fft.fftfreq(H, d=dy)  # shape (H,)
        KX, KY = np.meshgrid(kx, ky)
        # Discrete Laplacian denominator (periodic, 2nd-order):
        denom = (4.0 * (np.sin(0.5 * KX * dx)**2) / (dx*dx) +
                 4.0 * (np.sin(0.5 * KY * dy)**2) / (dy*dy))
        rhs_hat = np.fft.fft2(rhs)
        denom[0,0] = np.inf  # fix gauge by setting p_hat(0,0)=0
        p_hat = rhs_hat / (-denom)  # since Laplacian(u) ~ -denom * U_hat
        p_hat[0,0] = 0.0
        p_all = np.real(np.fft.ifft2(p_hat)).astype(np.float32)
    else:
        raise ValueError("Unknown UV derivation method. Use 'dynamic' or 'poisson'.")

    p_in = pick_2d(np.expand_dims(p_all, 0), in_frame).astype(np.float32) if in_frame != out_frame else p_all
    p_cond = np.full((H, W), float(p_in.mean()), dtype=np.float32)

    X = np.expand_dims(p_cond, axis=0)   # (1,H,W)
    Y = np.expand_dims(p_all.astype(np.float32), axis=0)  # (1,H,W)
    return X, Y

# ---------- HF listing + download ----------
def list_available(repo_id="chen-yingfa/CFDBench"):
    try:
        from huggingface_hub import HfApi
    except Exception as e:
        raise SystemExit("Please install: pip install -U huggingface_hub") from e

    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except TypeError:
        files = api.list_repo_files(repo_id=repo_id)

    zips = [p for p in files if p.lower().endswith(".zip")]
    return sorted(zips)

ALIASES = {
    "dam": ["dam", "dam.zip", "dambreak", "dam-break", "dam_break"],
    "cylinder": ["cylinder"],
    "channel": ["channel", "pipe", "duct"],
    "tube": ["tube"],
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

def _stem_from_path(name: str) -> str:
    base = os.path.basename(name)
    if base.lower().endswith(".npy"):
        return base[:-4].lower()
    return ""

def _safe_join(base: Path, member: str) -> Path:
    target = (base / member).resolve()
    return target

def inspect_zip_fields(zip_paths):
    """Return (all_stems: set[str], pressure_present: bool)."""
    all_stems = set()
    for zp in zip_paths:
        with zipfile.ZipFile(zp, "r") as zf:
            for zi in zf.infolist():
                if zi.is_dir(): continue
                stem = _stem_from_path(zi.filename)
                if stem:
                    all_stems.add(stem)
    pressure_present = any(stem in {c.lower() for c in P_CANDIDATES} for stem in all_stems)
    return all_stems, pressure_present


def build_from_npz_folder(root: Path, out_dir: Path, uv_method: str, rho: float, dx: float, dy: float):
    """
    Convenience: if data_root contains train/val/test npz files with targets holding u,v (first 2 channels),
    derive pressure via u,v and append it as an extra channel to target.
    """
    root = Path(root)
    splits = ["train", "val", "test"]
    if not all((root / f"{sp}.npz").exists() for sp in splits):
        return False  # not an npz layout

    def derive_p(arr_uv):
        u = arr_uv[:, 0]
        v = arr_uv[:, 1]
        if uv_method == "dynamic":
            p = 0.5 * rho * (u*u + v*v)
        else:
            du_dx = np.gradient(u, dx, axis=-1)
            du_dy = np.gradient(u, dy, axis=-2)
            dv_dx = np.gradient(v, dx, axis=-1)
            dv_dy = np.gradient(v, dy, axis=-2)
            rhs = -rho * (du_dx**2 + 2.0 * dv_dx * du_dy + dv_dy**2)
            H, W = rhs.shape[-2:]
            kx = 2.0 * np.pi * np.fft.fftfreq(W, d=dx)
            ky = 2.0 * np.pi * np.fft.fftfreq(H, d=dy)
            KX, KY = np.meshgrid(kx, ky)
            denom = (4.0 * (np.sin(0.5 * KX * dx)**2) / (dx*dx) +
                     4.0 * (np.sin(0.5 * KY * dy)**2) / (dy*dy))
            denom[0, 0] = np.inf
            p_list = []
            for rhs_i in rhs:
                rhs_hat = np.fft.fft2(rhs_i)
                p_hat = rhs_hat / (-denom)
                p_hat[0, 0] = 0.0
                p_list.append(np.real(np.fft.ifft2(p_hat)))
            p = np.stack(p_list, axis=0)
        return p.astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    for sp in splits:
        data = np.load(root / f"{sp}.npz")
        inp = data["input"]  # (N, Cin, H, W)
        tgt = data["target"]  # (N, Cout, H, W)
        if tgt.shape[1] < 2:
            raise SystemExit(f"{sp}.npz missing u,v channels; got target shape {tgt.shape}")
        p = derive_p(tgt[:, :2])
        tgt_out = np.concatenate([tgt, p[:, None]], axis=1)
        np.savez(out_dir / f"{sp}.npz", input=inp, target=tgt_out)
        print(f"[npz] {sp}: in{inp.shape} -> target{tgt_out.shape}")
    print("Wrote NPZ (npz-folder mode) to:", out_dir.resolve())
    return True


def build_from_samples_dir(root: Path, out_dir: Path, uv_method: str, rho: float, dx: float, dy: float):
    """
    If data_root contains export_samples outputs (sample_*.npz with pred u,v),
    derive pressure from pred u,v and pack them into a single test.npz.
    """
    import glob
    files = sorted(glob.glob(str(root / "sample_*.npz")))
    if not files:
        return False

    preds = []
    for f in files:
        d = np.load(f)
        pred = d["pred"]  # (2,H,W) expected
        if pred.shape[0] < 2:
            raise SystemExit(f"{f} missing u,v channels (pred shape {pred.shape})")
        preds.append(pred[None])
    arr = np.concatenate(preds, axis=0)  # (N, C, H, W)
    u = arr[:,0]; v = arr[:,1]
    if uv_method == "dynamic":
        p = 0.5 * rho * (u*u + v*v)
    else:
        du_dx = np.gradient(u, dx, axis=-1)
        du_dy = np.gradient(u, dy, axis=-2)
        dv_dx = np.gradient(v, dx, axis=-1)
        dv_dy = np.gradient(v, dy, axis=-2)
        rhs = -rho * (du_dx**2 + 2.0 * dv_dx * du_dy + dv_dy**2)
        H, W = rhs.shape[-2:]
        kx = 2.0 * np.pi * np.fft.fftfreq(W, d=dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(H, d=dy)
        KX, KY = np.meshgrid(kx, ky)
        denom = (4.0 * (np.sin(0.5 * KX * dx)**2) / (dx*dx) +
                 4.0 * (np.sin(0.5 * KY * dy)**2) / (dy*dy))
        denom[0,0] = np.inf
        p_list = []
        for rhs_i in rhs:
            rhs_hat = np.fft.fft2(rhs_i)
            p_hat = rhs_hat / (-denom)
            p_hat[0,0] = 0.0
            p_list.append(np.real(np.fft.ifft2(p_hat)))
        p = np.stack(p_list, axis=0)
    p = p.astype(np.float32)

    tgt = np.concatenate([arr, p[:,None]], axis=1)  # append pressure
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "test.npz", input=arr.astype(np.float32), target=tgt)
    print(f"[samples] {len(files)} files -> test.npz input{arr.shape} target{tgt.shape}")
    print("Wrote NPZ (samples mode) to:", out_dir.resolve())
    return True

def unzip_selected(zips, dest_root: Path, wanted_stems, skip_errors=True):
    """
    Extract ONLY files whose stem is in wanted_stems.
    Skips corrupt members gracefully if skip_errors=True.
    Returns a list of output directories created.
    """
    dest_root.mkdir(parents=True, exist_ok=True)
    out_dirs = []
    wanted = {w.lower() for w in wanted_stems}
    for zp in zips:
        zname = os.path.basename(zp)
        sub = zname.replace(".zip", "")
        out_dir = dest_root / sub
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_dirs.append(out_dir)
        with zipfile.ZipFile(zp, "r") as zf:
            for zi in zf.infolist():
                if zi.is_dir(): continue
                stem = _stem_from_path(zi.filename)
                if not stem or stem not in wanted:
                    continue
                try:
                    target_path = _safe_join(out_dir, zi.filename)
                    if not str(target_path).startswith(str(out_dir.resolve())):
                        if skip_errors:
                            print(f"[skip unsafe path] {zi.filename}")
                            continue
                        else:
                            raise RuntimeError(f"Unsafe path in zip: {zi.filename}")
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(zi) as src, open(target_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                except Exception as e:
                    msg = f"[skip corrupt] {zi.filename}: {e}"
                    if skip_errors:
                        print(msg)
                        continue
                    raise
    return out_dirs

def scan_cases(parent: Path):
    # consider any folder that contains at least one *.npy as a "case"
    cases = []
    for root, dirs, files in os.walk(parent):
        if any(f.endswith(".npy") for f in files):
            cases.append(root)
    cases = sorted(set(cases))
    return [c for c in cases if c != str(parent)]

def build_npz_from_dirs(case_dirs, out_dir: Path, in_frame: str, out_frame: str, seed=42,
                        max_cases=None, mode="direct",
                        derive_from=None, rho=1000.0, g=9.81, scale=1.0,
                        uv_method="dynamic", dx=1.0, dy=1.0):
    random.seed(seed)
    if max_cases is not None:
        random.shuffle(case_dirs)
        case_dirs = case_dirs[:max_cases]

    Xs, Ys = [], []
    for c in case_dirs:
        try:
            if mode == "direct":
                X, Y = load_pressure_direct(c, in_frame=in_frame, out_frame=out_frame)
            elif mode == "height":
                candidates = [s.strip() for s in derive_from] if isinstance(derive_from, (list, tuple)) \
                             else [s.strip() for s in derive_from.split(",") if s.strip()]
                X, Y = load_pressure_from_height(
                    c, in_frame=in_frame, out_frame=out_frame,
                    scalar_candidates=candidates, rho=rho, g=g, scale=scale)
            elif mode == "uv":
                X, Y = load_pressure_from_uv(
                    c, in_frame=in_frame, out_frame=out_frame,
                    method=uv_method, rho=rho, dx=dx, dy=dy)
            else:
                raise ValueError("Unknown mode")
            Xs.append(X); Ys.append(Y)
        except Exception as e:
            print(f"[skip] {c}: {e}")

    if not Xs:
        raise SystemExit("No valid cases built; nothing matched or derivation failed.")

    X = np.stack(Xs, axis=0)  # (N,1,H,W)
    Y = np.stack(Ys, axis=0)  # (N,1,H,W)
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
    ap.add_argument("--problem", default="dam",
                    help="Problem alias (dam, cavity, tube, cylinder) or exact zip path in repo")
    ap.add_argument("--out", default="cfdb_p_npz")
    ap.add_argument("--data-root", default=None,
                    help="If set, bypass download and read fields from a local directory tree (cfdb_*_npz).")
    ap.add_argument("--in-frame", choices=["first","mean","last"], default="first",
                    help="How to compute conditioning from raw frames (if present)")
    ap.add_argument("--out-frame", choices=["first","mean","last"], default="last",
                    help="Which target frame to use")
    ap.add_argument("--max_cases", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hf-repo", default="chen-yingfa/CFDBench")
    ap.add_argument("--list", action="store_true", help="List available *.zip paths in the repo and exit")

    # Height-derivation options
    ap.add_argument("--derive-pressure-from", default=None,
                    help="Comma-separated names to derive pressure from (e.g., 'h,depth,height'). Uses p=rho*g*field*scale.")
    ap.add_argument("--rho", type=float, default=1000.0, help="Fluid density (kg/m^3)")
    ap.add_argument("--g", type=float, default=9.81, help="Gravity (m/s^2)")
    ap.add_argument("--scale", type=float, default=1.0, help="Optional scale factor applied to the source field before rho*g")

    # UV-derivation options
    ap.add_argument("--derive-from-uv", choices=["dynamic","poisson"], default=None,
                    help="Derive pressure from u,v. 'dynamic' uses 0.5*rho*(u^2+v^2). 'poisson' solves PPE with periodic FFT.")
    ap.add_argument("--dx", type=float, default=1.0, help="Grid spacing in x for UV Poisson")
    ap.add_argument("--dy", type=float, default=1.0, help="Grid spacing in y for UV Poisson")

    # Utility
    ap.add_argument("--inspect-only", action="store_true",
                    help="Only print available .npy field stems in the matched zips and exit.")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit("Please install: pip install -U huggingface_hub") from e

    # 1) If data-root is provided, bypass HF download and read locally
    if args.data_root:
        # If the data_root itself is an npz folder (train/val/test), build directly and exit.
        root_path = Path(args.data_root)
        if (root_path / "train.npz").exists():
            print(f"[local] using npz folder: {root_path}")
            built = build_from_npz_folder(root_path, Path(args.out),
                                          uv_method=args.derive_from_uv or "dynamic",
                                          rho=args.rho, dx=args.dx, dy=args.dy)
            return
        # If the data_root is a folder of export_samples (sample_*.npz), handle it.
        built = build_from_samples_dir(root_path, Path(args.out),
                                       uv_method=args.derive_from_uv or "dynamic",
                                       rho=args.rho, dx=args.dx, dy=args.dy)
        if built:
            return
        disk_zips = [args.data_root]
        print(f"[local] using data root: {args.data_root}")
    else:
        # Special case: if data_root points to a folder with train/val/test npz, build directly.
        npz_root = Path(args.problem)
        if Path(args.problem).is_dir() and (npz_root / "train.npz").exists():
            built = build_from_npz_folder(npz_root, Path(args.out),
                                          uv_method=args.derive_from_uv or "dynamic",
                                          rho=args.rho, dx=args.dx, dy=args.dy)
            if built:
                return

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

        # Resolve local zip paths
        disk_zips = []
        for m in matches:
            p = Path(local_repo) / m
            if p.exists():
                disk_zips.append(str(p))
        if not disk_zips:
            raise SystemExit("Download finished but no zip files were found on disk.")

    # 3) Inspect available npy stems
    stems, have_p = inspect_zip_fields(disk_zips)
    print("\n== Fields found in zips (unique .npy stems) ==")
    print(", ".join(sorted(stems)) if stems else "(none)")
    print("Pressure present?:", have_p)

    if args.inspect_only:
        return

    # Decide extraction / mode
    mode = None
    wanted_stems = None

    if have_p and args.derive_pressure_from is None and args.derive_from_uv is None:
        # direct pressure available
        wanted_stems = [s for s in P_CANDIDATES if s in stems] or P_CANDIDATES
        mode = "direct"
        print("\nProceeding with direct pressure extraction.")
    elif args.derive_pressure_from:
        derive_list = [s.strip() for s in args.derive_pressure_from.split(",") if s.strip()]
        present_src = [s for s in derive_list if s in stems]
        if not present_src:
            raise SystemExit("No pressure in zip, and none of the derivation sources are present.\n"
                             f"Tried derivation candidates: {derive_list}")
        wanted_stems = present_src
        mode = "height"
        print(f"\nProceeding with height-derived pressure from: {present_src}")
        print(f"Formula: p = rho * g * field * scale  (rho={args.rho}, g={args.g}, scale={args.scale})")
    else:
        # Try UV derivation if u and v exist OR user asked for it
        if ("u" in stems or any(s in stems for s in U_CANDIDATES)) and \
           ("v" in stems or any(s in stems for s in V_CANDIDATES)):
            wanted_stems = ["u", "v"]
            mode = "uv"
            if args.derive_from_uv is None:
                print("\nPressure not available; deriving from u,v with default method: dynamic.")
                args.derive_from_uv = "dynamic"
            print(f"UV derivation method: {args.derive_from_uv} (rho={args.rho}, dx={args.dx}, dy={args.dy})")
        else:
            raise SystemExit("Neither pressure nor u,v nor height-like sources found to derive pressure.")

    # 4) Extract selected files only
    dest_root = Path("hf_cfdb") / args.problem
    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    print("\nExtracting selected:", disk_zips)
    out_dirs = unzip_selected(disk_zips, dest_root, wanted_stems=wanted_stems, skip_errors=True)

    # 5) Collect case folders and build NPZ
    case_dirs = []
    for d in out_dirs:
        case_dirs.extend(scan_cases(d))
    case_dirs = sorted(set(case_dirs))
    if not case_dirs:
        raise SystemExit("Unzipped, but found no case folders with the selected .npy files.")

    print(f"Found ~{len(case_dirs)} candidate case folders.")

    if mode == "direct":
        build_npz_from_dirs(case_dirs, Path(args.out),
                            in_frame=args.in_frame, out_frame=args.out_frame,
                            seed=args.seed, max_cases=args.max_cases,
                            mode="direct")
    elif mode == "height":
        build_npz_from_dirs(case_dirs, Path(args.out),
                            in_frame=args.in_frame, out_frame=args.out_frame,
                            seed=args.seed, max_cases=args.max_cases,
                            mode="height", derive_from=",".join(wanted_stems),
                            rho=args.rho, g=args.g, scale=args.scale)
    elif mode == "uv":
        build_npz_from_dirs(case_dirs, Path(args.out),
                            in_frame=args.in_frame, out_frame=args.out_frame,
                            seed=args.seed, max_cases=args.max_cases,
                            mode="uv", uv_method=args.derive_from_uv,
                            rho=args.rho, dx=args.dx, dy=args.dy)
    else:
        raise RuntimeError("Internal mode selection error.")

if __name__ == "__main__":
    main()
