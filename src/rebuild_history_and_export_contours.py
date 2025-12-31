#!/usr/bin/env python3
"""Rebuild training histories and export contour-ready CFDBench samples.

This utility pulls per-experiment history CSVs, normalizes them to a fixed
epoch count, and consolidates them into a single CSV that downstream plotting
scripts (e.g., plot_training_history.py) can reuse. It also standardizes the
CFDBench sample NPZ files into a publication-friendly layout so JCP-style
figures can be regenerated without rerunning training.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_arguments() -> argparse.Namespace:
    """Build the CLI parser."""
    ap = argparse.ArgumentParser(
        description="Normalize training histories and export contour data."
    )
    ap.add_argument(
        "--root_dir",
        default=".",
        help="Repository root. Default: current working directory.",
    )
    ap.add_argument(
        "--results_subdir",
        default="results_tube",
        help="Relative path under root_dir that contains experiment folders.",
    )
    ap.add_argument(
        "--max_epochs",
        type=int,
        default=2000,
        help="Normalize all training histories to this epoch count.",
    )
    ap.add_argument(
        "--export_dir",
        default="fig_data",
        help="Directory (relative to root_dir unless absolute) for merged CSV/NPZ.",
    )
    ap.add_argument(
        "--sample_idxs",
        default="0,1,2",
        help=(
            "Comma-separated list of sample indices to export (e.g. '0,2,4') or "
            "'all' to export every sample_XXX.npz that exists."
        ),
    )
    ap.add_argument(
        "--case_name",
        default=None,
        help=(
            "Optional explicit case name for contour exports (defaults to "
            "results_subdir with 'results_' stripped)."
        ),
    )
    return ap.parse_args()


# ---------------------------------------------------------------------------
# History processing
# ---------------------------------------------------------------------------

def list_experiments(results_root: Path) -> List[Path]:
    """Return sorted experiment directories inside results_root."""
    return sorted([p for p in results_root.iterdir() if p.is_dir()])


def find_history_file(exp_dir: Path) -> Optional[Path]:
    """Pick the best history CSV inside an experiment directory."""
    direct = exp_dir / "history.csv"
    if direct.exists():
        return direct
    candidates = sorted(exp_dir.glob("*history*.csv"))
    return candidates[0] if candidates else None


def load_history(path: Path) -> pd.DataFrame:
    """Read a history CSV and ensure an 'epoch' column exists."""
    df = pd.read_csv(path)
    epoch_col = None
    for col in df.columns:
        if col.lower() == "epoch":
            epoch_col = col
            break
    if epoch_col is None:
        raise ValueError(f"{path}: missing 'epoch' column")
    if epoch_col != "epoch":
        df = df.rename(columns={epoch_col: "epoch"})
    df["epoch"] = df["epoch"].astype(int)
    df = df.sort_values("epoch").drop_duplicates("epoch", keep="last")
    return df


def normalize_history(df: pd.DataFrame, max_epochs: int, model_name: str) -> pd.DataFrame:
    """Align epochs to 1..max_epochs and tag the model name."""
    template = pd.DataFrame({"epoch": np.arange(1, max_epochs + 1, dtype=int)})
    merged = template.merge(df, on="epoch", how="left")
    merged.insert(0, "model", model_name)
    return merged


def rebuild_histories(
    results_root: Path, max_epochs: int
) -> Tuple[pd.DataFrame, List[str]]:
    """Rebuild normalized histories for every experiment."""
    histories = []
    missing = []
    for exp_dir in list_experiments(results_root):
        hist_file = find_history_file(exp_dir)
        if hist_file is None:
            missing.append(exp_dir.name)
            continue
        try:
            df = load_history(hist_file)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[history] skip {exp_dir.name}: {exc}")
            continue
        histories.append(normalize_history(df, max_epochs, exp_dir.name))
    merged = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    return merged, missing


def save_history_csv(merged: pd.DataFrame, destination: Path) -> None:
    """Write the merged history CSV."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(destination, index=False)
    print(f"[history] wrote {destination}")


def summarize_val_nmse(merged: pd.DataFrame) -> None:
    """Print a quick summary focusing on NMSE (PIBERT should stay on top)."""
    if merged.empty or "val_nmse" not in merged.columns:
        print("[history] val_nmse column missing; skipping summary.")
        return
    recent = (
        merged.dropna(subset=["val_nmse"])
        .sort_values("epoch")
        .groupby("model", as_index=True)["val_nmse"]
        .last()
    )
    if recent.empty:
        print("[history] no finite val_nmse values to summarize.")
        return
    best_model = recent.idxmin()
    print("[history] final validation NMSE by model:")
    for model, value in recent.sort_values().items():
        badge = " (best)" if model == best_model else ""
        print(f"  - {model}: {value:.4e}{badge}")
    if not best_model.lower().startswith("pibert"):
        print("[history] warning: PIBERT is not currently the best model by val_nmse.")


# ---------------------------------------------------------------------------
# Contour export helpers
# ---------------------------------------------------------------------------

SAMPLE_PATTERN = re.compile(r"sample_(\d+)\.npz$")


def parse_sample_indices(value: str) -> Optional[List[int]]:
    """Parse --sample_idxs into explicit indices or None for 'all'."""
    if value is None:
        return [0, 1, 2]
    value = value.strip()
    if not value:
        return [0, 1, 2]
    if value.lower() == "all":
        return None
    parts = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts.append(int(chunk))
    return parts or [0, 1, 2]


def split_model_seed(name: str) -> Tuple[str, Optional[int]]:
    """Return (model, seed) given directory name like 'PIBERT_seed42'."""
    match = re.match(r"(.+?)_seed(\d+)$", name)
    if match:
        return match.group(1), int(match.group(2))
    return name, None


def list_sample_files(exp_dir: Path, indices: Optional[Sequence[int]]) -> List[Path]:
    """Collect sample_XXX.npz files based on explicit indices or all files."""
    if indices is None:
        return sorted(exp_dir.glob("sample_*.npz"))
    files = []
    for idx in indices:
        candidate = exp_dir / f"sample_{idx:03d}.npz"
        if candidate.exists():
            files.append(candidate)
        else:
            print(f"[contours] missing {candidate}")
    return files


def load_sample_npz(path: Path) -> Dict[str, np.ndarray]:
    """Load a sample NPZ with gt/pred (+optional relerr/x/y/meta)."""
    with np.load(path, allow_pickle=False) as data:
        sample = {
            "gt": data["gt"].astype(np.float32),
            "pred": data["pred"].astype(np.float32),
        }
        for key in ("relerr", "x", "y", "meta"):
            if key in data.files:
                sample[key] = data[key]
    return sample


def extract_uvp(arr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (u, v, p) slices from a (C,H,W) tensor."""
    channels = arr.shape[0]
    u = arr[0] if channels >= 1 else None
    v = arr[1] if channels >= 2 else None
    p = arr[2] if channels >= 3 else None
    if channels == 1:
        p = arr[0]
        u = v = None
    return (
        u.copy() if u is not None else None,
        v.copy() if v is not None else None,
        p.copy() if p is not None else None,
    )


def ensure_array(field: Optional[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    """Return a float32 array, filling with NaNs if the field is missing."""
    H, W = shape
    if field is None:
        return np.full((H, W), np.nan, dtype=np.float32)
    return field.astype(np.float32, copy=False)


def compute_speed(u: Optional[np.ndarray], v: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Compute velocity magnitude."""
    if u is None or v is None:
        return None
    return np.sqrt(np.maximum(u ** 2 + v ** 2, 0.0)).astype(np.float32)


def compute_vorticity(u: Optional[np.ndarray], v: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Compute 2D scalar vorticity (∂v/∂x − ∂u/∂y)."""
    if u is None or v is None:
        return None
    dv_dy, dv_dx = np.gradient(v)
    du_dy, _ = np.gradient(u)
    vort = dv_dx - du_dy
    return vort.astype(np.float32)


def normalized_mse(pred: np.ndarray, gt: np.ndarray) -> float:
    """Return NMSE for a stack of channels."""
    num = float(np.sum((pred - gt) ** 2, dtype=np.float64))
    den = float(np.sum(gt ** 2, dtype=np.float64))
    if den <= 0:
        return math.nan
    return num / den


def build_mesh(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Return a simple [0,1] meshgrid for (H,W) samples."""
    H, W = shape
    x = np.linspace(0.0, 1.0, num=W, dtype=np.float32)
    y = np.linspace(0.0, 1.0, num=H, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    return xx.astype(np.float32), yy.astype(np.float32)


def prepare_contour_payload(
    sample: Dict[str, np.ndarray],
    case: str,
    base_model: str,
    exp_name: str,
    seed: Optional[int],
    sample_path: Path,
) -> Tuple[Dict[str, np.ndarray], int]:
    """Transform a sample dict into a standardized payload."""
    gt = sample["gt"]
    pred = sample["pred"]
    H, W = gt.shape[1:]

    u_true, v_true, p_true = extract_uvp(gt)
    u_pred, v_pred, p_pred = extract_uvp(pred)

    speed_true = compute_speed(u_true, v_true)
    speed_pred = compute_speed(u_pred, v_pred)
    vort_true = compute_vorticity(u_true, v_true)
    vort_pred = compute_vorticity(u_pred, v_pred)

    mesh_x = sample.get("x")
    mesh_y = sample.get("y")
    if mesh_x is None or mesh_y is None:
        mesh_x, mesh_y = build_mesh((H, W))

    nmse_value = normalized_mse(pred, gt)
    match = SAMPLE_PATTERN.search(sample_path.name)
    sample_idx = int(match.group(1)) if match else -1

    meta = {
        "case": case,
        "experiment": exp_name,
        "model": base_model,
        "seed": seed,
        "sample_index": sample_idx,
        "source_sample": str(sample_path),
        "nmse": nmse_value,
        "channels": int(gt.shape[0]),
    }

    payload = {
        "gt": gt.astype(np.float32, copy=False),
        "pred": pred.astype(np.float32, copy=False),
        "u_true": ensure_array(u_true, (H, W)),
        "v_true": ensure_array(v_true, (H, W)),
        "p_true": ensure_array(p_true, (H, W)),
        "u_pred": ensure_array(u_pred, (H, W)),
        "v_pred": ensure_array(v_pred, (H, W)),
        "p_pred": ensure_array(p_pred, (H, W)),
        "speed_true": ensure_array(speed_true, (H, W)),
        "speed_pred": ensure_array(speed_pred, (H, W)),
        "vorticity_true": ensure_array(vort_true, (H, W)),
        "vorticity_pred": ensure_array(vort_pred, (H, W)),
        "mesh_x": mesh_x.astype(np.float32, copy=False),
        "mesh_y": mesh_y.astype(np.float32, copy=False),
        "meta_json": np.array(json.dumps(meta), dtype=object),
    }
    if "relerr" in sample:
        payload["relerr"] = sample["relerr"].astype(np.float32, copy=False)
    return payload, sample_idx


def export_contours(
    results_root: Path,
    export_dir: Path,
    case_name: str,
    sample_indices: Optional[Sequence[int]],
) -> List[Path]:
    """Export contour-ready NPZ files for every experiment/sample."""
    out_paths: List[Path] = []
    for exp_dir in list_experiments(results_root):
        if not any(exp_dir.glob("sample_*.npz")):
            continue
        base_model, seed = split_model_seed(exp_dir.name)
        samples = list_sample_files(exp_dir, sample_indices)
        if not samples:
            continue
        target_dir = export_dir / "contours" / case_name / base_model
        target_dir.mkdir(parents=True, exist_ok=True)
        for sample_path in samples:
            sample = load_sample_npz(sample_path)
            payload, idx = prepare_contour_payload(
                sample, case_name, base_model, exp_dir.name, seed, sample_path
            )
            seed_suffix = f"_seed{seed:02d}" if seed is not None else ""
            out_name = (
                f"{sample_path.stem}{seed_suffix}.npz"
                if seed_suffix
                else sample_path.name
            )
            out_path = target_dir / out_name
            np.savez_compressed(out_path, **payload)
            out_paths.append(out_path)
            print(f"[contours] wrote {out_path} (sample {idx:03d}, {base_model})")
    if not out_paths:
        print(f"[contours] no samples exported from {results_root}")
    return out_paths


def debug_plot_example(export_dir: str, case: str, model: str, sample_idx: int) -> None:
    """Quick visualization sanity check."""
    base = Path(export_dir) / "contours" / case / model
    path = base / f"sample_{sample_idx:03d}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        speed_true = data["speed_true"]
        speed_pred = data["speed_pred"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    axes[0].imshow(speed_true, origin="lower", cmap="viridis")
    axes[0].set_title("speed_true")
    axes[1].imshow(speed_pred, origin="lower", cmap="viridis")
    axes[1].set_title("speed_pred")
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_arguments()
    root_dir = Path(args.root_dir).resolve()
    results_root = (root_dir / args.results_subdir).resolve()
    if not results_root.exists():
        raise SystemExit(f"results_subdir not found: {results_root}")

    export_dir = Path(args.export_dir)
    if not export_dir.is_absolute():
        export_dir = root_dir / export_dir
    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"[config] root_dir         : {root_dir}")
    print(f"[config] results_root     : {results_root}")
    print(f"[config] export_dir       : {export_dir}")
    print(f"[config] max_epochs       : {args.max_epochs}")
    sample_indices = parse_sample_indices(args.sample_idxs)
    sample_desc = "all" if sample_indices is None else ",".join(f"{i}" for i in sample_indices)
    print(f"[config] sample indices   : {sample_desc}")

    # ----- training history -----
    merged_history, missing = rebuild_histories(results_root, args.max_epochs)
    if merged_history.empty:
        print(f"[history] no history CSVs found under {results_root}")
    else:
        hist_name = f"history_merged_{results_root.name}.csv"
        hist_path = export_dir / hist_name
        save_history_csv(merged_history, hist_path)
        summarize_val_nmse(merged_history)
    if missing:
        print(f"[history] experiments without CSV: {', '.join(missing)}")

    # ----- contour exports -----
    case_name = (args.case_name or results_root.name.replace("results_", "")).lower()
    export_contours(results_root, export_dir, case_name, sample_indices)


if __name__ == "__main__":
    main()
