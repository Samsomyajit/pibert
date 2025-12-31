#!/usr/bin/env python3
"""
Re-create the pressure panels (GT / Pred / Rel. Err.) in the paper-style grid.

Usage example (from repo root):
  python plot_pressure_jcp_grid.py \
      --cases results_dam_p:Dam results_tube_p:Tube \
      --models PINN_seed42 PIBERT_seed42 \
      --sample-idx 0 \
      --out fig_pressure_grid.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def robust_relerr(gt: np.ndarray, pr: np.ndarray, percentile: float = 95.0, eps_scale: float = 1e-3) -> np.ndarray:
    """Relative error with robust stabilization."""
    scale = np.percentile(np.abs(gt), percentile)
    eps = max(eps_scale * scale, 1e-8)
    return np.abs(pr - gt) / (np.abs(gt) + eps)


def parse_case_entry(entry: str) -> Tuple[Path, str]:
    """Parse 'path[:label]' strings."""
    if ":" in entry:
        path_str, label = entry.split(":", 1)
    else:
        path_str, label = entry, Path(entry).name
    return Path(path_str).expanduser().resolve(), label.strip()


def load_trip(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (GT, Pred, RelErr) from a sample npz."""
    data = np.load(path)
    gt = data["gt"].astype(np.float32)
    pr = data["pred"].astype(np.float32)
    GT = gt[0]
    PR = pr[0]
    RE = robust_relerr(GT, PR)
    return GT, PR, RE


def set_axis_style(ax: plt.Axes) -> None:
    ax.set_xlabel("x", fontsize=16)
    ax.set_ylabel("y", fontsize=16)
    ax.tick_params(direction="out", width=1.1, length=5, labelsize=14)
    ax.set_aspect("auto")
    sns.despine(ax=ax, top=True, right=True)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GT/Pred/Relative-error pressure grids for multiple cases.")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["results_dam_p:Dam Pressure", "results_tube_p:Tube Pressure"],
        help="List of 'path[:label]' entries for each case block.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["PINN_seed42", "PIBERT_seed42"],
        help="Model subdirectories to load inside each case folder (order = rows).",
    )
    parser.add_argument(
        "--model-labels",
        nargs="+",
        default=None,
        help="Optional pretty names per model (same length as --models).",
    )
    parser.add_argument("--sample-idx", type=int, default=0, help="sample_{idx:03d}.npz index to visualize.")
    parser.add_argument("--field-cmap", default="Spectral_r", help="Colormap for pressure fields.")
    parser.add_argument("--err-cmap", default="magma", help="Colormap for relative error.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--out", default="figure4_pressure_grid.png")
    parser.add_argument("--err-percentile", type=float, default=99.5, help="Clip relative error at this percentile.")
    parser.add_argument("--width", type=float, default=18.0, help="Figure width (inches).")
    parser.add_argument("--row-height", type=float, default=3.0, help="Height per model row (inches).")
    parser.add_argument("--case-label-offset", type=float, default=0.05, help="Fractional shift for case labels.")
    parser.add_argument("--model-label-offset", type=float, default=0.020, help="Fractional shift for model labels.")
    parser.add_argument("--share-field-scale", action="store_true", help="Use global vmin/vmax across all cases.")
    args = parser.parse_args()

    sns.set_theme(style="white", context="talk")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    case_entries = [parse_case_entry(ent) for ent in args.cases]
    models = args.models
    model_labels = args.model_labels if args.model_labels and len(args.model_labels) == len(models) else models

    records: List[dict] = []
    for case_idx, (case_path, case_label) in enumerate(case_entries):
        for model_idx, (model_name, model_label) in enumerate(zip(models, model_labels)):
            candidate = case_path / model_name / f"sample_{args.sample_idx:03d}.npz"
            fallback = case_path / model_name / f"sample_{args.sample_idx:03d}_{model_label.split('_seed')[0]}.npz"
            sample_path = candidate if candidate.exists() else fallback
            if not sample_path.exists():
                raise SystemExit(f"Missing {sample_path}.")
            GT, PR, RE = load_trip(sample_path)
            records.append(
                {
                    "case_index": case_idx,
                    "case_label": case_label,
                    "model_name": model_name,
                    "model_label": model_label.split("_seed")[0],
                    "gt": GT,
                    "pred": PR,
                    "err": RE,
                }
            )

    rows = len(records)
    ncols = 3
    fig_h = args.row_height * rows + 1.2
    fig = plt.figure(figsize=(args.width, fig_h), dpi=args.dpi)
    gs = fig.add_gridspec(rows, ncols)
    gs.update(hspace=0.08, wspace=0.03)

    # global color limits
    if args.share_field_scale:
        field_vals = np.concatenate([rec["gt"].ravel() for rec in records])
        fmin = float(field_vals.min())
        fmax = float(field_vals.max())
    else:
        field_per_case = {}
        for rec in records:
            idx = rec["case_index"]
            arr = rec["gt"]
            if idx not in field_per_case:
                field_per_case[idx] = [float(arr.min()), float(arr.max())]
            else:
                field_per_case[idx][0] = min(field_per_case[idx][0], float(arr.min()))
                field_per_case[idx][1] = max(field_per_case[idx][1], float(arr.max()))
    err_vals = np.concatenate([rec["err"].ravel() for rec in records])
    emax = np.percentile(err_vals, args.err_percentile)
    emax = max(float(emax), 1e-6)

    header_titles = ["Ground Truth p", "Prediction p", "Relative Error"]
    for j, title in enumerate(header_titles):
        x = (j + 0.5) / ncols
        fig.text(x, 0.987, title, ha="center", va="top", fontsize=24, weight="bold")

    im_field = im_err = None
    axes = []
    for row, rec in enumerate(records):
        ax_gt = fig.add_subplot(gs[row, 0])
        ax_pr = fig.add_subplot(gs[row, 1])
        ax_er = fig.add_subplot(gs[row, 2])
        axes.append((ax_gt, ax_pr, ax_er))

        if args.share_field_scale:
            vmin, vmax = fmin, fmax
        else:
            vmin, vmax = field_per_case[rec["case_index"]]

        im_field = ax_gt.imshow(rec["gt"], origin="lower", cmap=args.field_cmap,
                                vmin=vmin, vmax=vmax, interpolation="bilinear")
        ax_pr.imshow(rec["pred"], origin="lower", cmap=args.field_cmap,
                     vmin=vmin, vmax=vmax, interpolation="bilinear")
        im_err = ax_er.imshow(np.clip(rec["err"], 0.0, emax), origin="lower",
                              cmap=args.err_cmap, vmin=0.0, vmax=emax, interpolation="bilinear")

        for ax in (ax_gt, ax_pr, ax_er):
            set_axis_style(ax)

        # model label
        bbox = ax_gt.get_position()
        fig.text(bbox.x0 - args.model_label_offset, bbox.y0 + bbox.height / 2,
                 rec["model_label"], ha="right", va="center",
                 fontsize=17, weight="bold")

    # case labels (centered over the block of rows)
    rows_per_case = len(models)
    for case_idx, (_, case_label) in enumerate(case_entries):
        start = case_idx * rows_per_case
        end = start + rows_per_case - 1
        y0 = axes[start][0].get_position().y0
        y1 = axes[end][0].get_position().y0 + axes[end][0].get_position().height
        ymid = 0.5 * (y0 + y1)
        fig.text(0.015, ymid, case_label, ha="left", va="center",
                 fontsize=18, weight="bold", rotation=90)

    # room
    plt.subplots_adjust(left=0.16, right=0.985, bottom=0.10, top=0.95)

    # colorbars (broader)
    cax_field = fig.add_axes([0.20, 0.02, 0.32, 0.020])
    cax_err = fig.add_axes([0.58, 0.02, 0.32, 0.020])
    cbar_f = fig.colorbar(im_field, cax=cax_field, orientation="horizontal")
    cbar_f.set_label("Pressure p", fontsize=14)
    cbar_f.ax.tick_params(direction="out", width=1.0, length=4, labelsize=12)

    cbar_e = fig.colorbar(im_err, cax=cax_err, orientation="horizontal")
    cbar_e.set_label("Relative Error", fontsize=14)
    cbar_e.ax.tick_params(direction="out", width=1.0, length=4, labelsize=12)

    fig.savefig(args.out, bbox_inches="tight")
    print(f"[write] {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
