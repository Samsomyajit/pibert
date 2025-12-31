#!/usr/bin/env python3
"""
JCP-style grid plot for vorticity fields (GT / Pred / Relative Error).
Mimics the pressure figure layout but derives vorticity from u/v channels.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors as mcolors


def robust_relerr(gt: np.ndarray, pr: np.ndarray, percentile: float = 95.0, eps_scale: float = 1e-3) -> np.ndarray:
    scale = np.percentile(np.abs(gt), percentile)
    eps = max(eps_scale * scale, 1e-8)
    return np.abs(pr - gt) / (np.abs(gt) + eps)


def compute_vorticity(u: np.ndarray, v: np.ndarray, dx: float = 1.0, dy: float = 1.0) -> np.ndarray:
    dv_dy, dv_dx = np.gradient(v, dy, dx, edge_order=2)
    du_dy, du_dx = np.gradient(u, dy, dx, edge_order=2)
    return dv_dx - du_dy


def parse_case_entry(entry: str) -> Tuple[Path, str]:
    if ":" in entry:
        path_str, label = entry.split(":", 1)
    else:
        path_str, label = entry, Path(entry).name
    return Path(path_str).expanduser().resolve(), label.strip()


def load_trip(path: Path, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    gt = data["gt"].astype(np.float32)
    pr = data["pred"].astype(np.float32)
    if gt.shape[0] < 2 or pr.shape[0] < 2:
        raise ValueError(f"{path} does not contain at least two channels (u,v).")
    gt_vort = compute_vorticity(gt[0], gt[1], dx=dx, dy=dy)
    pr_vort = compute_vorticity(pr[0], pr[1], dx=dx, dy=dy)
    re = robust_relerr(gt_vort, pr_vort)
    return gt_vort, pr_vort, re


def set_axis_style(ax: plt.Axes) -> None:
    ax.set_xlabel("x", fontsize=16)
    ax.set_ylabel("y", fontsize=16)
    ax.tick_params(direction="out", width=1.1, length=5, labelsize=14)
    ax.set_aspect("auto")
    sns.despine(ax=ax, top=True, right=True)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GT/Pred/Relative Error grids for vorticity.")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["results_cavity:Cavity", "results_tube:Tube", "results_cylinder:Cylinder"],
        help="List of 'results_subdir[:label]' entries.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["PINN_seed42", "PIBERT_seed42"],
        help="Model subdirectories to display (order defines rows per case).",
    )
    parser.add_argument("--sample-idx", type=int, default=0, help="Index for sample_{idx:03d}.npz.")
    parser.add_argument("--field_cmap", default="rainbow", help="Colormap for vorticity fields.")
    parser.add_argument("--field-cmap", dest="field_cmap", help=argparse.SUPPRESS)
    parser.add_argument("--err_cmap", default="magma", help="Colormap for relative error.")
    parser.add_argument("--err-cmap", dest="err_cmap", help=argparse.SUPPRESS)
    parser.add_argument("--err-percentile", type=float, default=99.5, help="Percentile clip for relerr.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--width", type=float, default=18.0)
    parser.add_argument("--row-height", type=float, default=3.0)
    parser.add_argument("--model-label-offset", type=float, default=0.02)
    parser.add_argument("--field-label", default="Vorticity ω")
    parser.add_argument("--field-units", default="1/s")
    parser.add_argument("--err-label", default="Relative Error")
    parser.add_argument("--share-field-scale", action="store_true", help="Use global vmin/vmax.")
    parser.add_argument("--dx", type=float, default=1.0, help="Grid spacing along x for ∂/∂x.")
    parser.add_argument("--dy", type=float, default=1.0, help="Grid spacing along y for ∂/∂y.")
    parser.add_argument("--out", default="figure_vorticity_grid.png")
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

    case_entries = [parse_case_entry(entry) for entry in args.cases]
    model_labels = [m.split("_seed")[0] for m in args.models]

    records: List[dict] = []
    for case_idx, (case_path, case_label) in enumerate(case_entries):
        for model_name, model_label in zip(args.models, model_labels):
            sample = case_path / model_name / f"sample_{args.sample_idx:03d}.npz"
            if not sample.exists():
                raise SystemExit(f"Missing {sample}.")
            GT, PR, RE = load_trip(sample, dx=args.dx, dy=args.dy)
            records.append({
                "case_index": case_idx,
                "case_label": case_label,
                "model_label": model_label,
                "gt": GT,
                "pred": PR,
                "err": RE,
            })

    rows = len(records)
    fig_h = args.row_height * rows + 1.2
    fig = plt.figure(figsize=(args.width, fig_h), dpi=args.dpi)
    gs = fig.add_gridspec(rows, 3)
    gs.update(hspace=0.08, wspace=0.03)

    header_titles = ["Ground Truth ω", "Prediction ω", args.err_label]
    for j, title in enumerate(header_titles):
        x = (j + 0.5) / 3
        fig.text(x, 0.987, title, ha="center", va="top", fontsize=24, weight="bold")

    if args.share_field_scale:
        vals = np.concatenate([rec["gt"].ravel() for rec in records])
        vmin_global = float(vals.min())
        vmax_global = float(vals.max())
        case_ranges = {idx: (vmin_global, vmax_global) for idx in range(len(case_entries))}
    else:
        case_ranges = {}
        for rec in records:
            arr = rec["gt"]
            idx = rec["case_index"]
            if idx not in case_ranges:
                case_ranges[idx] = [float(arr.min()), float(arr.max())]
            else:
                case_ranges[idx][0] = min(case_ranges[idx][0], float(arr.min()))
                case_ranges[idx][1] = max(case_ranges[idx][1], float(arr.max()))
        case_ranges = {k: tuple(v) for k, v in case_ranges.items()}
        vmin_global = min(v[0] for v in case_ranges.values())
        vmax_global = max(v[1] for v in case_ranges.values())

    err_vals = np.concatenate([rec["err"].ravel() for rec in records])
    err_max = np.percentile(err_vals, args.err_percentile)
    err_max = max(float(err_max), 1e-6)

    im_field = im_err = None
    axes: List[Tuple[plt.Axes, plt.Axes, plt.Axes]] = []
    field_cbar_ranges: dict[int, Tuple[float, float]] = {}
    for row, rec in enumerate(records):
        ax_gt = fig.add_subplot(gs[row, 0])
        ax_pr = fig.add_subplot(gs[row, 1])
        ax_er = fig.add_subplot(gs[row, 2])
        axes.append((ax_gt, ax_pr, ax_er))

        vmin, vmax = case_ranges[rec["case_index"]]
        field_cbar_ranges.setdefault(rec["case_index"], (vmin, vmax))

        im_field = ax_gt.imshow(
            rec["gt"], origin="lower", cmap=args.field_cmap, vmin=vmin, vmax=vmax, interpolation="bilinear"
        )
        ax_pr.imshow(rec["pred"], origin="lower", cmap=args.field_cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
        im_err = ax_er.imshow(np.clip(rec["err"], 0.0, err_max), origin="lower",
                              cmap=args.err_cmap, vmin=0.0, vmax=err_max, interpolation="bilinear")

        for ax in (ax_gt, ax_pr, ax_er):
            set_axis_style(ax)

        bbox = ax_gt.get_position()
        fig.text(
            bbox.x0 - args.model_label_offset,
            bbox.y0 + bbox.height / 2,
            rec["model_label"],
            ha="right",
            va="center",
            fontsize=17,
            weight="bold",
        )

    rows_per_case = len(args.models)
    for case_idx, (_, case_label) in enumerate(case_entries):
        start = case_idx * rows_per_case
        end = start + rows_per_case - 1
        y0 = axes[start][0].get_position().y0
        y1 = axes[end][0].get_position().y0 + axes[end][0].get_position().height
        ymid = 0.5 * (y0 + y1)
        fig.text(0.015, ymid, case_label, ha="left", va="center",
                 fontsize=18, weight="bold", rotation=90)

    left_margin, right_margin = 0.16, 0.985
    plt.subplots_adjust(left=left_margin, right=right_margin, bottom=0.10, top=0.95)

    field_pad = 0.03
    reserved_err = 0.20
    err_pad = 0.04
    field_total = (right_margin - left_margin) - reserved_err - err_pad
    per_case_width = (field_total - field_pad * (len(case_entries) - 1)) / max(1, len(case_entries))
    per_case_width = max(per_case_width, 0.10)
    label_field = args.field_label
    if args.field_units:
        label_field = f"{label_field} ({args.field_units})"

    for idx, (_, case_label) in enumerate(case_entries):
        vmin, vmax = field_cbar_ranges.get(idx, (vmin_global, vmax_global))
        sm = plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap=args.field_cmap)
        x0 = left_margin + idx * (per_case_width + field_pad)
        cax = fig.add_axes([x0, 0.04, per_case_width, 0.02])
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_label(f"{case_label}: {label_field}", fontsize=12)
        cbar.ax.tick_params(direction="out", width=1.0, length=4, labelsize=10)

    cax_err = fig.add_axes([right_margin - reserved_err, 0.04, reserved_err, 0.02])
    cbar_e = fig.colorbar(im_err, cax=cax_err, orientation="horizontal")
    cbar_e.set_label(args.err_label, fontsize=14)
    cbar_e.ax.tick_params(direction="out", width=1.0, length=4, labelsize=12)

    fig.savefig(args.out, bbox_inches="tight")
    print(f"[write] {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
