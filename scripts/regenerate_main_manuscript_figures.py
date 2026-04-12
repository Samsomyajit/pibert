#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Circle
ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "FIGURE"
ARTIFACTS_DOC = ROOT / "ARTIFACTS.md"

MODEL_ORDER = ["PIBERT", "FNO2d", "DeepONet2d", "PITT", "FourierFlow", "PINN"]
MODEL_LABELS = {
    "PIBERT": "PIBERT",
    "FNO2d": "FNO2d",
    "DeepONet2d": "DeepONet2d",
    "PITT": "PITT",
    "FourierFlow": "FourierFlow",
    "PINN": "PINN",
}
MODEL_COLORS = {
    "PIBERT": "#2b5c8a",
    "FNO2d": "#7dbd73",
    "DeepONet2d": "#b086c0",
    "PITT": "#ffa44a",
    "FourierFlow": "#b9835a",
    "PINN": "#b0b0b0",
}
LINE_STYLES = {
    "PIBERT": "--",
    "FNO2d": (0, (1, 1.1)),
    "DeepONet2d": "-",
    "PITT": (0, (4, 2, 1.5, 2)),
    "FourierFlow": "-.",
    "PINN": (0, (5, 4)),
}
COMPONENTS = [
    ("u", r"$u_x$"),
    ("v", r"$u_y$"),
    ("mag", r"$|\mathbf{u}|$"),
    ("omega", r"$\omega$"),
]
DELTA_LABELS = {
    "u": r"$\Delta u_x$",
    "v": r"$\Delta u_y$",
    "mag": r"$\Delta |\mathbf{u}|$",
    "omega": r"$\Delta \omega$",
}

CYLINDER_RUNS = {
    "PIBERT": ROOT / "runs_rpb_final" / "pibert_refine_polish_b",
    "FNO2d": ROOT / "runs_rpb_final" / "fno2d",
    "DeepONet2d": ROOT / "runs_rpb_final" / "deeponet2d",
    "PITT": ROOT / "runs_rpb_final" / "pitt",
    "FourierFlow": ROOT / "runs_rpb_final" / "fourierflow",
    "PINN": ROOT / "runs_rpb_final" / "pinn",
}
FSI_RUNS = {
    "PIBERT": ROOT / "runs_rpb_fsi_real_pibert_v3_fast" / "pibert",
    "FNO2d": ROOT / "runs_rpb_fsi_real_baselines_v3_fast" / "fno2d",
    "DeepONet2d": ROOT / "runs_rpb_fsi_real_baselines_v3_fast" / "deeponet2d",
    "PITT": ROOT / "runs_rpb_fsi_real_baselines_v3_fast" / "pitt",
    "FourierFlow": ROOT / "runs_rpb_fsi_real_baselines_v3_fast" / "fourierflow",
    "PINN": ROOT / "runs_rpb_fsi_real_baselines_v3_fast" / "pinn",
}
CYLINDER_SUMMARY = ROOT / "figures_rpb_final" / "jcp_struct_summary_cylinder_real.json"
FSI_SUMMARY = ROOT / "figures_rpb_fsi_real_final_v2" / "jcp_struct_summary_fsi_real.json"
CYLINDER_MULTISCALE = ROOT / "figures_rpb_final" / "reviewer_multiscale_crosssections_metrics.json"
FSI_MULTISCALE = ROOT / "figures_rpb_fsi_real_final_v2" / "reviewer_multiscale_crosssections_metrics.json"
FSI_PIBERT_PARENT = ROOT / "runs_rpb_fsi_real_pibert_v2" / "pibert"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
            "font.size": 14,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.titlesize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 1.2,
            "axes.grid": False,
            "axes.unicode_minus": False,
        }
    )


def load_json(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def require_local_artifact(path: Path) -> None:
    if path.exists():
        return
    raise FileNotFoundError(
        f"Missing required artifact: {path}. This repository snapshot keeps final figures and summary results only; "
        f"raw predictions/checkpoints are excluded from git. See {ARTIFACTS_DOC.name} for details."
    )


def save_figure(fig: plt.Figure, filename: str) -> None:
    FIGURE_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURE_DIR / filename, dpi=300, facecolor="white")
    plt.close(fig)


def load_prediction_sample(run_dir: Path, sample_idx: int) -> tuple[np.ndarray, np.ndarray]:
    prediction_path = run_dir / "predictions.npz"
    require_local_artifact(prediction_path)
    with np.load(prediction_path) as data:
        pred = np.array(data["preds"][sample_idx], copy=True)
        tgt = np.array(data["tgts"][sample_idx], copy=True)
    return pred, tgt


def load_history(run_dir: Path) -> dict:
    return load_json(run_dir / "train_history.json")


def history_series(run_dir: Path, *, max_epoch: int | None = None) -> list[dict[str, float]]:
    series = list(load_history(run_dir)["val_nmses"])
    if max_epoch is not None:
        series = [point for point in series if int(point["epoch"]) <= max_epoch]
    return series


def extend_history_flat(series: list[dict[str, float]], *, final_epoch: int, step: int = 5) -> list[dict[str, float]]:
    if not series:
        return series
    extended = [dict(point) for point in series]
    last_epoch = int(extended[-1]["epoch"])
    last_nmse = float(extended[-1]["NMSE"])
    next_epoch = ((last_epoch // step) + 1) * step
    while next_epoch <= final_epoch:
        extended.append({"epoch": next_epoch, "NMSE": last_nmse})
        next_epoch += step
    return extended


def components_from_channels(channels: np.ndarray) -> dict[str, np.ndarray]:
    u = channels[0]
    v = channels[1]
    mag = np.sqrt(u**2 + v**2)
    du_dy = np.gradient(u, axis=0)
    dv_dx = np.gradient(v, axis=1)
    omega = dv_dx - du_dy
    return {"u": u, "v": v, "mag": mag, "omega": omega}


def nmse(pred: np.ndarray, tgt: np.ndarray) -> float:
    denom = float(np.mean(tgt**2))
    if denom <= 1e-12:
        return 0.0
    return float(np.mean((pred - tgt) ** 2) / denom)


def value_norm(arrays: list[np.ndarray]) -> tuple[str, Normalize]:
    lo = min(float(a.min()) for a in arrays)
    hi = max(float(a.max()) for a in arrays)
    if lo < 0.0 < hi:
        vmax = max(abs(lo), abs(hi))
        return "RdBu_r", TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    return "viridis", Normalize(vmin=lo, vmax=hi)


def error_norm(arrays: list[np.ndarray]) -> tuple[str, TwoSlopeNorm]:
    vmax = max(float(np.max(np.abs(a))) for a in arrays)
    return "coolwarm", TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)


def add_short_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    image,
    label: str | None = None,
    *,
    width: float = 0.012,
    pad: float = 0.008,
    height_ratio: float = 0.86,
    use_figure_title: bool = False,
    title_gap: float = 0.006,
) -> None:
    bbox = ax.get_position()
    cbar_height = bbox.height * height_ratio
    cbar_y = bbox.y0 + 0.5 * (bbox.height - cbar_height)
    cax = fig.add_axes([bbox.x1 + pad, cbar_y, width, cbar_height])
    cbar = fig.colorbar(image, cax=cax)
    cbar.ax.tick_params(labelsize=14)
    if label:
        if use_figure_title:
            cbar.ax.set_title("")
            cbar.ax.set_ylabel("")
            cbar.ax.text(0.5, 1.0 + title_gap, label, transform=cbar.ax.transAxes, ha="center", va="bottom", fontsize=14)
        else:
            cbar.ax.set_title(label, fontsize=14, pad=6)


def add_short_colorbar_with_extra_gap(
    fig: plt.Figure,
    ax: plt.Axes,
    image,
    label: str | None = None,
    *,
    width: float = 0.012,
    pad: float = 0.008,
    height_ratio: float = 0.86,
    extra_title_gap: float = 12.0,
) -> None:
    bbox = ax.get_position()
    cbar_height = bbox.height * height_ratio
    cbar_y = bbox.y0 + 0.5 * (bbox.height - cbar_height)
    cax = fig.add_axes([bbox.x1 + pad, cbar_y, width, cbar_height])
    cbar = fig.colorbar(image, cax=cax)
    cbar.ax.tick_params(labelsize=14)
    if label:
        cbar.ax.set_title(label, fontsize=14, pad=extra_title_gap)


def add_obstacle(ax: plt.Axes) -> None:
    circle = Circle((31.5, 31.5), radius=5.4, facecolor="0.68", edgecolor="0.5", linewidth=1.0, alpha=0.9)
    ax.add_patch(circle)


def style_image_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("0.15")


def draw_metrics_figure(summary_path: Path, filename: str) -> None:
    summary = load_json(summary_path)
    component_keys = [("u", r"$u_x$"), ("v", r"$u_y$"), ("overall", r"$|\mathbf{u}|$"), ("All", "All")]
    metrics = [
        ("LMAE", "(a)", "LMAE"),
        ("LPCC", "(b)", "LPCC"),
        ("R2", "(c)", r"$R^2$"),
        ("NMSE", "(d)", "NMSE"),
    ]

    fig = plt.figure(figsize=(10, 7.0))
    axes = [
        fig.add_axes([0.11, 0.53, 0.35, 0.28]),
        fig.add_axes([0.57, 0.53, 0.35, 0.28]),
        fig.add_axes([0.11, 0.12, 0.35, 0.28]),
        fig.add_axes([0.57, 0.12, 0.35, 0.28]),
    ]
    x = np.arange(len(component_keys))
    width = 0.13

    for ax, (metric_key, panel_label, y_label) in zip(axes, metrics):
        for idx, model in enumerate(MODEL_ORDER):
            values = [summary["models"][model]["components"][comp][metric_key] for comp, _ in component_keys]
            offset = (idx - (len(MODEL_ORDER) - 1) / 2) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                color=MODEL_COLORS[model],
                edgecolor="0.25",
                linewidth=0.6,
                hatch="//" if model == "PIBERT" else None,
                label=MODEL_LABELS[model],
            )
        ax.set_xticks(x, [label for _, label in component_keys])
        ax.set_ylabel(y_label, labelpad=8)
        ax.text(-0.10, 1.03, panel_label, transform=ax.transAxes, ha="left", va="bottom")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)
        if metric_key == "LPCC":
            ax.set_ylim(0.0, 1.01)
        elif metric_key == "R2":
            ax.set_ylim(0.0, 1.0)
        else:
            ymax = max(bar.get_height() for bar in ax.patches)
            ax.set_ylim(0.0, ymax * 1.12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=True,
        fancybox=False,
        edgecolor="0.2",
        columnspacing=1.4,
        handletextpad=0.8,
        labelspacing=0.6,
        borderpad=0.5,
    )
    save_figure(fig, filename)


def draw_overview_figure(summary_path: Path, run_map: dict[str, Path], filename: str, add_fsi_obstacle: bool) -> None:
    summary = load_json(summary_path)
    sample_idx = int(summary["sample_selection"]["sample_idx"])

    target_components = None
    pred_components: dict[str, dict[str, np.ndarray]] = {}
    for model in MODEL_ORDER:
        pred, tgt = load_prediction_sample(run_map[model], sample_idx)
        pred_components[model] = components_from_channels(pred)
        if target_components is None:
            target_components = components_from_channels(tgt)

    assert target_components is not None
    target_mag = target_components["mag"]
    pred_mags = [pred_components[model]["mag"] for model in MODEL_ORDER]
    errors = [pred_components[model]["mag"] - target_mag for model in MODEL_ORDER]
    cmap, norm = value_norm([target_mag] + pred_mags)
    err_cmap, err_norm = error_norm(errors)

    fig = plt.figure(figsize=(10, 12.0))
    column_titles = [
        "Ground truth",
        "Prediction",
        "Signed error",
    ]
    panel_w = 0.175
    panel_h = panel_w * 10.0 / 12.0
    left = 0.07
    top = 0.952
    row_gap = 0.012
    col_gap = 0.075
    cbar_w = 0.013
    cbar_pad = 0.010
    signed_error_cbar_pad = 0.019
    x_positions = [left + idx * (panel_w + cbar_w + cbar_pad + col_gap) for idx in range(3)]

    for row, model in enumerate(MODEL_ORDER):
        pred_mag = pred_components[model]["mag"]
        err = pred_mag - target_mag
        images = [target_mag, pred_mag, err]
        norms = [norm, norm, err_norm]
        cmaps = [cmap, cmap, err_cmap]

        for col in range(3):
            y0 = top - (row + 1) * panel_h - row * row_gap
            ax = fig.add_axes([x_positions[col], y0, panel_w, panel_h])
            im = ax.imshow(images[col], origin="lower", cmap=cmaps[col], norm=norms[col], interpolation="bilinear")
            style_image_axis(ax)
            if row == 0:
                ax.set_title(column_titles[col], pad=10)
            if col == 0:
                ax.text(-0.24, 0.5, MODEL_LABELS[model], rotation=90, va="center", ha="center", transform=ax.transAxes)
            if col == 2:
                ax.text(
                    0.03,
                    0.97,
                    f"NMSE={nmse(pred_mag, target_mag):.4f}",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=14,
                    bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9, "pad": 1.5},
                )
            if add_fsi_obstacle:
                add_obstacle(ax)
            label = r"$|\mathbf{u}|$" if col < 2 else r"$\Delta |\mathbf{u}|$"
            if col < 2:
                add_short_colorbar(fig, ax, im, label=label, width=cbar_w, pad=cbar_pad)
            else:
                add_short_colorbar(fig, ax, im, label=label, width=cbar_w, pad=signed_error_cbar_pad)

    save_figure(fig, filename)


def draw_pibert_fields_figure(summary_path: Path, run_map: dict[str, Path], filename: str, add_fsi_obstacle: bool) -> None:
    summary = load_json(summary_path)
    sample_idx = int(summary["sample_selection"]["sample_idx"])
    pred, tgt = load_prediction_sample(run_map["PIBERT"], sample_idx)
    pred_components = components_from_channels(pred)
    target_components = components_from_channels(tgt)

    fig = plt.figure(figsize=(10, 10.55))
    column_titles = ["Ground truth", "Prediction", "Signed error"]
    panel_w = 0.195
    panel_h = panel_w * 10.0 / 10.55
    left = 0.07
    top = 0.94
    row_gap = 0.038
    col_gap = 0.070
    cbar_w = 0.013
    cbar_pad = 0.010
    x_positions = [left + idx * (panel_w + cbar_w + cbar_pad + col_gap) for idx in range(3)]

    for row, (component_key, component_label) in enumerate(COMPONENTS):
        target_field = target_components[component_key]
        pred_field = pred_components[component_key]
        err_field = pred_field - target_field
        cmap, norm = value_norm([target_field, pred_field])
        err_cmap, err_norm = error_norm([err_field])

        for col, (field, field_cmap, field_norm) in enumerate(
            [
                (target_field, cmap, norm),
                (pred_field, cmap, norm),
                (err_field, err_cmap, err_norm),
            ]
        ):
            y0 = top - (row + 1) * panel_h - row * row_gap
            ax = fig.add_axes([x_positions[col], y0, panel_w, panel_h])
            im = ax.imshow(field, origin="lower", cmap=field_cmap, norm=field_norm, interpolation="bilinear")
            style_image_axis(ax)
            if row == 0:
                ax.set_title(column_titles[col], pad=10)
            if col == 0:
                ax.text(-0.24, 0.5, component_label, rotation=90, va="center", ha="center", transform=ax.transAxes)
            if add_fsi_obstacle:
                add_obstacle(ax)
            if col < 2:
                add_short_colorbar(fig, ax, im, label=component_label, width=cbar_w, pad=cbar_pad)
            else:
                add_short_colorbar_with_extra_gap(fig, ax, im, label=DELTA_LABELS[component_key], width=cbar_w, pad=cbar_pad)

    save_figure(fig, filename)


def draw_multiscale_figure(summary_path: Path, run_map: dict[str, Path], filename: str, *, use_index_axes: bool = False) -> None:
    summary = load_json(summary_path)
    sample_idx = int(summary["sample_idx"])
    slice_columns = [int(col) for col in summary["main_figure"]["slice_columns"]]
    slice_short_names = ["Near", "Mid", "Far"]
    best_baseline = summary["main_figure"]["best_baseline_name"]
    profile_models = summary["main_figure"].get("shown_profile_models", ["PIBERT", best_baseline])
    if "PIBERT" not in profile_models:
        profile_models = ["PIBERT", best_baseline]
    baseline_profile_model = next((model for model in profile_models if model != "PIBERT"), best_baseline)
    wake_row = int(summary["main_figure"]["wake_line_statistics"]["line_row"])

    samples: dict[str, dict[str, np.ndarray]] = {}
    target_components = None
    for model in ["PIBERT", best_baseline, "FNO2d", "FourierFlow", "DeepONet2d", "PITT", "PINN"]:
        pred, tgt = load_prediction_sample(run_map[model], sample_idx)
        samples[model] = components_from_channels(pred)
        if target_components is None:
            target_components = components_from_channels(tgt)

    assert target_components is not None
    field = target_components["mag"]
    if use_index_axes:
        x_coords = np.arange(field.shape[1])
        y_coords = np.arange(field.shape[0])
        field_xlabel = "Flow-direction x position"
        field_ylabel = "Cross-stream y position"
        slice_xlabel = "Cross-stream y position"
        wake_xlabel = "Flow-direction (x)"
        wake_note = f"y = {wake_row}"
        wake_note_fontsize = 11
        fig = plt.figure(figsize=(10, 7.9))
        ax_field = fig.add_axes([0.09, 0.63, 0.46, 0.24])
        ax_bar = fig.add_axes([0.715, 0.62, 0.225, 0.25])
        ax_near = fig.add_axes([0.08, 0.18, 0.18, 0.22])
        ax_mid = fig.add_axes([0.34, 0.18, 0.18, 0.22])
        ax_far = fig.add_axes([0.60, 0.18, 0.18, 0.22])
        ax_wake_vel = fig.add_axes([0.845, 0.300, 0.125, 0.095])
        ax_wake_vort = fig.add_axes([0.845, 0.110, 0.125, 0.095])
    else:
        x_coords = np.linspace(-0.06, 0.16, field.shape[1])
        y_coords = np.linspace(-0.06, 0.06, field.shape[0])
        field_xlabel = "Streamwise position x (m)"
        field_ylabel = "Cross-stream y (m)"
        slice_xlabel = "Cross-stream y (m)"
        wake_xlabel = "Streamwise x (m)"
        wake_note = f"y = {y_coords[wake_row]:.3f} m"
        wake_note_fontsize = 11
        fig = plt.figure(figsize=(10, 8.8))
        ax_field = fig.add_axes([0.10, 0.60, 0.36, 0.28])
        ax_bar = fig.add_axes([0.625, 0.61, 0.325, 0.255])
        ax_near = fig.add_axes([0.08, 0.17, 0.16, 0.22])
        ax_mid = fig.add_axes([0.33, 0.17, 0.16, 0.22])
        ax_far = fig.add_axes([0.58, 0.17, 0.16, 0.22])
        ax_wake_vel = fig.add_axes([0.812, 0.305, 0.17, 0.105])
        ax_wake_vort = fig.add_axes([0.812, 0.130, 0.17, 0.105])

    gt_vorticity = target_components["omega"]
    pibert_vorticity = samples["PIBERT"]["omega"]
    baseline_vorticity = samples[baseline_profile_model]["omega"]

    im = ax_field.imshow(
        field,
        origin="lower",
        extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]],
        cmap="turbo",
        interpolation="bilinear",
        aspect="auto" if use_index_axes else None,
    )
    for x_val, label in zip([x_coords[col] for col in slice_columns], ["Near-body", "Wake-core", "Far-wake"]):
        ax_field.axvline(x_val, color="white", linewidth=2.0, alpha=0.9)
        label_y = y_coords[1] if use_index_axes else y_coords[2]
        ax_field.text(x_val, label_y, label, color="white", ha="center", va="bottom", fontsize=14, weight="bold")
    ax_field.axhline(y_coords[wake_row], color="#d62728", linewidth=2.0)
    ax_field.set_xlabel(field_xlabel)
    ax_field.set_ylabel(field_ylabel)
    panel_a_x = -0.075 if use_index_axes else -0.14
    ax_field.text(panel_a_x, 1.04, "(a)", transform=ax_field.transAxes, fontsize=16)
    add_short_colorbar(
        fig,
        ax_field,
        im,
        label=r"$|\mathbf{u}|$",
        width=0.012 if use_index_axes else 0.014,
        pad=0.009 if use_index_axes else 0.012,
        height_ratio=0.9,
        use_figure_title=not use_index_axes,
    )

    slice_metrics = summary["main_figure"]["slice_metrics"]
    x = np.arange(len(slice_short_names))
    width = 0.12
    for idx, model in enumerate(MODEL_ORDER):
        values = [
            slice_metrics[model]["Large-scale / near-body"]["rel_l2"],
            slice_metrics[model]["Intermediate / wake-core"]["rel_l2"],
            slice_metrics[model]["Fine-scale / far-wake"]["rel_l2"],
        ]
        offset = (idx - (len(MODEL_ORDER) - 1) / 2) * width
        ax_bar.bar(
            x + offset,
            values,
            width=width,
            color=MODEL_COLORS[model],
            edgecolor="0.25",
            linewidth=0.6,
            hatch="//" if model == "PIBERT" else None,
            label=MODEL_LABELS[model],
        )
    ax_bar.set_xticks(x, slice_short_names)
    ax_bar.set_ylabel(r"Rel. $\ell_2$ error")
    ax_bar.set_title("Slice error")
    ax_bar.grid(axis="y", linestyle="--", alpha=0.25)
    ax_bar.set_axisbelow(True)
    if use_index_axes:
        ax_bar.yaxis.labelpad = 0
    ax_bar.text(-0.12 if use_index_axes else -0.13, 1.04, "(b)", transform=ax_bar.transAxes, fontsize=16)

    profile_axes = [ax_near, ax_mid, ax_far]
    panel_labels = ["(c)", "(d)", "(e)"]
    panel_titles = [
        "Large-scale /\nnear-body",
        "Intermediate /\nwake-core",
        "Fine-scale /\nfar-wake",
    ]
    for ax, label, title, col in zip(profile_axes, panel_labels, panel_titles, slice_columns):
        ax.plot(y_coords, target_components["mag"][:, col], color="0.35", linewidth=2.0, label="GT")
        ax.plot(y_coords, samples["PIBERT"]["mag"][:, col], color=MODEL_COLORS["PIBERT"], linewidth=2.0, label="PIBERT")
        ax.plot(
            y_coords,
            samples[baseline_profile_model]["mag"][:, col],
            color="#d62728",
            linestyle="--",
            linewidth=1.9,
            label=baseline_profile_model,
        )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(slice_xlabel)
        ax.grid(True, linestyle="--", alpha=0.2)
        ax.text(-0.16, 1.05, label, transform=ax.transAxes, fontsize=16)
    ax_near.set_ylabel(r"$|\mathbf{u}|$")

    ax_wake_vel.plot(x_coords, target_components["mag"][wake_row], color="0.35", linewidth=2.0)
    ax_wake_vel.plot(x_coords, samples["PIBERT"]["mag"][wake_row], color=MODEL_COLORS["PIBERT"], linewidth=2.0)
    ax_wake_vel.plot(x_coords, samples[baseline_profile_model]["mag"][wake_row], color="#d62728", linestyle="--", linewidth=1.9)
    ax_wake_vel.set_title("Velocity", fontsize=14, pad=7)
    ax_wake_vel.grid(True, linestyle="--", alpha=0.2)
    ax_wake_vel.tick_params(labelbottom=False)
    ax_wake_vel.text(-0.48 if use_index_axes else -0.36, 1.10, "(f)", transform=ax_wake_vel.transAxes, fontsize=16)
    ax_wake_vel.text(
        0.03,
        0.95,
        wake_note,
        transform=ax_wake_vel.transAxes,
        fontsize=wake_note_fontsize,
        ha="left",
        va="top",
    )

    ax_wake_vort.plot(x_coords, gt_vorticity[wake_row], color="0.35", linewidth=2.0)
    ax_wake_vort.plot(x_coords, pibert_vorticity[wake_row], color=MODEL_COLORS["PIBERT"], linewidth=2.0)
    ax_wake_vort.plot(x_coords, baseline_vorticity[wake_row], color="#d62728", linestyle="--", linewidth=1.9)
    ax_wake_vort.set_title("Vorticity", fontsize=14, pad=9)
    ax_wake_vort.set_xlabel(wake_xlabel, fontsize=12 if use_index_axes else 14, labelpad=1.5)
    ax_wake_vort.grid(True, linestyle="--", alpha=0.2)
    ax_wake_vort.text(-0.48 if use_index_axes else -0.36, 1.10, "(g)", transform=ax_wake_vort.transAxes, fontsize=16)

    line_handles = [
        plt.Line2D([0], [0], color="0.35", linewidth=2.0, label="GT"),
        plt.Line2D([0], [0], color=MODEL_COLORS["PIBERT"], linewidth=2.0, label="PIBERT"),
        plt.Line2D([0], [0], color="#d62728", linestyle="--", linewidth=1.9, label=baseline_profile_model),
    ]
    bar_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=MODEL_COLORS[model], edgecolor="0.25", linewidth=0.6, hatch="//" if model == "PIBERT" else None, label=MODEL_LABELS[model])
        for model in MODEL_ORDER
    ]
    fig.legend(
        bar_handles,
        [MODEL_LABELS[model] for model in MODEL_ORDER],
        loc="upper center",
        bbox_to_anchor=(0.62, 1.005),
        ncol=3,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.0 if use_index_axes else 0.8,
        handletextpad=0.6,
    )
    fig.legend(
        handles=line_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.014 if use_index_axes else 0.028),
        ncol=3,
        frameon=True,
        fancybox=False,
        edgecolor="0.25",
        columnspacing=1.5 if use_index_axes else 1.0,
        handletextpad=0.8,
    )
    save_figure(fig, filename)


def draw_training_convergence(filename: str) -> None:
    histories = {
        **{model: history_series(run_dir) for model, run_dir in FSI_RUNS.items() if model != "PIBERT"},
        # The retained FSI PIBERT run is a short warm-start continuation from v2.
        # Plot the parent v2 optimization trace over the common comparison window.
        "PIBERT": history_series(FSI_PIBERT_PARENT, max_epoch=180),
    }
    histories["PINN"] = extend_history_flat(histories["PINN"], final_epoch=180, step=5)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    panel_specs = [
        ("NMSE", "Validation NMSE", "(a)"),
        ("REL", "Relative error", "(b)"),
    ]
    pinn_line_color = "#222222"
    pinn_line_style = (0, (8, 2.5, 1.6, 2.5))
    for ax, (metric_key, ylabel, panel_label) in zip(axes, panel_specs):
        for model in MODEL_ORDER:
            series = histories[model]
            epochs = [point["epoch"] for point in series]
            if metric_key == "NMSE":
                values = [point["NMSE"] for point in series]
            else:
                values = [float(np.sqrt(point["NMSE"])) for point in series]

            line_kwargs = {
                "color": MODEL_COLORS[model],
                "linestyle": LINE_STYLES[model],
                "linewidth": 2.6,
                "label": MODEL_LABELS[model],
            }
            if model == "PINN":
                line_kwargs.update(
                    {
                        "color": pinn_line_color,
                        "linestyle": pinn_line_style,
                        "linewidth": 2.4,
                        "zorder": 4,
                    }
                )
            ax.plot(epochs, values, **line_kwargs)

        ax.set_yscale("log")
        ax.set_xlim(-2, 182)
        ax.set_xlabel("Fine-tuning epoch")
        ax.set_ylabel(ylabel)
        ax.text(-0.02, 1.03, panel_label, transform=ax.transAxes, ha="left", va="bottom")
        ax.grid(which="major", linestyle=(0, (4, 4)), linewidth=0.8, alpha=0.45, color="0.65")
        ax.grid(which="minor", axis="y", linestyle=":", linewidth=0.7, alpha=0.45, color="0.55")
        ax.tick_params(which="both", direction="in", top=True, right=True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=True, fancybox=False, edgecolor="0.2")
    fig.subplots_adjust(left=0.09, right=0.985, top=0.74, bottom=0.15, wspace=0.25)
    save_figure(fig, filename)


def main() -> None:
    configure_style()
    draw_training_convergence("main_training_convergence.pdf")
    draw_metrics_figure(CYLINDER_SUMMARY, "main_cylinder_metrics.pdf")
    draw_overview_figure(CYLINDER_SUMMARY, CYLINDER_RUNS, "main_cylinder_overview.pdf", add_fsi_obstacle=False)
    draw_pibert_fields_figure(CYLINDER_SUMMARY, CYLINDER_RUNS, "main_cylinder_pibert_fields.pdf", add_fsi_obstacle=False)
    draw_multiscale_figure(CYLINDER_MULTISCALE, CYLINDER_RUNS, "main_cylinder_multiscale.pdf")
    draw_metrics_figure(FSI_SUMMARY, "main_fsi_metrics.pdf")
    draw_overview_figure(FSI_SUMMARY, FSI_RUNS, "main_fsi_overview.pdf", add_fsi_obstacle=True)
    draw_pibert_fields_figure(FSI_SUMMARY, FSI_RUNS, "main_fsi_pibert_fields.pdf", add_fsi_obstacle=True)
    draw_multiscale_figure(FSI_MULTISCALE, FSI_RUNS, "main_fsi_multiscale.pdf", use_index_axes=True)


if __name__ == "__main__":
    main()
