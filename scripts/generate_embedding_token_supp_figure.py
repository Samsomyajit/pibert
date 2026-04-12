#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "FIGURE"
ARTIFACTS_DOC = ROOT / "ARTIFACTS.md"

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


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
            "font.size": 14,
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "figure.titlesize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 1.1,
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


def components_from_channels(channels: np.ndarray) -> dict[str, np.ndarray]:
    u = channels[0]
    v = channels[1]
    mag = np.sqrt(u**2 + v**2)
    return {"u": u, "v": v, "mag": mag}


def load_prediction_sample(run_dir: Path, sample_idx: int) -> tuple[np.ndarray, np.ndarray]:
    prediction_path = run_dir / "predictions.npz"
    require_local_artifact(prediction_path)
    with np.load(prediction_path) as data:
        pred = np.array(data["preds"][sample_idx], copy=True)
        tgt = np.array(data["tgts"][sample_idx], copy=True)
    return pred, tgt


def token_patch_rel_l2(pred: np.ndarray, tgt: np.ndarray, patch_size: int) -> np.ndarray:
    _, height, width = tgt.shape
    grid_y = height // patch_size
    grid_x = width // patch_size
    values = np.zeros((grid_y, grid_x), dtype=np.float64)
    for gy in range(grid_y):
        for gx in range(grid_x):
            ys = slice(gy * patch_size, (gy + 1) * patch_size)
            xs = slice(gx * patch_size, (gx + 1) * patch_size)
            pred_patch = pred[:, ys, xs]
            tgt_patch = tgt[:, ys, xs]
            denom = float(np.linalg.norm(tgt_patch.reshape(-1)))
            numer = float(np.linalg.norm((pred_patch - tgt_patch).reshape(-1)))
            values[gy, gx] = numer / max(denom, 1e-12)
    return values


def checkpoint_diagnostics(ckpt_path: Path) -> dict[str, np.ndarray | float | int]:
    require_local_artifact(ckpt_path)
    obj = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" in obj:
        state_dict = obj["model_state_dict"]
    elif "state_dict" in obj:
        state_dict = obj["state_dict"]
    else:
        state_dict = obj

    weight_pos = state_dict["encoder.fourier.weight_pos"].detach().cpu().numpy()
    weight_neg = state_dict["encoder.fourier.weight_neg"].detach().cpu().numpy()
    fourier_energy = np.sqrt(np.mean(weight_pos**2 + weight_neg**2, axis=(2, 3, 4)))

    gamma_f = float(state_dict["encoder.gamma_f"])
    gamma_w = float(state_dict["encoder.gamma_w"])
    logits = np.array([gamma_f, gamma_w], dtype=np.float64)
    logits -= logits.max()
    mix = np.exp(logits)
    mix /= mix.sum()

    patch_size = int(state_dict["patch_embed.weight"].shape[-1])
    pos_embed = state_dict["pos_embed"].detach().cpu().numpy()[0]
    param_token_norm = float(np.linalg.norm(pos_embed[-1]))
    spatial_tokens = pos_embed.shape[0] - 1
    token_grid = int(round(np.sqrt(spatial_tokens)))

    return {
        "fourier_energy": fourier_energy,
        "mix": mix,
        "patch_size": patch_size,
        "param_token_norm": param_token_norm,
        "token_grid": token_grid,
    }


def add_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    image,
    label: str,
    *,
    width: float = 0.011,
    pad: float = 0.010,
    height_ratio: float = 0.88,
    side: str = "right",
    label_mode: str = "title",
    label_position: str | None = None,
    tick_side: str | None = None,
    tick_labelsize: int = 12,
) -> None:
    bbox = ax.get_position()
    cbar_height = bbox.height * height_ratio
    cbar_y = bbox.y0 + 0.5 * (bbox.height - cbar_height)

    if side == "right":
        cbar_x = bbox.x1 + pad
    elif side == "left":
        cbar_x = bbox.x0 - pad - width
    else:
        raise ValueError(f"Unsupported colorbar side: {side}")

    cax = fig.add_axes([cbar_x, cbar_y, width, cbar_height])
    cbar = fig.colorbar(image, cax=cax)
    if label_mode == "title":
        cbar.ax.set_title(label, fontsize=12, pad=6)
    elif label_mode == "ylabel":
        cbar.set_label(label, fontsize=11, rotation=90, labelpad=6)
        if label_position in {"left", "right"}:
            cbar.ax.yaxis.set_label_position(label_position)
            cbar.ax.yaxis.set_ticks_position(label_position)
    elif label_mode == "none":
        pass
    else:
        raise ValueError(f"Unsupported colorbar label mode: {label_mode}")

    if tick_side in {"left", "right"}:
        cbar.ax.yaxis.set_ticks_position(tick_side)

    cbar.ax.tick_params(labelsize=tick_labelsize)


def draw_token_grid(ax: plt.Axes, field: np.ndarray, patch_size: int, *, use_index_axes: bool) -> None:
    if use_index_axes:
        extent = [0, field.shape[1] - 1, 0, field.shape[0] - 1]
    else:
        extent = [-0.06, 0.16, -0.06, 0.06]
    im = ax.imshow(field, origin="lower", extent=extent, cmap="viridis", interpolation="bilinear", aspect="auto" if use_index_axes else None)

    x0, x1, y0, y1 = extent
    dx = (x1 - x0) / field.shape[1]
    dy = (y1 - y0) / field.shape[0]
    for x_idx in range(0, field.shape[1] + 1, patch_size):
        xpos = x0 + x_idx * dx
        ax.axvline(xpos, color="white", linewidth=0.5, alpha=0.35)
    for y_idx in range(0, field.shape[0] + 1, patch_size):
        ypos = y0 + y_idx * dy
        ax.axhline(ypos, color="white", linewidth=0.5, alpha=0.35)

    if use_index_axes:
        ax.set_xlabel("Flow-direction x position")
        ax.set_ylabel("Cross-stream y position")
    else:
        ax.set_xlabel("Streamwise position x (m)")
        ax.set_ylabel("Cross-stream y (m)")
    return im


def draw_row(
    fig: plt.Figure,
    axes: list[plt.Axes],
    *,
    panel_labels: tuple[str, str, str],
    benchmark_name: str,
    summary_path: Path,
    run_map: dict[str, Path],
    checkpoint_path: Path,
    use_index_axes: bool,
) -> None:
    summary = load_json(summary_path)
    sample_idx = int(summary["sample_idx"])
    baseline = str(summary["main_figure"]["best_baseline_name"])

    pibert_pred, target = load_prediction_sample(run_map["PIBERT"], sample_idx)
    baseline_pred, _ = load_prediction_sample(run_map[baseline], sample_idx)
    diag = checkpoint_diagnostics(checkpoint_path)
    patch_size = int(diag["patch_size"])

    target_field = components_from_channels(target)["mag"]
    pibert_patch = token_patch_rel_l2(pibert_pred, target, patch_size)
    baseline_patch = token_patch_rel_l2(baseline_pred, target, patch_size)
    gain = baseline_patch - pibert_patch
    share_better = float((gain > 0.0).mean())

    ax_field, ax_gain, ax_fourier = axes

    im_field = draw_token_grid(ax_field, target_field, patch_size, use_index_axes=use_index_axes)
    ax_field.set_title("Field + token grid", pad=10, fontsize=13)
    ax_field.text(
        -0.14,
        1.04,
        panel_labels[0],
        transform=ax_field.transAxes,
        ha="left",
        va="bottom",
        fontsize=15,
    )
    grid_note = f"{diag['token_grid']}x{diag['token_grid']} spatial tokens\n+ 1 parameter token"
    ax_field.text(
        0.98,
        0.03,
        grid_note,
        transform=ax_field.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        color="white",
        bbox=dict(facecolor=(0, 0, 0, 0.30), edgecolor="none", pad=3.0),
    )
    add_colorbar(fig, ax_field, im_field, r"$|\mathbf{u}|$", width=0.010, pad=0.010)

    vmax = max(abs(float(gain.min())), abs(float(gain.max())))
    im_gain = ax_gain.imshow(
        gain,
        origin="lower",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    ax_gain.set_title("Token gain", pad=10, fontsize=13)
    ax_gain.text(
        -0.22,
        1.05,
        panel_labels[1],
        transform=ax_gain.transAxes,
        ha="left",
        va="bottom",
        fontsize=15,
    )
    ax_gain.set_xlabel("Token x")
    ax_gain.set_ylabel("Token y", labelpad=2)
    ax_gain.text(
        0.98,
        0.03,
        f"{100.0 * share_better:.1f}% tokens improved\nvs {baseline}",
        transform=ax_gain.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.0,
        bbox=dict(facecolor="white", edgecolor="0.75", pad=2.5),
    )
    gain_cbar_side = "right"
    gain_cbar_label_mode = "none" if benchmark_name == "FSI-real" else "title"
    gain_cbar_label_position = None
    gain_cbar_tick_side = "right" if benchmark_name == "FSI-real" else None
    gain_cbar_pad = 0.008 if benchmark_name == "FSI-real" else 0.010
    gain_cbar_ticksize = 10 if benchmark_name == "FSI-real" else 12
    add_colorbar(
        fig,
        ax_gain,
        im_gain,
        r"$\Delta$ rel. $\ell_2$",
        width=0.010,
        pad=gain_cbar_pad,
        side=gain_cbar_side,
        label_mode=gain_cbar_label_mode,
        label_position=gain_cbar_label_position,
        tick_side=gain_cbar_tick_side,
        tick_labelsize=gain_cbar_ticksize,
    )

    fourier_energy = np.asarray(diag["fourier_energy"])
    im_fourier = ax_fourier.imshow(fourier_energy.T, origin="lower", cmap="magma", interpolation="nearest", aspect="auto")
    ax_fourier.set_title("Fourier energy", pad=10, fontsize=13)
    ax_fourier.text(
        -0.22,
        1.05,
        panel_labels[2],
        transform=ax_fourier.transAxes,
        ha="left",
        va="bottom",
        fontsize=15,
    )
    ax_fourier.set_xlabel(r"Mode $k_x$")
    fourier_ylabel_pad = -4 if benchmark_name == "FSI-real" else 2
    ax_fourier.set_ylabel(r"Mode $k_y$", labelpad=fourier_ylabel_pad)
    mix = np.asarray(diag["mix"])
    ax_fourier.text(
        0.98,
        0.03,
        f"mix: F={mix[0]:.2f}, W={mix[1]:.2f}\nparam-token norm={float(diag['param_token_norm']):.2f}",
        transform=ax_fourier.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        color="white",
        bbox=dict(facecolor=(0, 0, 0, 0.38), edgecolor="none", pad=3.0),
    )
    add_colorbar(fig, ax_fourier, im_fourier, "RMS", width=0.010, pad=0.010)


def main() -> None:
    configure_style()
    FIGURE_DIR.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(11.2, 8.6))
    fig.text(0.08, 0.945, "Cylinder-real", fontsize=18, weight="bold")
    fig.text(0.08, 0.485, "FSI-real", fontsize=18, weight="bold")
    axes = [
        fig.add_axes([0.08, 0.56, 0.24, 0.27]),
        fig.add_axes([0.43, 0.56, 0.17, 0.27]),
        fig.add_axes([0.72, 0.56, 0.17, 0.27]),
        fig.add_axes([0.08, 0.11, 0.24, 0.27]),
        fig.add_axes([0.43, 0.11, 0.17, 0.27]),
        fig.add_axes([0.74, 0.11, 0.16, 0.27]),
    ]

    draw_row(
        fig,
        axes[:3],
        panel_labels=("(a)", "(b)", "(c)"),
        benchmark_name="Cylinder-real",
        summary_path=ROOT / "figures_rpb_final" / "reviewer_multiscale_crosssections_metrics.json",
        run_map=CYLINDER_RUNS,
        checkpoint_path=CYLINDER_RUNS["PIBERT"] / "best.pt",
        use_index_axes=False,
    )
    draw_row(
        fig,
        axes[3:],
        panel_labels=("(d)", "(e)", "(f)"),
        benchmark_name="FSI-real",
        summary_path=ROOT / "figures_rpb_fsi_real_final_v2" / "reviewer_multiscale_crosssections_metrics.json",
        run_map=FSI_RUNS,
        checkpoint_path=FSI_RUNS["PIBERT"] / "best.pt",
        use_index_axes=True,
    )

    fig.savefig(FIGURE_DIR / "supp_embedding_token_diagnostics.pdf", dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
