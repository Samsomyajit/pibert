#!/usr/bin/env python3
"""
Render GT/Pred/AbsErr cubes for multiple experiments/models in a vertical layout.
Uses sample_XXX.npz (GT/Pred channels) extruded to pseudo-3D. Colorbars vertical,
coolwarm colormap for all, labels (a),(b)... outside axes, larger fonts.

Example:
  python plot_volume_grid_from_npz.py \
    --roots results_cylinder_vort300 results_cavity \
    --models PINN PIBERT \
    --sample-idx 0 \
    --stack-depth 24 \
    --out cubes_grid.png
"""
import argparse
import glob
import os
import numpy as np


def load_sample(root, model, idx):
    pat = os.path.join(root, f"{model}_seed*", f"sample_{idx:03d}.npz")
    files = sorted(glob.glob(pat))
    if not files:
        return None
    try:
        d = np.load(files[0])
        gt = d["gt"].astype(np.float32)
        pr = d["pred"].astype(np.float32)
        return gt, pr
    except Exception:
        return None


def magnitude_2d(t):
    if t.ndim != 3:
        raise ValueError(f"expected (C,H,W), got {t.shape}")
    if t.shape[0] >= 2:
        return np.sqrt(t[0] ** 2 + t[1] ** 2)
    return t[0]


def extrude_to_volume(arr2d, depth):
    return np.stack([arr2d] * depth, axis=0)


def make_grid(arr3d):
    import pyvista as pv
    nz, ny, nx = arr3d.shape
    grid_cls = getattr(pv, "UniformGrid", None) or getattr(pv, "ImageData", None)
    if grid_cls is None:
        raise AttributeError("PyVista missing UniformGrid/ImageData; update pyvista.")
    grid = grid_cls()
    grid.dimensions = np.array([nx, ny, nz])
    grid.spacing = (1.0 / max(nx - 1, 1), 1.0 / max(ny - 1, 1), 1.0 / max(nz - 1, 1))
    grid.origin = (0.0, 0.0, 0.0)
    return grid


def parse_opacity(op):
    """PyVista accepts preset strings or iterable; convert numeric to flat list."""
    try:
        val = float(op)
        # create a simple 2-point transfer function at constant opacity
        return [val, val]
    except Exception:
        return op


def main():
    ap = argparse.ArgumentParser(description="Vertical grid of GT/Pred/Err cubes (PyVista).")
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=["PINN", "PIBERT"])
    ap.add_argument("--sample-idx", type=int, default=0)
    ap.add_argument("--stack-depth", type=int, default=2,
                    help="Depth to extrude 2D fields into a thin volume (default: 2 for flat faces).")
    ap.add_argument("--opacity", default=1.0, help="Opacity (float or PyVista preset). Default 1.0 for solid faces.")
    ap.add_argument("--out", default="cubes_grid.png")
    args = ap.parse_args()

    try:
        import pyvista as pv
    except ImportError as e:
        raise SystemExit("pyvista required: pip install pyvista") from e

    pv.set_plot_theme("document")

    cases = []
    for root in args.roots:
        if "cavity" in os.path.basename(root).lower():
            continue
        for model in args.models:
            sample = load_sample(root, model, args.sample_idx)
            if sample is None:
                print(f"[skip] sample_{args.sample_idx:03d}.npz missing for {model} in {root}")
                continue
            cases.append((root, model, sample))

    if not cases:
        raise SystemExit("No samples found.")

    rows = len(cases)
    cols = 3  # GT, Pred, Err
    plotter = pv.Plotter(shape=(rows, cols), window_size=(1800, 700 + 200 * max(rows - 1, 0)), border=False)
    plotter.set_background("white")
    cmap_main = "rainbow_r"  # brighter rainbow for velocity
    cmap_err  = "coolwarm"   # error colormap
    total_panels = rows * cols
    panel_tags = []
    opacity_arg = parse_opacity(args.opacity)

    tag_idx = 0
    for r, (_root, model, (gt, pr)) in enumerate(cases):
        mag_gt = magnitude_2d(gt)
        mag_pr = magnitude_2d(pr)
        mag_err = np.abs(mag_pr - mag_gt)
        depth = max(1, int(args.stack_depth))
        vols = [
            ("GT", extrude_to_volume(mag_gt, depth)),
            ("Pred", extrude_to_volume(mag_pr, depth)),
            ("Abs Err", extrude_to_volume(mag_err, depth)),
        ]
        for c, (label, vol) in enumerate(vols):
            grid = make_grid(vol)
            grid.point_data[label] = vol.ravel(order="F")
            plotter.subplot(r, c)
            surf = grid.extract_surface()
            cmap_use = cmap_err if label == "Abs Err" else cmap_main
            cb_title = f"{model} | {label if label=='Abs Err' else 'Velocity'}"
            plotter.add_mesh(
                surf, scalars=label, cmap=cmap_use, opacity=opacity_arg, show_scalar_bar=False,
                ambient=0.9, diffuse=0.1, specular=0.0,
            )
            plotter.add_scalar_bar(
                title=cb_title,
                n_labels=4,
                vertical=True,
                title_font_size=20,
                label_font_size=16,
                shadow=False,
                position_x=0.88,
                position_y=0.08,
                width=0.08,
                height=0.55,
            )

            plotter.add_axes(line_width=1.4, color="black")

    plotter.link_views()
    plotter.view_vector((1, -0.2, 0.4))
    plotter.show_axes()
    plotter.show(screenshot=args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
