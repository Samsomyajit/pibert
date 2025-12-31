#!/usr/bin/env python3
"""
Quick 3D cube rendering for volumetric fields (GT / Pred / Error) using PyVista.

Example:
  python plot_volume_cubes.py \
    --gt gt.npy --pred pred.npy --err err.npy \
    --out cubes.png
"""
import argparse
import os
import numpy as np


def make_grid(arr):
    """Create a PyVista UniformGrid from a 3D numpy array (z, y, x)."""
    import pyvista as pv
    nz, ny, nx = arr.shape
    grid_cls = getattr(pv, "UniformGrid", None) or getattr(pv, "ImageData", None)
    if grid_cls is None:
        raise AttributeError("PyVista does not expose UniformGrid/ImageData; please update pyvista.")
    grid = grid_cls()
    grid.dimensions = np.array([nx, ny, nz])
    grid.spacing = (1.0 / max(nx - 1, 1), 1.0 / max(ny - 1, 1), 1.0 / max(nz - 1, 1))
    grid.origin = (0.0, 0.0, 0.0)
    return grid


def add_volume_cube(plotter, grid, scalars, cmap, opacity="sigmoid_6", label=None):
    """Add a volume-rendered cube."""
    plotter.add_volume(grid, scalars=scalars, cmap=cmap, opacity=opacity, shade=True)
    plotter.add_axes(line_width=2, color="black")
    if label:
        plotter.add_text(label, position="upper_edge", font_size=11, color="black")


def add_slices(plotter, grid, scalars, cmap, n_slices=3, label=None):
    """Add evenly spaced slices along z."""
    slices = grid.slice_along_axis(n=n_slices, axis="z")
    plotter.add_mesh(slices, scalars=scalars, cmap=cmap, show_edges=False)
    plotter.add_axes(line_width=2, color="black")
    if label:
        plotter.add_text(label, position="upper_edge", font_size=11, color="black")


def load_array(path):
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"{path}: expected 3D array (z,y,x), got shape {arr.shape}")
    return arr.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Render GT/Pred/Error cubes (PyVista).")
    ap.add_argument("--gt", required=True, help="Path to GT volume (.npy)")
    ap.add_argument("--pred", help="Path to Pred volume (.npy)")
    ap.add_argument("--err", help="Path to Error volume (.npy). If omitted and pred is set, uses |pred-gt|.")
    ap.add_argument("--out", default="cubes.png", help="Screenshot output (png).")
    ap.add_argument("--opacity", default="sigmoid_6", help="PyVista opacity preset or float/list.")
    ap.add_argument("--slices", type=int, default=3, help="Number of z-slices for slice panel.")
    args = ap.parse_args()

    gt = load_array(args.gt)
    pred = load_array(args.pred) if args.pred else None
    if args.err:
        err = load_array(args.err)
    elif pred is not None:
        err = np.abs(pred - gt)
    else:
        err = None

    import pyvista as pv
    pv.set_plot_theme("document")

    cols = 3 if pred is not None and err is not None else (2 if pred is not None else 1)
    plotter = pv.Plotter(shape=(1, cols), window_size=(1800, 600), border=False)

    # GT
    grid_gt = make_grid(gt)
    grid_gt.point_data["gt"] = gt.ravel(order="F")
    plotter.subplot(0, 0)
    add_volume_cube(plotter, grid_gt, "gt", cmap="turbo", opacity=args.opacity, label="GT")

    col_idx = 1
    if pred is not None:
        grid_pr = make_grid(pred)
        grid_pr.point_data["pred"] = pred.ravel(order="F")
        plotter.subplot(0, col_idx)
        add_volume_cube(plotter, grid_pr, "pred", cmap="turbo", opacity=args.opacity, label="Pred")
        col_idx += 1

    if err is not None:
        grid_err = make_grid(err)
        grid_err.point_data["err"] = err.ravel(order="F")
        plotter.subplot(0, col_idx)
        add_volume_cube(plotter, grid_err, "err", cmap="magma", opacity=args.opacity, label="Abs Err")

    # Optional slice panel on GT for quick interior view
    if cols >= 3:
        plotter.subplot(0, 0)
        add_slices(plotter, grid_gt, "gt", cmap="turbo", n_slices=max(1, args.slices))

    plotter.link_views()
    plotter.show_axes()
    plotter.show(screenshot=args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
